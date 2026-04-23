from hyperparams import hp
from model import AR_TTS
from utils import encode_tokens
from torch import nn
from torch.nn import functional as F
import torch
import torchaudio
from scheduler import NoamScheduler
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import collateFn
from sampler import DualBucketBatchSampler



train_dataset = torch.load(hp.TRAIN_DATASET_PATH, weights_only=False) 
test_dataset = torch.load(hp.TEST_DATASET_PATH, weights_only=False) 

bucket_size_text = 60
bucket_size_mel = 300

train_text_lengths = [item[1] for item in train_dataset]
train_mel_lengths = [item[3] for item in train_dataset]

test_text_lengths = [item[1] for item in test_dataset]
test_mel_lengths = [item[3] for item in test_dataset]

train_bucket_sampler = DualBucketBatchSampler(
        text_lengths=train_text_lengths,
        mel_lengths=train_mel_lengths,
        batch_size=hp.batch_size,
        bucket_size_text=bucket_size_text,
        bucket_size_mel=bucket_size_mel,
        shuffle=True,
        drop_last=True
    )

test_bucket_sampler = DualBucketBatchSampler(
        text_lengths=test_text_lengths,
        mel_lengths=test_mel_lengths,
        batch_size=hp.batch_size,
        bucket_size_text=bucket_size_text,
        bucket_size_mel=bucket_size_mel,
        shuffle=True,
        drop_last=True
    )

train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_bucket_sampler,
        collate_fn=collateFn,
        pin_memory=True,
        num_workers=0
    )

test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_bucket_sampler,
        collate_fn=collateFn,
        pin_memory=True,
        num_workers=0
    )



class TrainModel():
    def __init__(self, model, optimizer, scheduler, scaler, melLossFn, postMelLossFn, stopLossFn, best_eval, current_step, trainLoader, testLoader):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.melLossFn = melLossFn
        self.postMelLossFn = postMelLossFn
        self.stopLossFn = stopLossFn
        self.best_eval = best_eval
        self.current_step = current_step
        self.scaler = scaler

        self.trainLoader = trainLoader
        self.testLoader = testLoader

        self.model = self.model.to(hp.device)

    def _attention_entropy(self, attn_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Computes the average entropy of attention weights.
        Args:
            attn_weights: Tensor of shape [B, H, Q_T, K_T] (after softmax)
        Returns:
            Scalar tensor: mean entropy over all heads and time steps
        """
        # Clamp for numerical stability
        attn_clamped = attn_weights.clamp(min=eps)
        
        # Entropy: -sum(p * log(p))
        entropy = -attn_clamped * torch.log(attn_clamped)
        entropy = entropy.sum(dim=-1)  # Sum over K_T → shape [B, H, Q_T]
        
        # Mean over batch, heads, and time steps
        return entropy.mean()

    def _guided_attention_loss(self, text_len, mel_len, attn_w, sigma):

        B, H, seq_mel, seq_text = attn_w.shape
        total_loss = 0.0

        for batch in range(B):
            ml = mel_len[batch]
            tl = text_len[batch]

            a = torch.arange(ml, device=hp.device).unsqueeze(1) / ml
            b = torch.arange(tl, device=hp.device).unsqueeze(0) / tl

            GA_mask = 1.0 - torch.exp(-((a - b) ** 2) / (2 * sigma ** 2))
            attn_real = (attn_w[batch, : , :ml, :tl])

            total_loss += (attn_real * GA_mask).mean()

        return total_loss / B
    """
        _, _, seq_mel, seq_text = attn_w.shape   # [batch, n_head, mel_seq, text_seq]

        a = torch.arange(seq_mel, device=hp.device).unsqueeze(1) / seq_mel
        b = torch.arange(seq_text, device=hp.device).unsqueeze(0) / seq_text
        GA_mask = 1.0 - torch.exp(-((a - b) ** 2) / (2 * sigma ** 2)) # [1, 1, T_tgt, T_src]
        GA_mask = GA_mask.unsqueeze(0).unsqueeze(0)

        loss = (attn_w * GA_mask).mean()
        return loss
    """
    def _calculateLoss(self, mel_pred, post_mel_pred, mel_true, stop_pred, stop_true):
        B, T, _ = mel_true.shape


        # Create matching shape for targets. Where 0 = continue, 1 = stop
        range_t = torch.arange(T, device=hp.device).unsqueeze(0).expand(B, T) 
        stop_true = stop_true.unsqueeze(1).to(hp.device)  # [B, 1]
        stop_target = (range_t >= (stop_true - 1)).float()

        stop_pred = torch.clamp(stop_pred, min=-40.0, max=40.0)

        # Get raw per-element stop loss (shape [B, T])
        stop_loss_raw = F.binary_cross_entropy_with_logits(stop_pred, stop_target, reduction='none')

        # Mask all padding values (False after stop_true)
        stop_mask = (range_t <= stop_true)
        stop_loss_masked = stop_loss_raw[stop_mask]
        
        stop_loss = stop_loss_masked.mean()


        mel_loss_raw = F.huber_loss(mel_pred, mel_true, reduction='none')

        mel_mask = (mel_true == 0).all(dim=-1)
        mel_loss_masked = mel_loss_raw[~mel_mask]

        mel_loss = mel_loss_masked.mean()

        post_mel_loss_raw = F.huber_loss(post_mel_pred, mel_true, reduction='none')
        post_mel_loss_masked = post_mel_loss_raw[~mel_mask]

        post_mel_loss = post_mel_loss_masked.mean()

  #      mel_loss = self.melLossFn(mel_pred, mel_true)
  #      post_mel_loss = self.postMelLossFn(post_mel_pred, mel_true)

 #       print(f"_calculateLoss: | mel_loss: {mel_loss}, post_mel_loss: {post_mel_loss}, stop_loss: {stop_loss}")

        return mel_loss, post_mel_loss, stop_loss


    def saveModel(self, mel_loss, postmel_loss, stop_loss, GA_loss, GE_loss, GA_gamma, GE_gamma, stop_gamma, p_teacher, learning_rate, current_step, attn_weights, saving):
        # Add here statistics logging too!
        # total loss, effective total loss
        total_loss = mel_loss + postmel_loss + stop_loss + GA_loss + GE_loss
        effective_loss = mel_loss + postmel_loss + (stop_loss * stop_gamma) + (GA_loss * GA_gamma) + (GE_loss * GE_gamma)
        # effective step

        # Effective step resets every epoch, make it so it does not reset.
        # include effective step inside of save; maybe over epoch.
        # write data to a file for later plotting.

        # maybe completely get rid of testing set and testing loader.
        # probably better to try and validate by using actual autoregressive loop -> compare to ground truth
        # and that can be easily done during/from training data. eg. self.autoRegressiveSample()

        print(f"Step: {current_step} | Learning Rate: {learning_rate:.7f} | p_teacher: {p_teacher:.4f} | Guided Attention {GA_loss:.4f} | Guided Entropy {GE_loss:.4f}\nMel: {mel_loss:.3f} | Postnet: {postmel_loss:.3f} | Stop: {stop_loss:.3f}\nTotal Loss {total_loss:.4f} | Total Effective Loss {effective_loss:.4f}")
        self._plot_attention(attn_list=attn_weights, batch_idx=0)
        if saving:

           # self._plot_attention(attn_list=attn_weights, batch_idx=0)

            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'scaler': self.scaler.state_dict(),
                'best_eval': self.best_eval,
                'epoch': self.current_step,
            }, f'{hp.MODEL_SAVE_ROOT}checkpoint.pth')

            with open(f"{hp.MODEL_SAVE_ROOT}/checkpoint_history.csv", "a") as f:
                """
                Current step | Learning rate | p_teacher | GA_loss | Mel_loss | Postnet_loss | stop_loss | total_loss | effective loss
                """
                f.write(f"{current_step};{learning_rate};{p_teacher};{GA_loss};{GE_loss};{mel_loss};{postmel_loss};{stop_loss};{total_loss};{effective_loss}\n")
                f.close()
        else:
            print("NaNs detected! NOT SAVING")

    def plotData(self, epoch, train_loss, test_loss, learning_rate):
        pass

    def _plot_attention(self, attn_list, batch_idx=0):
        """
        Plot cross-attention maps.
        Args:
            attn_list: list of tensors [B, H, T_dec, T_enc], one per layer
            batch_idx: which item in batch to plot
        """
        n_layers = len(attn_list)
        n_heads = attn_list[0].shape[1]
        

        fig, axes = plt.subplots(
            n_layers, n_heads,
            figsize=(4 * n_heads, 4 * n_layers),
            squeeze=False
        )

        for l, attn in enumerate(attn_list):
            attn = attn.cpu()
            for h in range(n_heads):
                ax = axes[l, h]
                ax.imshow(attn[batch_idx, h], aspect="auto", origin="lower")
                if l == 0:
                    ax.set_title(f"Head {h}")
                if h == 0:
                    ax.set_ylabel(f"Layer {l}")
                ax.set_xlabel("Text pos")

        plt.suptitle(f"Cross-attention (step {self.current_step})")
        save_path = f"{hp.MODEL_SAVE_ROOT}/ch_attn_head_plots/{self.current_step}.png"
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path)
        plt.close(fig)


    def check_for_nan_params(self, model):
        nan_params = []
        for name, param in model.named_parameters():
            if param is not None:
                if torch.isnan(param).any():
                    nan_params.append(name)
        return nan_params

    def check_for_nan_grads(self, model):
        nan_grads = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_grads.append(name)
        return nan_grads
    
    def smartSampling(self, text, GT, p_teacher, window_size, p_continue):

        B, T, M = GT.shape # batch, seq, mel

        sample_mask = torch.zeros((B, window_size), dtype=torch.bool, device=hp.device)

        for b in range(B):
            for t in range(window_size):
                if t == 0:
                    sample_mask[b, t] = False
                if sample_mask[b, t - 1]:
                    sample_mask[b, t] = torch.rand(1, device=hp.device) < p_continue
                else: 
                    sample_mask[b, t] = torch.rand(1, device=hp.device) < p_teacher

        if not sample_mask.any():
            return GT
        
        furthest_idx = sample_mask.any(dim=0).nonzero().max().item() + 1
        mel = torch.full((hp.batch_size, 1, hp.n_mel_bin), fill_value=hp.start_frame_mel_fill_value, dtype=torch.float32, device=hp.device)

        with torch.no_grad():
            with autocast(device_type=hp.device):
                for _ in range(furthest_idx):
                    linear_mel = self.model(text, mel, sampling=True)

                    mel_frame = linear_mel[:, -1, :] 
                    mel_frame = mel_frame.unsqueeze(1) # [B, 1, BIN]
                    mel_frame = torch.clamp(mel_frame, -11.5, 4.0)

                    mel = torch.cat([mel, mel_frame], dim=1).detach()
        self.model.train()

        mel_sampled_padded = torch.zeros_like(GT)
        mel_sampled_padded[:, :mel.shape[1], :] = mel

        sample_mask_padded = sample_mask.unsqueeze(-1).expand(-1, -1, M)


#        start_frame_equal = torch.allclose(GT[:, 0, :], mel[:, 0, :])
#        print(start_frame_equal)

        GT[:, :window_size, :] = torch.where(sample_mask_padded, mel_sampled_padded[:, :window_size, :], GT[:, :window_size, :])

        return GT


    def autoRegressiveSample(self, text, GT_mel_max_len):
        
        # Create starting frame
        mel = torch.full((hp.batch_size, 1, hp.n_mel_bin), fill_value=hp.start_frame_mel_fill_value, dtype=torch.float32, device=hp.device)

        with torch.no_grad():
            with autocast(device_type=hp.device):
                for _ in range(GT_mel_max_len - 1):
                    linear_mel = self.model(text, mel, sampling=True)

                    mel_frame = linear_mel[:, -1, :] 
                    mel_frame = mel_frame.unsqueeze(1) # [B, 1, BIN]

                    mel = torch.cat([mel, mel_frame], dim=1).detach()

        self.model.train()
        return mel.float()
    


    def trainingLoop(self, dataloader):
        self.model.train()

        learning_rates = 0.0
        stop_losses = 0.0
        mel_losses = 0.0
        postmel_losses = 0.0
        GA_losses = 0.0
        GE_losses = 0.0
        effective_steps = 0
        gammas = 0.0

        stop_gamma = scheduler.get_gamma(speedup=1.2)
        GA_gamma = scheduler.get_gamma(decay_factor=10)
        GE_gamma = scheduler.get_gamma(decay_factor=2)
        self.optimizer.zero_grad()
        save = True

        for batch, (text, text_len, mel, mel_len) in enumerate(dataloader):
            x, y, mel_len = text.to(hp.device), mel.to(hp.device), mel_len.to(hp.device)
            x = x.type(torch.long)

            # Apply Start Of Sequence Frame to MEL
            start_frame = torch.full((hp.batch_size, 1, hp.n_mel_bin), fill_value=hp.start_frame_mel_fill_value, dtype=torch.float32, device=hp.device)
            y_in = torch.cat([start_frame, y[:, :-1, :]], dim=1) # Shift MEL right by one frame

            padding_mask_mel = (y_in == 0).all(dim=-1)   # Padding value: 0


            p_teacher, sampling_window_size, p_continue = scheduler.get_sampling_params(seq_len=y_in.shape[1])
            if p_teacher < 1 and GA_gamma == 0.0:
                # print(f"p_teacher: {p_teacher} . p_continue: {p_continue} . Swindow size: {sampling_window_size}")
                
                y_in = self.smartSampling(
                    text=x,
                    GT=y_in,
                    p_teacher=p_teacher,
                    window_size=sampling_window_size,
                    p_continue=p_continue
                )

            if stop_gamma < 0.3:
                stop_gamma = 0.3


            with autocast(device_type=hp.device):
                melResult, postnetResult, stopResult, guided_attn_weights = self.model(x, y_in, padding_mask_mel=padding_mask_mel)
                stopResult = stopResult.squeeze(-1)

            melResult = melResult.float()
            postnetResult = postnetResult.float()
            stopResult = stopResult.float()

            GA_loss = 0.0
            GE_loss = 0.0
            if guided_attn_weights != None:
                cross_attn_weights = guided_attn_weights[0]
                self_attn_entropy = guided_attn_weights[1]



                for i, w in enumerate(cross_attn_weights):
                    if i < hp.dec_n_block and i < hp.enc_n_block:
                        GA_loss += self._guided_attention_loss(text_len=text_len,
                                                                mel_len=mel_len,
                                                                attn_w=w, 
                                                                sigma=hp.sigma
                                                                )

                for i, e in enumerate(self_attn_entropy):
                    if i < hp.dec_n_block and i < hp.enc_n_block:
                        GE_loss += self._attention_entropy(e) * hp.entropy_lambda

            melLoss, postMelLoss, stopLoss = self._calculateLoss(
                mel_pred=melResult,
                post_mel_pred=postnetResult,
                mel_true=y,
                stop_pred=stopResult,
                stop_true=mel_len
            )


            loss = (melLoss + postMelLoss + (stopLoss * stop_gamma) + (GA_loss * GA_gamma) + (GE_loss * GE_gamma)) / hp.gradient_accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN or Inf detected in loss before backward!")
                save = False

            self.scaler.scale(loss).backward()

            if ((batch+1) % hp.gradient_accumulation_steps == 0) or ((batch+1) == len(dataloader)) or hp.gradient_accumulation_steps == 1:
                
                self.scaler.unscale_(self.optimizer)    # Manually unscale grads for optimizer

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    # Clip gradients

                if self.current_step % 1000 == 0 or ((batch+1) == len(dataloader)) or self.current_step == 1:
                    nan_params = self.check_for_nan_params(self.model)
                    if nan_params:
                        print(f"Step: {effective_steps} | NaN detected in parameters: {nan_params}")
                        save = False

                    nan_grads = self.check_for_nan_grads(self.model)
                    if nan_grads:
                        print(f"Step: {effective_steps} | NaN detected in gradients: {nan_grads}")
                        save = False

                self.scaler.step(self.optimizer) # Automatically does NOT unscale grads if .unscale_ is called already! More than one unscale -> runtime error

                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

                mel_losses += melLoss.item()
                postmel_losses += postMelLoss.item()
                stop_losses += stopLoss.item()
                GA_losses += GA_loss.item()
                GE_losses += GE_loss.item()
                learning_rates += self.scheduler.get_last_lr()[0]
                gammas += (GE_gamma + GA_gamma + stop_gamma) / 3

                stop_gamma = scheduler.get_gamma(speedup=1.2)
                GA_gamma = scheduler.get_gamma(decay_factor=10)
                GE_gamma = scheduler.get_gamma(decay_factor=2)
                p_teacher, sampling_window_size, p_continue = scheduler.get_sampling_params(seq_len=y_in.shape[1])

                self.current_step += 1
                effective_steps += 1

                # NOTE Maybe implement scheduled gradient accumulation. eg. first x steps accum = 1, then accum 2 at x steps.
                if self.current_step % 1000 == 0 or self.current_step == 1:# or ((batch+1) == len(dataloader)) or effective_steps == 1:
                    self.saveModel(
                        mel_loss=melLoss.item(),
                        postmel_loss=postMelLoss.item(),
                        stop_loss=stopLoss.item(),
                        GA_loss=GA_loss.item(),
                        GE_loss=GE_loss.item(),
                        GA_gamma=GA_gamma,
                        GE_gamma=GE_gamma,
                        stop_gamma=stop_gamma,
                        p_teacher=p_teacher,
                        learning_rate=self.scheduler.get_last_lr()[0],
                        current_step=self.current_step,
                        attn_weights=cross_attn_weights,
                        saving=save
                    )
                save = True
                                        
#        return (mel_losses / effective_steps), (postmel_losses / effective_steps), (stop_losses / effective_steps), (GA_losses / effective_steps), (GE_losses / effective_steps), (learning_rates / effective_steps), (gammas / effective_steps)
    
    def testingLoop(self, dataloader):
        self.model.eval()

        stop_losses = 0.0
        mel_losses = 0.0
        postmel_losses = 0.0
        count = 0

        with torch.inference_mode():
            for batch, (text, mel, mel_len) in enumerate(dataloader):
                x, y = text.to(hp.device), mel.to(hp.device)
                x = x.type(torch.long)

                
                # Apply Start Of Sequence Frame to MEL
                start_frame = torch.full((hp.batch_size, 1, hp.n_mel_bin), fill_value=hp.start_frame_mel_fill_value, dtype=torch.float32, device=hp.device)
                y_in = torch.cat([start_frame, y[:, :-1, :]], dim=1) # Shift MEL right by one frame

                with autocast(device_type=hp.device):
                    melResult, postnetResult, stopResult, _ = self.model(x, y_in)
                    stopResult = stopResult.squeeze(-1)

                melResult = melResult.float()
                postnetResult = postnetResult.float()
                stopResult = stopResult.float()

                melLoss, postMelLoss, stopLoss = self._calculateLoss(
                    mel_pred=melResult,
                    mel_true=y,
                    post_mel_pred=postnetResult,
                    stop_pred=stopResult,
                    stop_true=mel_len
                )

                mel_losses += melLoss.item()
                postmel_losses += postMelLoss.item()
                stop_losses += stopLoss.item()
                count += 1


        avg_loss = (stop_losses + mel_losses) / count

        if avg_loss < self.best_eval or hp.save_latest:
            self.best_eval = avg_loss

            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'scaler': self.scaler.state_dict(),
                'best_eval': self.best_eval,
                'epoch': self.current_step,
            }, f'{hp.MODEL_SAVE_ROOT}checkpoint.pth')

            #print(f'[i] Saving model/optimizer state dict, with new best loss.-: {avg_loss:.3f}')
        return (mel_losses / count), (postmel_losses / count), (stop_losses / count)

    def mainLoop(self):
        self.current_step += 1

#        with open(f"{hp.MODEL_SAVE_ROOT}progress.csv", mode='a') as progress_file:
#            for _ in range(hp.epoch):

  #              train_mel_avg, train_postmel_avg, train_stop_avg, GA_losses_avg, GE_losses_avg, learning_rate_avg, gamma_avg = self.trainingLoop(self.trainLoader)
  #              test_mel_avg, test_postmel_avg, test_stop_avg = self.testingLoop(self.testLoader)

#                print(f'Epoch: {self.current_step} | Learning Rate: {learning_rate_avg:.6f} | Guided Attention Loss: {GA_losses_avg:.4f} | Guided Entropy Loss: {GE_losses_avg:.4f} | Gamma: {gamma_avg:.4f}\nTraining loss: (MEL: {train_mel_avg:.3f}) (POSTNET MEL: {train_postmel_avg:.3f}) (STOP: {train_stop_avg:.3f}) -> {train_postmel_avg + train_mel_avg + train_stop_avg + GA_losses_avg + GE_losses_avg:.3f}\nTesting loss: (MEL: {test_mel_avg:.3f}) (POSTNET MEL: {test_postmel_avg:.3f}) (STOP: {test_stop_avg:.3f}) -> {test_postmel_avg + test_mel_avg + test_stop_avg:.3f}')
#                progress_file.write(f"{self.current_step};{learning_rate_avg};{train_mel_avg + train_stop_avg + train_postmel_avg};{test_mel_avg + test_stop_avg + test_postmel_avg}\n")
#                progress_file.flush()

  #              self.current_step += 1

#            progress_file.close()


            
            





model = AR_TTS()
optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr_noam)
scheduler = NoamScheduler(optimizer=optimizer, d_model=hp.n_embd, warmup_steps=hp.warmup_steps, gamma=hp.gamma)
scaler = GradScaler(init_scale=2**15)
melLossFn = nn.HuberLoss()
postMelLossFn = nn.HuberLoss()
stopLossFn = nn.BCEWithLogitsLoss(reduction="none") # Not directly used.
best_eval = 9999
current_step = 0

if hp.continue_training_from_checkpoint:
    # Loads checkpoint if one is found. TODO: add best_eval here and update.
    try:
        checkpoint = torch.load(f"{hp.MODEL_SAVE_ROOT}checkpoint.pth", map_location=hp.device)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        best_eval = checkpoint["best_eval"]
        current_step = checkpoint["epoch"]

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(hp.device)

        print(f"Successfully loaded checkpoint. Continuing training.. at step: {current_step}")

    except FileNotFoundError as e:
        print('Tried loading model/optimizer/scheduler state dicts, but process failed. [FileNotFoundError]')
        print('Proceeding to start training from scratch.')

trainer = TrainModel(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    melLossFn=melLossFn,
    postMelLossFn=postMelLossFn,
    stopLossFn=stopLossFn,
    best_eval=best_eval,
    current_step=current_step,
    trainLoader=train_loader,
    testLoader=test_loader
                     )


if __name__ == "__main__":
#    with torch.autograd.set_detect_anomaly(True):
#        trainer.trainingLoop(train_loader)
    if hp.gradient_accumulation_steps * hp.batch_size != 64:
        print(f"Virtual batch size does not match 64! - batch size: {hp.gradient_accumulation_steps * hp.batch_size}")

    for _ in range(hp.epoch):
        trainer.trainingLoop(train_loader)

#    trainer.mainLoop()
