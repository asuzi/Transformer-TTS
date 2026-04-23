from hyperparams import hp
import torch
from torch import nn
from torch.nn import functional as F
import math
from torch.amp import autocast

class Attention(nn.Module):
    def __init__(self, n_head, n_embd):
        super().__init__()

        self.q_linear = nn.Linear(n_embd, n_embd, bias=False)
        self.k_linear = nn.Linear(n_embd, n_embd, bias=False)
        self.v_linear = nn.Linear(n_embd, n_embd, bias=False)
        self.out = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(0.1)

        self.n_head = n_head
        self.head_size = n_embd // n_head
        assert n_embd % n_head == 0, "[model.Attention] - n_embd has to be divisible by n_head!"

    def generate_windowed_mask(self, tgt_len, window_size):
        mask = torch.full((tgt_len, tgt_len), float('-inf'), device=hp.device)
        for i in range(tgt_len):
            start = max(0, i - window_size)
            mask[i, start:i + 1] = 0.0  # allow attending within window
        return mask  # shape: [tgt_len, tgt_len]

        # A: (B, H, T_dec, T_enc)
    def attn_stats(self, A, band_g=0.2):  # A: (B,H,T_dec,T_enc)
        with torch.no_grad():
            probs, idx = A.max(dim=-1)  # (B,H,T_dec)
            entropy = -(A.clamp_min(1e-8) * A.clamp_min(1e-8).log()).sum(-1)
            entropy = entropy.masked_fill(torch.isnan(entropy), 0.0)
            entropy = entropy.mean().item()
            peak_mean = probs.mean().item()
            peak_max = probs.max().item()

            B,H,T_dec,T_enc = A.shape
            # linear mapping target
            target = torch.linspace(0, T_enc - 1, T_dec, device=A.device)
            diag_err = (idx.float() - target[None,None,:]).abs().mean().item()
            diag_err_norm = diag_err / T_enc  # 0..1, easier to compare across utts

            # % mass inside diagonal band (guided-attn style)
            t = torch.arange(T_dec, device=A.device).float()[:, None] / T_dec
            s = torch.arange(T_enc, device=A.device).float()[None, :] / T_enc
            band = torch.exp(-((t - s) ** 2) / (2 * (band_g ** 2)))  # (T_dec,T_enc)
            band_cov = (A * band).sum(-1).mean().item()

            return dict(entropy=entropy, peak_mean=peak_mean, peak_max=peak_max,
                        diag_err=diag_err, diag_err_norm=diag_err_norm,
                        band_cov=band_cov)


    def forward(self, query, key, value, tgt_mask=None, memory_mask=None, padding_mask=None, apply_guided_loss=False, apply_entropy_loss=False):
        # (Batch, Sequence, n_embd) -> (Batch, Sequence, n_embd)
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        # (Batch, Sequence, n_embd) -> (Batch, Sequence, n_head, head_size) -> transpose -> (Batch, n_head, Sequence, head_size)
        q = q.view(q.shape[0], q.shape[1], self.n_head, self.head_size).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.n_head, self.head_size).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.n_head, self.head_size).transpose(1, 2)


        with autocast(device_type=hp.device, enabled=False):
            q_ = q.to(torch.float32)
            k_ = k.to(torch.float32)

            w = torch.matmul(q_, k_.transpose(-2, -1)) / (self.head_size ** 0.5)  # Scaled Dot Product

            if hp.attn_temperature > 0:
                w = w / hp.attn_temperature


            # Hide future tokens
            if tgt_mask is not None:
                tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
                w = w.masked_fill(tgt_mask, float("-1e9"))

            # Hide padding tokens
            if padding_mask is not None:
                padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
                w = w.masked_fill(padding_mask, float("-1e9"))

            # Hide padding tokens (cross attention)
            if memory_mask is not None:
                memory_mask = memory_mask.unsqueeze(1).unsqueeze(2)
                w = w.masked_fill(memory_mask, float("-1e9"))


            _attn_weights = None
            w = F.softmax(w, dim=-1)   
            if self.training and memory_mask is not None and apply_guided_loss:
                _attn_weights = w.detach()
            elif self.training and tgt_mask is not None and apply_entropy_loss:
                _attn_weights = w.detach()


        w = w.to(v.dtype)   # if autocast turns dtype back to float32

        if query.shape[0] == 1 and memory_mask is not None: 
            s = self.attn_stats(w)
            print("[Decoder cross attention] - "
                f"[Attn Stats] Entropy: {s['entropy']:.3f}, "
                f"Peak Mean: {s['peak_mean']:.3f}, "
                f"Diag Error: {s['diag_err']:.2f}, "
                f"Diag Error Norm: {s['diag_err_norm']:.3f}, "
                f"Band cov: {s['band_cov']:.3f}, ")

        elif query.shape[0] == 1 and tgt_mask is not None: 
            s = self.attn_stats(w)
            print(f"[Decoder masked attention] - "
                f"[Attn Stats] Entropy: {s['entropy']:.3f}, "
                f"Peak Mean: {s['peak_mean']:.3f}, "
                f"Diag Error: {s['diag_err']:.2f}, "
                f"Diag Error Norm: {s['diag_err_norm']:.3f}, "
                f"Band cov: {s['band_cov']:.3f}, ")

        else:
            if query.shape[0] == 1:
                s = self.attn_stats(w)
                print("[Encoder attention] - "
                    f"[Attn Stats] Entropy: {s['entropy']:.3f}, "
                    f"Peak Mean: {s['peak_mean']:.3f}, "
                    f"Diag Error: {s['diag_err']:.2f}, "
                    f"Diag Error Norm: {s['diag_err_norm']:.3f}, "
                    f"Band cov: {s['band_cov']:.3f}, ")



        w = self.dropout(w)         
        w = torch.matmul(w, v)      
    

        # (Batch, n_head, Sequence, head_size) -> (Batch, Sequence, n_head, head_size) -> contiguous view -> (batch, sequence, n_embd)
        w = w.transpose(1, 2).contiguous().view(w.shape[0], -1, self.head_size * self.n_head)
        output = self.out(w)

        return output, _attn_weights


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(hp.n_embd, hp.n_embd * hp.forward_expansion),
            nn.Dropout(hp.enc_dropout),
            nn.ReLU(),
            nn.Dropout(hp.enc_dropout),
            nn.Linear(hp.n_embd * hp.forward_expansion, hp.n_embd)
        )

        self.multihead_attention = Attention(n_head=hp.enc_n_head, n_embd=hp.n_embd)

        self.norm1 = nn.LayerNorm(hp.n_embd)
        self.norm2 = nn.LayerNorm(hp.n_embd)
        self.dropout = nn.Dropout(hp.enc_dropout)

    def forward(self, x, tgt_mask=None, memory_mask=None, padding_mask=None):
        # Pre-Norm architecture
        
        y = self.norm1(x)
        y, _ = self.multihead_attention(y, y, y, tgt_mask, memory_mask, padding_mask)
        y = self.dropout(y)
        x = x+y

        y = self.norm2(x)
        y = self.ffn(y)
        y = self.dropout(y)
        x = x+y

        return x


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(hp.n_embd, hp.n_embd * hp.forward_expansion),
            nn.ReLU(),
            nn.Dropout(hp.dec_dropout),
            nn.Linear(hp.n_embd * hp.forward_expansion, hp.n_embd)
        )

        self.masked_multihead_attention = Attention(n_head=hp.dec_n_head, n_embd=hp.n_embd)
        self.memory_multihead_attention = Attention(n_head=hp.dec_n_head, n_embd=hp.n_embd)

        self.norm1 = nn.LayerNorm(hp.n_embd)
        self.norm2 = nn.LayerNorm(hp.n_embd)
        self.norm3 = nn.LayerNorm(hp.n_embd)
        self.dropout = nn.Dropout(hp.dec_dropout)

    def forward(self, x, memory=None, tgt_mask=None, memory_mask=None, padding_mask=None, apply_guided_loss=False, apply_entropy_loss=False):
        # Pre-Norm architecture
        
        y = self.norm1(x)
        y, _self_attn_entropy = self.masked_multihead_attention(y, y, y, tgt_mask=tgt_mask, memory_mask=None, padding_mask=padding_mask, apply_entropy_loss=apply_entropy_loss)
        y = self.dropout(y)
        x = x + y

        y = self.norm2(x)
        y, _cross_attn_w = self.memory_multihead_attention(y, memory, memory, tgt_mask=None, memory_mask=memory_mask, padding_mask=None, apply_guided_loss=apply_guided_loss)
        y = self.dropout(y)
        x = x + y

        y = self.norm3(x)
        y = self.ffn(y)
        y = self.dropout(y)
        x = x + y

        return x, _cross_attn_w, _self_attn_entropy

class Encoder_PreNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=hp.n_embd, out_channels=hp.n_embd, kernel_size=(1, hp.kernel_size), stride=(1, hp.stride), padding=(0, int(hp.kernel_size // 2))),
            nn.BatchNorm2d(hp.n_embd),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=hp.n_embd, out_channels=hp.n_embd, kernel_size=(1, hp.kernel_size), stride=(1, hp.stride), padding=(0, int(hp.kernel_size // 2))),
            nn.BatchNorm2d(hp.n_embd),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=hp.n_embd, out_channels=hp.n_embd, kernel_size=(1, hp.kernel_size), stride=(1, hp.stride), padding=(0, int(hp.kernel_size // 2))),
            nn.BatchNorm2d(hp.n_embd),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        # Due to CuDNN complications, using Conv2d instead of Conv1d, transforming 

        x = x.transpose(2, 1)           # [B, n_embd, T]
        x = x.unsqueeze(2)              # [B, n_embd, 1, T] — height=1, width=T
        x = self.conv_net(x)            # [B, n_embd, 1, T_out]
        x = x.squeeze(2)                # [B, n_embd, T_out]
        x = x.transpose(2, 1)  

        return x

class Decoder_PreNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hp.n_mel_bin, hp.n_embd),
            nn.GELU(),
            nn.Dropout(p=hp.dec_dropout),
            nn.Linear(hp.n_embd, hp.n_embd),
            nn.GELU(),
            nn.Dropout(p=hp.dec_dropout),
        )

    def forward(self, x):
        x = self.net(x)

        return x


class PostNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv2d_start = nn.Sequential(

            nn.Conv2d(in_channels=hp.n_mel_bin, out_channels=hp.postnet_n_embd, kernel_size=(1, hp.postnet_kernel), stride=(1, hp.postnet_stride), padding=(0, int(hp.postnet_kernel // 2))),
            nn.BatchNorm2d(hp.postnet_n_embd),
            nn.Tanh(),
            nn.Dropout(hp.postnet_dropout)   
        )

        self.conv2d_middle = nn.Sequential(
            nn.Conv2d(in_channels=hp.postnet_n_embd, out_channels=hp.postnet_n_embd, kernel_size=(1, hp.postnet_kernel), stride=(1, hp.postnet_stride), padding=(0, int(hp.postnet_kernel // 2))),
            nn.BatchNorm2d(hp.postnet_n_embd),
            nn.Tanh(),
            nn.Dropout(hp.postnet_dropout)   
        )

        self.conv2d_end = nn.Sequential(
            nn.Conv2d(in_channels=hp.postnet_n_embd, out_channels=hp.n_mel_bin, kernel_size=(1, hp.postnet_kernel), stride=(1, hp.postnet_stride), padding=(0, int(hp.postnet_kernel // 2))),
            nn.BatchNorm2d(hp.n_mel_bin),
            nn.Dropout(hp.postnet_dropout)   
        )

        self.postnet_middle = nn.ModuleList(self.conv2d_middle for _ in range(hp.postnet_layers - 2))

    def forward(self, x):

        x = x.permute(0, 2, 1)      # [batch, seq, bin] -> [batch, bin, seq]
        x = x.unsqueeze(2)          # [batch, bin, seq] -> [batch, bin, 1, seq]
        
        x = self.conv2d_start(x)
        for layer in self.postnet_middle:
            x = layer(x)
        x = self.conv2d_end(x)


        x = x.squeeze(2)            # [batch, seq, 1, bin] -> [batch, bin, seq]
        x = x.permute(0, 2, 1)      # [batch, bin, seq] -> [batch, seq, bin]

        return x


class ScaledPositionalEncoding(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.n_embd = n_embd
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor([1.5]))

    def forward(self, x):
        seq_len = x.shape[1]  # dynamic sequence length


        position = torch.arange(0, seq_len, dtype=torch.float, device=hp.device).unsqueeze(1)
        divider = torch.exp(torch.arange(0, self.n_embd, 2, dtype=torch.float, device=hp.device) * (-math.log(10000.0) / self.n_embd))

        pe = torch.zeros(seq_len, self.n_embd, device=hp.device)
        pe[:, 0::2] = torch.sin(position * divider)
        pe[:, 1::2] = torch.cos(position * divider)

        x = x + self.alpha * pe.unsqueeze(0)  # [1, seq_len, n_embd]
        x = self.dropout(x)

        return x

class AR_TTS(nn.Module):
    def __init__(self):
        super().__init__()

    # Initialize encoder layers (input type: text)
        self.encoderPreNet = Encoder_PreNet()
        self.Encoder = nn.ModuleList([EncoderBlock() for _ in range(hp.enc_n_block)])

    # Initialize decoder layers (input type: mel-spectrogram)
        self.decoderPreNet = Decoder_PreNet()
        self.Decoder = nn.ModuleList([DecoderBlock() for _ in range(hp.dec_n_block)])

    # Initalize embedding layers (input type: both)
        self.positionalEmbedding = ScaledPositionalEncoding(n_embd=hp.n_embd)
        self.textEmbedding = nn.Embedding(hp.ph_vocab_len, hp.n_embd, padding_idx=0)
        self.melEmbedding = nn.Embedding(hp.n_mel_bin, hp.n_embd)   # <- NOT USED! is applied linearly in prenet.

    # Initialize post processing
        self.melProjection = nn.Linear(hp.n_embd, hp.n_mel_bin)
        self.stopProjection = nn.Linear(hp.n_embd, 1)

        self.postNet = PostNet()

    def _log_mel_signal_strength(self, mel):
        # [B, T, D]

        mean_per_timestep = mel.mean(dim=-1)       # [B, T]
        std_per_timestep = mel.std(dim=-1)         # [B, T]

        print("Mean per Timestep (batch 0)\n:", mean_per_timestep[0])
        print("Std per Timestep (batch 0):\n", std_per_timestep[0])

    def forward(self, text, mel, padding_mask_mel=None, sampling=False):

        # Apply masking
        padding_mask_text = (text == 0)             # Padding value: 0s

        if padding_mask_mel == None:    
            padding_mask_mel = (mel == 0).all(dim=-1)   # Padding value: 0

        mel_seq_len = mel.shape[1]
        tgt_mask_mel = torch.triu(torch.ones((mel_seq_len, mel_seq_len), dtype=torch.bool, device=mel.device), diagonal=1)

        # Process text input -> Encoder
        text = self.textEmbedding(text)                             # [batch, seq] -> [batch, seq, n_embed]
        text = self.encoderPreNet(text)                             # Encoder pre-net

        text = self.positionalEmbedding(x=text)  # Apply positional embedding.

        for i, encoderBlock in enumerate(self.Encoder):
            text = encoderBlock(
                                text,
                                padding_mask=padding_mask_text
                                 )




        mel = self.decoderPreNet(mel)                               # [batch, seq, bin] -> [batch, seq, n_embed]
        mel = mel.masked_fill(padding_mask_mel.unsqueeze(-1).expand_as(mel), 0.0) 
        mel = self.positionalEmbedding(x=mel)


        guided_attention_weights = []

        _cross_attn_weights = []
        _self_attn_weights = []

        for i, decoderBlock in enumerate(self.Decoder):
            apply_guided_loss = self.training
            apply_entropy_loss = self.training

            mel, _cross_attn_w, _self_attn_entropy = decoderBlock(
                x=mel,
                memory=text,
                padding_mask=padding_mask_mel,
                tgt_mask=tgt_mask_mel,
                memory_mask=padding_mask_text,
                apply_guided_loss=apply_guided_loss,
                apply_entropy_loss=apply_entropy_loss
                )


            if apply_guided_loss:
                _cross_attn_weights.append(_cross_attn_w)

            if apply_entropy_loss:
                _self_attn_weights.append(_self_attn_entropy)

        guided_attention_weights.append(_cross_attn_weights)
        guided_attention_weights.append(_self_attn_weights)



        melProj = self.melProjection(mel)                         # [batch, seq, n_embed] -> [batch, seq, bin]
        if sampling:
            return melProj

        postMel = self.postNet(melProj)                           # Refine mel
        stopProj = self.stopProjection(mel)        

        postMel = postMel + melProj                               # Residual connection
        postMel = postMel.masked_fill(padding_mask_mel.unsqueeze(-1).expand_as(melProj), 0.0)

        return melProj, postMel, stopProj, guided_attention_weights









