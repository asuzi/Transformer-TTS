from torch.optim.lr_scheduler import _LRScheduler
import math
from hyperparams import Hyperparams as hp

class NoamScheduler(_LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps, gamma, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps

        # Guided attention loss scaling
        self.gamma = gamma

        super(NoamScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self._step_count, 1)
        scale = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        return [base_lr * scale for base_lr in self.base_lrs]

    def get_gamma(self, speedup=1, decay_factor=1):
        step = max(self._step_count * speedup, 1)
        if step >= self.warmup_steps * decay_factor:
            return 0.0
        return self.gamma * (1.0 - step / (self.warmup_steps * decay_factor))
    
    def get_sampling_params(self, seq_len):
        step = max(self._step_count, 1)
        if step <= self.warmup_steps + hp.TF_duration:
            return 1.0, 1, 0
        
        p = min((step - (self.warmup_steps + hp.TF_duration)) / hp.decay_duration, 1.0)
        
        p_teacher = 1.0 - p * (1.0 - hp.min_p_teacher)
        window_size = max(1, int(p * seq_len))

        p_continue = hp.p_continue_min + (1.0 - p_teacher) * (hp.p_continue_max - hp.p_continue_min)
        p_continue = min(max(p_continue, hp.p_continue_min), hp.p_continue_max)

        return p_teacher, window_size, p_continue
    