import torch.nn as nn
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


def create_scheduler(optimizer, scheduler_config, steps_per_epoch, max_epoch):
    
    multiplier = scheduler_config.multiplier
    warmup_steps = scheduler_config.warmip_epoch * steps_per_epoch
    buffer_steps = scheduler_config.buffer_epoch * steps_per_epoch
    final_steps = max_epoch * steps_per_epoch
    min_lr = scheduler_config.min_lr
    warmup_policy = scheduler_config.warmup_policy
    start_from_zero = scheduler_config.start_from_zero

    scheduler = CosineAnnealingLR(
        optimizer, T_max=final_steps - warmup_steps - buffer_steps, eta_min=min_lr
    )

    if warmup_steps > 0.0:
        if warmup_policy == 'linear':
            multiplier = max(1.0, multiplier)
        elif warmup_policy == 'sqrt':
            multiplier = max(1.0, multiplier)
        elif warmup_policy == 'fix':
            multiplier = max(1.0, multiplier)
        elif warmup_policy == 'none':
            pass
        else:
            raise NotImplementedError(f'{warmup_policy} is not a valid warmup policy')
        warmup = GradualWarmup(
            optimizer,
            steps=warmup_steps,
            buffer_steps=buffer_steps,
            multiplier=multiplier,
            start_from_zero=start_from_zero
        )
    else:
        warmup = None

    scheduler = Scheduler(warmup_scheduler=warmup, after_scheduler=scheduler)

    return scheduler


class GradualWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, steps, buffer_steps, multiplier, start_from_zero=True, last_epoch=-1):
        self.steps = steps
        self.t_steps = steps + buffer_steps
        self.multiplier = multiplier
        self.start_from_zero = start_from_zero

        super(GradualWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.steps:
            return [group['lr'] for group in self.optimizer.param_groups]

        if self.start_from_zero:
            multiplier = self.multiplier * min(1.0, (self.last_epoch / self.steps))
        else:
            multiplier = 1 + ((self.multiplier - 1) * min(1.0, (self.last_epoch / self.steps)))
        return [lr * multiplier for lr in self.base_lrs]


class Scheduler:
    def __init__(self, warmup_scheduler, after_scheduler):
        self.warmup_scheduler = warmup_scheduler
        self.after_scheduler = after_scheduler

    def step(self, epoch=None):
        if self.warmup_scheduler is not None:
            self.warmup_scheduler.step(epoch=epoch)

        if self.warmup_scheduler is None or \
           self.warmup_scheduler.last_epoch > self.warmup_scheduler.t_steps:
            self.after_scheduler.step(epoch=epoch)

    def get_last_lr(self):
        if self.warmup_scheduler is not None and \
           self.warmup_scheduler.last_epoch <= self.warmup_scheduler.t_steps:
            print("self.warmup_scheduler.get_last_lr(): ", self.warmup_scheduler.get_last_lr())
            return self.warmup_scheduler.get_last_lr()
        else:
            return self.after_scheduler.get_last_lr()

    def state_dict(self):
        return {
            'warmup': None if self.warmup_scheduler is None else self.warmup_scheduler.state_dict(),
            'after': self.after_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict):
        if self.warmup_scheduler is not None:
            self.warmup_scheduler.load_state_dict(state_dict['warmup'])
        self.after_scheduler.load_state_dict(state_dict['after'])