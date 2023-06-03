import torch
import torch.nn as nn

class BudgetConstraint_RatioMSE_DualGrain(nn.Module):
    def __init__(self, target_ratio=0., gamma=1.0, min_grain_size=8, max_grain_size=16, calculate_all=True):
        super().__init__()
        self.target_ratio = target_ratio  # e.g., 0.8 means 80% are fine-grained
        self.gamma = gamma
        self.calculate_all = calculate_all  # calculate all grains
        self.loss = nn.MSELoss()

        self.const = min_grain_size * min_grain_size
        self.max_const = max_grain_size * max_grain_size - self.const
        
    def forward(self, gate):
        # 0 for coarse-grained and 1 for fine-grained
        # gate: (batch, 2, min_grain_size, min_grain_size)
        beta = 1.0 * gate[:, 0, :, :] + 4.0 * gate[:, 1, :, :]
        beta = (beta.sum() / gate.size(0)) - self.const
        budget_ratio = beta / self.max_const
        target_ratio = self.target_ratio * torch.ones_like(budget_ratio).to(gate.device)
        loss_budget = self.gamma * self.loss(budget_ratio, target_ratio)

        if self.calculate_all:
            loss_budget_last = self.gamma * self.loss(1 - budget_ratio, 1 - target_ratio)
            return loss_budget_last + loss_budget_last

        return loss_budget
        
class BudgetConstraint_NormedSeperateRatioMSE_TripleGrain(nn.Module):
    def __init__(self, target_fine_ratio=0., target_median_ratio=0., gamma=1.0, min_grain_size=8, median_grain_size=16, max_grain_size=32):
        super().__init__()
        assert target_fine_ratio + target_median_ratio <= 1.0
        self.target_fine_ratio = target_fine_ratio  # e.g., 0.8 means 80% are fine-grained
        self.target_median_ratio = target_median_ratio
        self.gamma = gamma
        self.loss = nn.MSELoss()

        self.min_const = min_grain_size * min_grain_size
        self.median_const = median_grain_size * median_grain_size - self.min_const
        self.max_const = max_grain_size * max_grain_size - self.min_const

    def forward(self, gate):
        # 0 for coarse-grained, 1 for median-grained, 2 for fine grained
        # gate: (batch, 3, min_grain_size, min_grain_size)
        beta_median = 1.0 * gate[:, 0, :, :] + 4.0 * gate[:, 1, :, :] + 1.0 * gate[:, 2, :, :]  # the last term is the compensation for median ratio
        beta_median = (beta_median.sum() / gate.size(0)) - self.min_const
        budget_ratio_median = beta_median / self.median_const

        target_ratio_median = self.target_median_ratio * torch.ones_like(budget_ratio_median).to(gate.device)
        loss_budget_median = self.loss(budget_ratio_median, target_ratio_median)

        beta_fine = 1.0 * gate[:, 0, :, :] + 16.0 * gate[:, 2, :, :] + 1.0 * gate[:, 1, :, :]  # the last term is the compensation for fine ratio
        beta_fine = (beta_fine.sum() / gate.size(0)) - self.min_const
        budget_ratio_fine = beta_fine / self.max_const

        target_ratio_fine = self.target_fine_ratio * torch.ones_like(budget_ratio_fine).to(gate.device)
        loss_budget_fine = self.gamma * self.loss(budget_ratio_fine, target_ratio_fine)

        return loss_budget_fine + loss_budget_median