import torch
import torch.nn as nn
import numpy as np
import json

class TripleGrainFeatureRouter(nn.Module):
    def __init__(self, num_channels, normalization_type="none", gate_type="1layer-fc"):
        super().__init__()
        self.gate_median_pool = nn.AvgPool2d(2, 2)
        self.gate_fine_pool = nn.AvgPool2d(4, 4)

        self.num_splits = 3

        self.gate_type = gate_type
        if gate_type == "1layer-fc":
            self.gate = nn.Linear(num_channels * self.num_splits, self.num_splits)
        elif gate_type == "2layer-fc-SiLu":
            self.gate = nn.Sequential(
                nn.Linear(num_channels * self.num_splits, num_channels * self.num_splits),
                nn.SiLU(inplace=True),
                nn.Linear(num_channels * self.num_splits, self.num_splits),
            )
        elif gate_type == "2layer-fc-ReLu":
            self.gate = nn.Sequential(
                nn.Linear(num_channels * self.num_splits, num_channels * self.num_splits),
                nn.ReLU(inplace=True),
                nn.Linear(num_channels * self.num_splits, self.num_splits),
            )
        else:
            raise NotImplementedError()

        self.normalization_type = normalization_type
        if self.normalization_type == "none":
            self.feature_norm_fine = nn.Identity()
            self.feature_norm_median = nn.Identity()
            self.feature_norm_coarse = nn.Identity()
        elif "group" in self.normalization_type:  # like "group-32"
            num_groups = int(self.normalization_type.split("-")[-1])
            self.feature_norm_fine = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)
            self.feature_norm_median = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)
            self.feature_norm_coarse = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)
        else:
            raise NotImplementedError()


    def forward(self, h_fine, h_median, h_coarse, entropy=None):
        h_fine = self.feature_norm_fine(h_fine)
        h_median = self.feature_norm_median(h_median)
        h_coarse = self.feature_norm_coarse(h_coarse)

        avg_h_fine = self.gate_fine_pool(h_fine)
        avg_h_median = self.gate_median_pool(h_median)

        h_logistic = torch.cat([h_coarse, avg_h_median, avg_h_fine], dim=1).permute(0,2,3,1)
        gate = self.gate(h_logistic)
        return gate