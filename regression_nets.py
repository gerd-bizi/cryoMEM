import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from utils.base_nets import FCBlock


class InMembraneCircleRegressor(nn.Module):
    def __init__(self, in_features, hidden_features=[512, 256, 128]):
        super(InMembraneCircleRegressor, self).__init__()
        self.fc = FCBlock(
            in_features=in_features,
            features=hidden_features,
            out_features=2,
            nonlinearity='relu',
            last_nonlinearity=None,
            batch_norm=True,
            group_norm=0,
            dropout=0.1
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        phi_vals = self.fc(x)  # shape: [B, 2]
        norm = torch.norm(phi_vals, p=2, dim=-1, keepdim=True)
        phi_unit = phi_vals / (norm + 1e-6)
        return phi_unit


class GatedResidualAngleRegressor(nn.Module):
    def __init__(self, latent_dim, angle, angle_dim=1, hidden_features=[256, 128], max_delta=np.deg2rad(10)):
        super(GatedResidualAngleRegressor, self).__init__()
        self.max_delta = max_delta
        self.angle = angle  # either 'inplane' or 'tilt'

        self.delta_fc = FCBlock(
            in_features=latent_dim + angle_dim,
            features=hidden_features,
            out_features=1,
            nonlinearity='relu',
            last_nonlinearity='tanh',
            batch_norm=True
        )

        self.gate_fc = FCBlock(
            in_features=latent_dim,
            features=hidden_features,
            out_features=1,
            nonlinearity='relu',
            last_nonlinearity='sigmoid',
            batch_norm=True
        )

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            last_delta = self.delta_fc.net[-1]
            last_gate = self.gate_fc.net[-1]
            if isinstance(last_delta, nn.Linear):
                nn.init.zeros_(last_delta.weight)
                nn.init.zeros_(last_delta.bias)
            if isinstance(last_gate, nn.Linear):
                nn.init.zeros_(last_gate.weight)
                nn.init.ones_(last_gate.bias)

    def forward(self, latent, init_angles):
        x = torch.cat([latent, init_angles], dim=1)
        delta = self.delta_fc(x) * self.max_delta
        gate = self.gate_fc(x)
        refined_angles = init_angles + (1 - gate) * delta

        if self.angle == 'inplane':
            refined_angles = (refined_angles + np.pi) % (2 * np.pi) - np.pi
        elif self.angle == 'tilt':
            refined_angles = torch.clamp(refined_angles, min=0.0, max=math.pi)

        return refined_angles


class TiltRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], output_range=(0, np.pi)):
        super(TiltRegressor, self).__init__()
        self.layers = nn.Sequential()
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.add_module(f"fc{i}", nn.Linear(prev_dim, hidden_dim))
            self.layers.add_module(f"relu{i}", nn.ReLU(inplace=True))
            prev_dim = hidden_dim
        self.fc_out = nn.Linear(prev_dim, 1)
        self.min_angle, self.max_angle = output_range

        self._init_weights()

    def _init_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, x):
        x = self.layers(x)
        cos_theta = torch.tanh(self.fc_out(x))  # output in [-1, 1]
        tilt_angle = torch.acos(cos_theta)
        return tilt_angle


class S2S2Regressor(nn.Module):
    def __init__(self, latent_code_size, heads=1):
        super(S2S2Regressor, self).__init__()
        self.heads = heads
        self.orientation_regressor = nn.ModuleList([
            FCBlock(
                in_features=latent_code_size,
                out_features=6,
                features=[512, 256],
                nonlinearity='relu',
                last_nonlinearity=None,
                batch_norm=True,
                group_norm=0
            ) for _ in range(heads)
        ])
        self._init_weights()

    def _init_weights(self):
        for reg in self.orientation_regressor:
            for m in reg.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, latent):
        outputs = [reg(latent) for reg in self.orientation_regressor]
        return torch.stack(outputs, dim=1)  # shape: [B, heads, 6]