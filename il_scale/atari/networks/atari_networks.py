import torch.nn as nn
from torch.nn import Sequential

from il_scale.utils.atari_conf import OBS_SHAPE
from il_scale.utils.model_utils import conv_outdim, count_params


class SimpleCNN(nn.Module):
    def __init__(self, w_scale: int, n: int = 4):
        super().__init__()

        self.w_scale = w_scale
        self.n = n
        self.cnn = Sequential(
            nn.Conv2d(4, 1 * w_scale, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(1 * w_scale, 2 * w_scale, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(2 * w_scale, 2 * w_scale, 3, stride=1),
        )

        w = OBS_SHAPE[0]
        w = conv_outdim(w, 8, stride=4)
        w = conv_outdim(w, 4, stride=2)
        w = conv_outdim(w, 3, stride=1)

        h = OBS_SHAPE[1]
        h = conv_outdim(h, 8, stride=4)
        h = conv_outdim(h, 4, stride=2)
        h = conv_outdim(h, 3, stride=1)

        self.policy = nn.Linear(w * h * 2 * w_scale, self.n)

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.permute(0, 1, 4, 2, 3)
        x = x.view(B * T, C, H, W)

        x = self.cnn(x)
        x = x.flatten(start_dim=1)
        x = self.policy(x)

        return x.view(B, T, -1)


model = SimpleCNN(w_scale=167)
print(f"{count_params(model)[0]:e}")

# 1: 725 ~1k
# 3: 2,575 ~2k
# 5: 4,969 ~5k
# 8: 9,580 ~10k
# 13: 19,985 ~20k
# 23: 50,995 ~50k
# 34: 100,814 ~100k
# 50: 202,654 ~200k
# 81: 499,045 ~500k
# 117: 1,007,257 ~1M
# 167: 2.005507e+06 ~2M
# 266: 4.985110e+06 ~5M
