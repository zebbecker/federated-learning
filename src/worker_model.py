import torch.nn as nn

model = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),  # in_ch, out_ch, k, stride, pad
            nn.ReLU(),
            nn.Conv2d(64, 4, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 10),
        )