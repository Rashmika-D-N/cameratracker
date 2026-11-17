"""Small causal next-pose predictor; outputs are not class probabilities."""
import torch.nn.functional as F
from torch import nn

class PosePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv1d(68, 16, 3)
        self.b = nn.Conv1d(16, 16, 3, dilation=2)
        self.head = nn.Linear(16, 34)

    def encode(self, x):
        x = F.relu(self.a(F.pad(x, (2, 0))))
        return F.relu(self.b(F.pad(x, (4, 0))))

    def forward(self, x):
        return self.head(self.encode(x.transpose(1, 2))[:, :, -1])
