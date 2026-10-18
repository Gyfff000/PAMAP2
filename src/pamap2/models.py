import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, channels, widths=(16, 24, 32)):
        super().__init__()
        layers = []
        for width, kernel in zip(widths, (7, 5, 3)):
            layers.extend([nn.Conv1d(channels, width, kernel, stride=2, padding=kernel//2, bias=False),
                           nn.BatchNorm1d(width), nn.ReLU()])
            channels = width
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)

class EarlyFusionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(18, (24, 48, 64))
        self.head = nn.Sequential(nn.Dropout(0.25), nn.Linear(64, 12))

    def forward(self, x, features):
        return self.head(self.encoder(x))

class SensorFeatureFusion(nn.Module):
    """Three local temporal encoders plus a window-statistics branch."""
    def __init__(self):
        super().__init__()
        self.sensors = nn.ModuleList([Encoder(6) for _ in range(3)])
        self.features = nn.Sequential(nn.Linear(150, 64), nn.ReLU(), nn.Dropout(0.25))
        self.head = nn.Sequential(nn.Linear(160, 64), nn.ReLU(), nn.Dropout(0.25), nn.Linear(64, 12))

    def forward(self, x, features):
        temporal = [encoder(x[:, i*6:(i+1)*6]) for i, encoder in enumerate(self.sensors)]
        return self.head(torch.cat(temporal + [self.features(features)], dim=1))

def make_model(name):
    return {'cnn': EarlyFusionCNN, 'fusion': SensorFeatureFusion}[name]()
