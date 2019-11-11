import torch
from torch import nn

###4224-2048-1024-512-256
###4224-2048-1024-512-256-128
####auto-encoder model
class autoencoder_prelu_4224(nn.Module):
    def __init__(self, dimension):
        super(autoencoder_prelu_4224, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4224, 2048),
            nn.BatchNorm1d(2048),
            nn.PReLU(),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),

            nn.Linear(256, dimension),
            nn.BatchNorm1d(dimension),
            nn.PReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dimension, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.PReLU(),

            nn.Linear(2048, 4224),
            nn.BatchNorm1d(4224),
            nn.PReLU(),
        )

    def forward(self,x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return encoded, decoded


