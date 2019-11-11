import torch 
from torch import nn
import torchvision

class AutoEncoder_alexnet(nn.Module):
    def __init__(self, dimension):
        super(AutoEncoder_alexnet, self).__init__()
        dimension = dimension//4
        self.encoder = nn.Sequential(
            nn.Conv2d(256, 1024, 3, stride=1, padding=0), ###11*11 -> 9*9
            nn.BatchNorm2d(1024),
            nn.PReLU(),

            nn.Conv2d(1024, 512, 3, stride=2, padding=0), ###9*9 -> 4*4
            nn.BatchNorm2d(512),
            nn.PReLU(),

            nn.Conv2d(512, dimension, 3, stride=1, padding=0),  ###4*4 -> 2*2
            nn.BatchNorm2d(dimension),
            nn.PReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dimension, 512, 3, stride=1, padding=0),  #2*2 -> 4*4
            nn.BatchNorm2d(512),
            nn.PReLU(),

            nn.ConvTranspose2d(512, 1024, 3, stride=2, padding=0),  #4*4 -> 9*9
            nn.BatchNorm2d(1024),
            nn.PReLU(),

            nn.ConvTranspose2d(1024, 256, 3, stride=1, padding=0),  #9*9 -> 11*11
            nn.BatchNorm2d(256),
            nn.PReLU()
        )
    
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded.view(x.size(0),-1),decoded

class AutoEncoder_resnet101(nn.Module):
    def __init__(self, dimension):
        super(AutoEncoder_resnet101, self).__init__()
        dimension = dimension//4
        self.encoder = nn.Sequential(
            nn.Conv2d(1024, 640, 4, stride=1, padding=0), ###1024*12*12 -> 640*9*9
            nn.BatchNorm2d(640),
            nn.PReLU(),

            nn.Conv2d(640, 256, 3, stride=2, padding=0), ###640*9*9 -> 256*4*4
            nn.BatchNorm2d(256),
            nn.PReLU(),

            nn.Conv2d(256, dimension, 3, stride=1, padding=0),  ###256*4*4 -> dimension*2*2
            nn.BatchNorm2d(dimension),
            nn.PReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dimension, 256, 3, stride=1, padding=0),  #dimension*2*2 -> 256*4*4
            nn.BatchNorm2d(256),
            nn.PReLU(),

            nn.ConvTranspose2d(256, 640, 3, stride=2, padding=0),  #256*4*4 -> 640*9*9
            nn.BatchNorm2d(640),
            nn.PReLU(),

            nn.ConvTranspose2d(640, 1024, 4, stride=1, padding=0),  #640*9*9 -> 1024*12*12
            nn.BatchNorm2d(1024),
            nn.PReLU()
        )

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded.view(x.size(0),-1),decoded


class AutoEncoder_densenet169(nn.Module):
    def __init__(self, dimension):
        super(AutoEncoder_densenet169, self).__init__()
        dimension = dimension//4
        self.encoder = nn.Sequential(
            nn.Conv2d(1280, 640, 4, stride=1, padding=0), ###1280*12*12 -> 640*9*9
            nn.BatchNorm2d(640),
            nn.PReLU(),

            nn.Conv2d(640, 256, 3, stride=2, padding=0), ###640*9*9 -> 256*4*4
            nn.BatchNorm2d(256),
            nn.PReLU(),
            
            nn.Conv2d(256, dimension, 3, stride=1, padding=0),  ###256*4*4 -> dimension*2*2
            nn.BatchNorm2d(dimension),
            nn.PReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dimension, 256, 3, stride=1, padding=0),  #dimension*2*2 -> 256*4*4
            nn.BatchNorm2d(256),
            nn.PReLU(),

            nn.ConvTranspose2d(256, 640, 3, stride=2, padding=0),  #256*4*4 -> 640*9*9
            nn.BatchNorm2d(640),
            nn.PReLU(),

            nn.ConvTranspose2d(640, 1280, 4, stride=1, padding=0),  #640*9*9 -> 1280*12*12
            nn.BatchNorm2d(1280),
            nn.PReLU()
        )
    
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded.view(x.size(0),-1),decoded
