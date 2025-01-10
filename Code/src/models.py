import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5,5))
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=(5,5))
        self.fc = nn.Linear(12 * 4 * 4, 10)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool(x)
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc(x)
        return x

class GenericClassifer(nn.Module):
    def __init__(self, input_size):
        super(GenericClassifer, self).__init__()
        self.fc = nn.Linear(input_size,10)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        out = self.fc(x)       
        
        return out
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.ReLU(),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
 
        return decoded