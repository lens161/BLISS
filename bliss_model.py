import torch
from torch import nn
from torch.utils.data import Dataset


class BLISS_NN(nn.Module):
    '''
    Create a pytorch model to use in training.
    Model consists of input layer (size: amount of dimensions), hidden layer (size: 512) and
    output layer (size: amount of buckets). Layers are fully connected.
    Forward pass just provides raw loss and not softmax score as BCEWithLogitsLoss is used.
    '''
    def __init__(self, input_size, output_size):
        super(BLISS_NN, self).__init__()
        # takes input and projects it to 512 hidden neurons
        # fc stands for fully connected, referring to a fully connected matrix being created
        self.fc1 = nn.Linear(input_size, 512)
        # activation function
        self.relu = nn.ReLU()
        # output layer maps 512 hidden neurons to output neurons (representing the buckets)
        self.fc2 = nn.Linear(512, output_size)
        # turns all output values into softmax values that sum to 1 -> probabilities

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    
class BLISSDataset(Dataset):
    '''
    The dataset used for handling and loading training samples
    '''
    def __init__(self, data, labels, device, mode = 'train'):
        self.device = device
        self.labels = labels
        self.mode = mode
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == 'train':
            vector = torch.from_numpy(self.data[idx]).to(torch.float32)
            label = self.labels[idx].float()
            return vector, label, idx
        elif self.mode == 'build':
            return torch.from_numpy(self.data[idx]).float(), idx
 