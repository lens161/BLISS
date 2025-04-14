import torch
from torch import nn
from torch.utils.data import Dataset

class PQEmbedding(nn.Module):
    def __init__(self, m, num_codes, emb_dim):
        super(PQEmbedding, self).__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_codes, emb_dim) for _ in range(m)]
        )
    
    def forward(self, pq_codes):
        emb_list = [emb(pq_codes[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(emb_list, dim=1)

class BLISS_NN(nn.Module):
    '''
    Create a pytorch model to use in training.
    Model consists of input layer (size: amount of dimensions), hidden layer (size: 512) and
    output layer (size: amount of buckets). Layers are fully connected.
    Forward pass just provides raw loss and not softmax score as BCEWithLogitsLoss is used.
    '''
    def __init__(self, output_size, m=8, num_codes=256, emb_dim=16):
        super(BLISS_NN, self).__init__()
        self.pq_embedding = PQEmbedding(m, num_codes, emb_dim)
        input_size = m * emb_dim
        # takes input and projects it to 512 hidden neurons
        # fc stands for fully connected, referring to a fully connected matrix being created
        self.fc1 = nn.Linear(input_size, 512)
        # activation function
        self.relu = nn.ReLU()
        # output layer maps 512 hidden neurons to output neurons (representing the buckets)
        self.fc2 = nn.Linear(512, output_size)
        # turns all output values into softmax values that sum to 1 -> probabilities
        # self.sigmoid = nn.Sigmoid(dim=1)

    def forward(self, x):
        x = self.pq_embedding(x)
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
        if device == torch.device("cpu"):
            self.data = data
        else:
            self.data = torch.from_numpy(data).int()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == 'train':
        # turn nd.array into tensor when fetched from the Dataset
            if self.device == torch.device("cpu"):
                vector = torch.from_numpy(self.data[idx]).int()
                label = torch.from_numpy(self.labels[idx]).float()
            else:
                vector = self.data[idx]
                label = self.labels[idx]
            return vector, label, idx
        elif self.mode == 'map':
            return torch.from_numpy(self.data[idx]).int(), idx