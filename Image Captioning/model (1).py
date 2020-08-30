import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """ Load the pretrained resnet """
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """ the forward() will return the embedded features in the image """
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.embed(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        """ Set hyper-parameters and build the layers
        Parameters:
        - embed_size: Dimensionality of image and word embeddings
        - hidden_size: number of features in hidden state of the RNN decoder
        - vocab_size: The size of vocabulary or output size
        - num_layers: Number of layers
        """
        self.embed = nn.Embedding(vocab_size, embed_size)
        # LSTM neural network will take the vectors as inputs
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.hidden2word = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embed = self.embed(captions[:,:-1])
        embed = torch.cat((features.unsqueeze(1), embed), 1)
        lstm_out, _ = self.lstm(embed)
        tag_outputs = self.hidden2word(lstm_out)
        return tag_outputs

    def sample(self, inputs, states=None, max_len=20):
        " This function accepts pre-processed image tensor (inputs) and returns predicted sentence. "
        result = []

        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            tag_output = self.hidden2word(lstm_out)
            predicted = torch.argmax(tag_output, dim=-1)
            result.append(predicted[0,0].item())
            inputs = self.embed(predicted)

        return result