import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.bidirectional = bidirectional
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=bidirectional)

    def forward(self, inp, hidden):
        embedded = self.embedding(inp).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
        # return (torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size, device=device),
        #         torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size, device=device))
