import torch

class LSTM_ASR(torch.nn.Module):
    def __init__(self, feature_type="discrete", input_size=64, hidden_size=256, num_layers=2,output_size=28):
        super().__init__()
        assert feature_type in ['discrete', 'mfcc']
        self.feature_type = feature_type
        if feature_type == "discrete":
            vocabs = 257 
            self.word_embeddings = torch.nn.Embedding(vocabs, input_size)
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, batch_features):
        if self.feature_type == "discrete":
            batch_features = self.word_embeddings(batch_features)
        lout, _ = self.lstm(batch_features) 
        lin = self.linear(lout)
        lp = torch.nn.functional.log_softmax(lin, dim=2)
        return lp
