import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout=0.5, device=torch.device("cpu")):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.device = device
        # self.activation = nn.Softmax(dim=-1)
        self.init_weights()


    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)



    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x[:, -1, :])
        # x = self.activation(x)
        return x

    def generate(self, init_text, tokenizer, max_len):
        self.eval()
        tokens = tokenizer.encode(init_text)
        x = torch.tensor(tokens, device=self.device)
        x = torch.unsqueeze(x, 0)
        with torch.no_grad():
            x_out = self(x)
            y = torch.argmax(x_out, dim=1).item()
            while (len(tokenizer.decode(tokens).strip().split()) < max_len) and (y != tokenizer.eos_token) and (
                    y != tokens[-1]):
                tokens.append(y)
                x = torch.tensor(tokens)
                x = torch.unsqueeze(x, 0)
                x_out = self(x)
                y = torch.argmax(x_out, dim=1).item()
        result = tokenizer.decode(tokens)
        return result
