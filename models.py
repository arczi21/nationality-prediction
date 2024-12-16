import torch
import torch.nn as nn
import torch.optim as optim


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, num_layers, final_dropout=0):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=final_dropout)
        self.fc = nn.Linear(self.hidden_dim, output_size)

    def forward(self, x, lengths=None):
        out = self.embeddings(x)
        out, (h, c) = self.lstm(out)
        if lengths is None:
            out = out[:, -1, :]
        else:
            out = out.gather(1, lengths.unsqueeze(-1).unsqueeze(-1).expand(out.size(0), 1, self.hidden_dim)-1).squeeze(1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class GRU(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, num_layers, final_dropout=0):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(input_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=final_dropout)
        self.fc = nn.Linear(self.hidden_dim, output_size)

    def forward(self, x, lengths=None):
        out = self.embeddings(x)
        out, _ = self.gru(out)
        if lengths is None:
            out = out[:, -1, :]
        else:
            out = out.gather(1, lengths.unsqueeze(-1).unsqueeze(-1).expand(out.size(0), 1, self.hidden_dim)-1).squeeze(1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class A(nn.Module):
    def __init__(self, hidden_size, n_categories):
        super(A, self).__init__()
        self.fc = nn.Linear(hidden_size, n_categories)
        self.categories = nn.Parameter(torch.empty(n_categories, hidden_size))
        nn.init.normal_(self.categories, mean=0.0, std=0.02)

    def forward(self, x):
        p = self.fc(x)
        p = nn.functional.softmax(p, dim=-1)
        v = torch.matmul(p, self.categories)

        return v


class B(nn.Module):
    def __init__(self, hidden_size, n_categories):
        super(B, self).__init__()
        self.fc = nn.Linear(hidden_size, n_categories)
        self.categories = nn.Parameter(torch.empty(n_categories, hidden_size))
        nn.init.normal_(self.categories, mean=0.0, std=0.02)
        self.fc2 = nn.Linear(2*hidden_size, hidden_size)

    def forward(self, x):
        p = self.fc(x)
        p = nn.functional.softmax(p, dim=-1)
        v = torch.matmul(p, self.categories)
        v = torch.cat((x, v), dim=-1)
        v = self.fc2(v)

        return v


class GRUCAT(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, num_layers, n_cats):
        super(GRUCAT, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(input_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, self.hidden_dim, self.num_layers, batch_first=True)

        self.a1 = B(hidden_dim, n_cats)
        self.a2 = B(hidden_dim, n_cats)

        self.fc = nn.Linear(self.hidden_dim, len(nat_labels))

    def forward(self, x, lengths=None):
        out = self.embeddings(x)
        out, _ = self.gru(out)
        if lengths is None:
            out = out[:, -1, :]
        else:
            out = out.gather(1, lengths.unsqueeze(-1).unsqueeze(-1).expand(out.size(0), 1, self.hidden_dim)-1).squeeze(1)

        out = self.a1(out)
        out = self.a2(out)

        out = self.fc(out)
        return out


class RNNSearch(nn.Module):
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, num_layers, dropout=0):
        super(RNNSearch, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, self.hidden_dim, self.num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.p = nn.Linear(self.hidden_dim * 2, 1)
        self.fc = nn.Linear(self.hidden_dim * 2, output_size)

    def forward(self, x):
        x = self.embeddings(x)
        h = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, h)

        concatenated_hidden = torch.cat((out[:, :, :self.hidden_dim], out[:, :, self.hidden_dim:].flip(1)), dim=2)
        prob = self.p(concatenated_hidden)
        # weighted = torch.sum(prob*concatenated_hidden, dim=1)
        weighted = torch.sum(nn.functional.softmax(prob, dim=1)*concatenated_hidden, dim=1)


        out = self.fc(weighted)
        return out


class TransformerEncoderClassifier(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, num_classes, max_len):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            batch_first=True  # Ensures batch comes first in the input
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]

        encoded = self.encoder(x)

        pooled = torch.mean(encoded, dim=1)
        out = self.fc(pooled)
        return out