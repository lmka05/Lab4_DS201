import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))


        outputs, (hidden, cell) = self.rnn(embedded)

        return outputs, hidden, cell

# LUONG ATTENTION 
class LuongAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        #  score(h_t, h_s) = h_t^T * W * h_s
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):

        query = self.W(decoder_hidden)


        energy = torch.bmm(query, encoder_outputs.permute(0, 2, 1))

        #  Softmax để ra trọng số
        return F.softmax(energy, dim=2)
# DECODER 
class LuongDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

        self.attention = LuongAttention(hidden_dim)

        self.fc_concat = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))

        rnn_output, (hidden, cell) = self.rnn(embedded, (hidden, cell))


        weights = self.attention(rnn_output, encoder_outputs)


        context = torch.bmm(weights, encoder_outputs)


        concat_input = torch.cat((rnn_output, context), dim=2)
        attn_hidden = torch.tanh(self.fc_concat(concat_input))

        prediction = self.fc_out(attn_hidden.squeeze(1))

        return prediction, hidden, cell

# SEQ2SEQ 
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs