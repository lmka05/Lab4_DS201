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
        # 3 lớp LSTM
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))


        outputs, (hidden, cell) = self.rnn(embedded)

        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # score = v^T * tanh(W1 * h_dec + W2 * h_enc)
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):

        src_len = encoder_outputs.shape[1]


        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)


        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))


        attention = self.v(energy).squeeze(2)

        # Softmax để trọng số có tổng bằng 1
        return F.softmax(attention, dim=1)

class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.attention = Attention(hidden_dim)

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(hidden_dim + emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

        self.fc_out = nn.Linear(hidden_dim, output_dim) # Đầu ra dự đoán từ
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):

        input = input.unsqueeze(1) 
        embedded = self.dropout(self.embedding(input)) 


        a = self.attention(hidden[-1], encoder_outputs) 
        a = a.unsqueeze(1) 


        weighted = torch.bmm(a, encoder_outputs)


        rnn_input = torch.cat((embedded, weighted), dim=2) 
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden, cell

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

        # Input đầu tiên là <sos>
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)

            outputs[:, t, :] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs