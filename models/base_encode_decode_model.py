import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# Cấu hình thiết bị (GPU nếu có)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(num_embeddings=input_dim,embedding_dim= emb_dim)

        self.LSTM = nn.LSTM(input_size = emb_dim, hidden_size=hidden_dim, num_layers= n_layers, dropout=dropout, batch_first=True)

        # Tự custom thêm lớp dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))

        # outputs: [batch_size, src_len, hidden_dim]
        # hidden, cell: [n_layers, batch_size, hidden_dim]
        outputs, (hidden, cell) = self.LSTM(embedded)

        # Trả về hidden và cell state để khởi tạo cho Decoder
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input: [batch_size] -> cần thêm chiều seq_len = 1
        input = input.unsqueeze(1)

        embedded = self.dropout(self.embedding(input))

        # output: [batch_size, 1, hidden_dim]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # Tensor chứa các dự đoán
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Đưa src qua Encoder để lấy context vector (hidden, cell)
        hidden, cell = self.encoder(src)

        # Đầu vào đầu tiên cho decoder là token <SOS> (token đầu tiên của trg)
        input = trg[:, 0]

        for t in range(1, trg_len):
            # Giải mã từng bước
            output, hidden, cell = self.decoder(input, hidden, cell)

            # Lưu dự đoán
            outputs[:, t, :] = output

            # Teacher Forcing: Quyết định dùng dự đoán của mô hình hay ground truth làm input tiếp theo
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs