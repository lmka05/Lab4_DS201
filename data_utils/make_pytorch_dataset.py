import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from data_utils.build_vocab import tokenize_en, tokenize_vi

MAX_LENGTH = 50

# Hàm build pytorch dataset
class TranslationDataset(Dataset):
    def __init__(self, dataframe, vocab_src, vocab_trg):
        self.df = dataframe
        self.vocab_src = vocab_src # Tiếng Anh
        self.vocab_trg = vocab_trg # Tiếng Việt

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        src_text = row['english']
        trg_text = row['vietnamese']


        # Chuyển text thành list số
        # Thêm <sos> vào đầu và <eos> vào cuối
        src_indices = [self.vocab_src.stoi["<sos>"]] + \
                      self.vocab_src.numericalize(src_text, tokenize_en)[:MAX_LENGTH] + \
                      [self.vocab_src.stoi["<eos>"]]


        trg_indices = [self.vocab_trg.stoi["<sos>"]] + \
                      self.vocab_trg.numericalize(trg_text, tokenize_vi)[:MAX_LENGTH] + \
                      [self.vocab_trg.stoi["<eos>"]]

        return torch.tensor(src_indices), torch.tensor(trg_indices)

# Hàm collate giúp đệm (pad) các câu ngắn cho bằng câu dài nhất trong batch
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # batch là list các tuple (src_tensor, trg_tensor) lấy từ Dataset
        src_batch = [item[0] for item in batch]
        trg_batch = [item[1] for item in batch]

        # Pad sequence: thêm số 0 (<pad>) vào sau để độ dài bằng nhau
        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=self.pad_idx)
        trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=self.pad_idx)

        return {'src': src_batch, 'trg': trg_batch}# Hàm build pytorch dataset
class TranslationDataset(Dataset):
    def __init__(self, dataframe, vocab_src, vocab_trg):
        self.df = dataframe
        self.vocab_src = vocab_src # Tiếng Anh
        self.vocab_trg = vocab_trg # Tiếng Việt

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        src_text = row['english']
        trg_text = row['vietnamese']


        # Chuyển text thành list số
        # Thêm <sos> vào đầu và <eos> vào cuối
        src_indices = [self.vocab_src.stoi["<sos>"]] + \
                      self.vocab_src.numericalize(src_text, tokenize_en)[:MAX_LENGTH] + \
                      [self.vocab_src.stoi["<eos>"]]


        trg_indices = [self.vocab_trg.stoi["<sos>"]] + \
                      self.vocab_trg.numericalize(trg_text, tokenize_vi)[:MAX_LENGTH] + \
                      [self.vocab_trg.stoi["<eos>"]]

        return torch.tensor(src_indices), torch.tensor(trg_indices)

# Hàm collate giúp đệm (pad) các câu ngắn cho bằng câu dài nhất trong batch
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # batch là list các tuple (src_tensor, trg_tensor) lấy từ Dataset
        src_batch = [item[0] for item in batch]
        trg_batch = [item[1] for item in batch]

        # Pad sequence: thêm số 0 (<pad>) vào sau để độ dài bằng nhau
        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=self.pad_idx)
        trg_batch = pad_sequence(trg_batch, batch_first=True, padding_value=self.pad_idx)

        return {'src': src_batch, 'trg': trg_batch}