import torch
from pyvi import ViTokenizer
from collections import Counter
import spacy
import os

# Tải mô hình ngôn ngữ tiếng Anh của Spacy
spacy_en = spacy.load("en_core_web_sm")

# TOKENIZATION
def tokenize_en(text):
    """
    Tách từ tiếng Anh: "It begins with..." -> ["it", "begins", "with", "..."]
    """
    return [tok.text.lower() for tok in spacy_en.tokenizer(str(text))]

def tokenize_vi(text):
    """
    Tách từ tiếng Việt: "Câu chuyện bắt đầu..." -> ["câu_chuyện", "bắt_đầu", "..."]
    Dùng pyvi để gộp từ ghép.
    """
    # ViTokenizer.tokenize sẽ biến "bắt đầu" -> "bắt_đầu"
    text = text.lower()
    tokenized_str = ViTokenizer.tokenize(str(text))
    return tokenized_str.split()

# Hàm build VOCABULARY 
class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list, tokenizer_fn):
        frequencies = Counter()
        idx = 4

        # Đếm tần suất xuất hiện của các từ
        for sentence in sentence_list:
            for word in tokenizer_fn(sentence):
                frequencies[word] += 1

        # Chỉ thêm vào từ điển những từ xuất hiện nhiều hơn ngưỡng (threshold)
        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text, tokenizer_fn):
        """Chuyển câu văn thành list các số index"""
        tokenized_text = tokenizer_fn(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
            for token in tokenized_text
        ]

#  lưu từ điển để khi làm tiếp không cần build lại 
    def save_vocab(self, filepath):
        """Lưu nội dung từ điển vào file"""
        data = {
            "stoi": self.stoi,
            "itos": self.itos,
            "freq_threshold": self.freq_threshold
        }
        torch.save(data, filepath)
        print(f"Đã lưu từ điển tại: {filepath}")

# đọc từ điển 
    def load_vocab(self, filepath):
        """Đọc từ điển từ file đã lưu"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

        data = torch.load(filepath)
        self.stoi = data["stoi"]
        self.itos = data["itos"]
        self.freq_threshold = data["freq_threshold"]
        print(f"Đã tải từ điển từ: {filepath} (Size: {len(self.itos)})")
