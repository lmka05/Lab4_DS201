import evaluate
import torch

# Tải các metrics từ thư viện evaluate của Hugging Face
rouge = evaluate.load('rouge')

def train_model(model, iterator, optimizer, criterion, clip,device):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch['src'].to(device)
        trg = batch['trg'].to(device)

        optimizer.zero_grad()
        output = model(src, trg)

        # output: [batch_size, trg_len, output_dim] -> bỏ token đầu
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def eval_model(model, iterator, vocab_trg_stoi, vocab_trg_itos,device):
    model.eval()

    refs = [] # Dữ liệu thật (Tiếng Việt)
    preds = [] # Dự đoán của máy

    with torch.no_grad():
        for batch in iterator:
            src = batch['src'].to(device)
            trg = batch['trg'].to(device)

            # Tắt teacher forcing khi đánh giá
            output = model(src, trg, 0)

            # Lấy token có xác suất cao nhất
            output = output.argmax(2)

            # Chuyển từ index sang text
            for i in range(output.shape[0]):
                # Lấy câu dự đoán (bỏ qua token 0 là <SOS> nếu có)
                pred_tokens = [vocab_trg_itos[idx.item()] for idx in output[i] if idx.item() not in [vocab_trg_stoi['<pad>'], vocab_trg_stoi['<sos>'], vocab_trg_stoi['<eos>']]]

                # Lấy câu gốc
                trg_tokens = [vocab_trg_itos[idx.item()] for idx in trg[i] if idx.item() not in [vocab_trg_stoi['<pad>'], vocab_trg_stoi['<sos>'], vocab_trg_stoi['<eos>']]]

                preds.append(" ".join(pred_tokens))
                refs.append(" ".join(trg_tokens)) # Refs cần là list of list cho một số thư viện


    rouge_results = rouge.compute(predictions=preds, references=refs)
    return rouge_results