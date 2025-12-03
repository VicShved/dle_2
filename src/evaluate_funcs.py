import torch
from attr.validators import max_len
from torch.utils.data import DataLoader

from src.generate import generate


def evaluate_f1(model, loader, criterion):
    model.eval()
    correct, total = 0, 0
    sum_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_output = model(x_batch)
            loss = criterion(x_output, y_batch)
            preds = torch.argmax(x_output, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            sum_loss += loss.item()
    return sum_loss / len(loader), correct / total

def evaluate_rouge(model, loader: DataLoader, tokenizer, rouge):
    model.eval()
    results_x = []
    results_y = []
    with torch.no_grad():
        for i,  (x_batch, y_batch) in enumerate(loader):
            if i % 1000 == 0:
                print("Evaluating model on batch {}/{}".format(i, len(loader)))
            for i in range(len(x_batch)):
                x_output = generate(model, init_text=x_batch[i], tokenizer=tokenizer, max_len=(len(x_batch[i].split()) + len(y_batch[i].split())))
                x_arr= x_output.split()[len(x_batch[i].split()):]
                results_x.append(" ".join(x_arr) if len(x_arr) >= 1 else " ")
                results_y.append(y_batch[i])
    res = rouge.get_scores(results_y, results_x, avg=True)
    print("rouge = ", res)
    return res