import torch
from torch.utils.data import DataLoader


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
    results = []
    with torch.no_grad():
        for x_batch, y_batch in loader.generator:
            x_output = model(x_batch)
            preds = torch.argmax(x_output, dim=1)
            print("preds = ", preds)
            results.append(rouge.compute(preds, y_batch))

    res = dict(
        rouge1=sum([x["rouge1"] for x in results])/len(results),
        rouge2=sum([x["rouge2"] for x in results])/len(results),
    )
    return res