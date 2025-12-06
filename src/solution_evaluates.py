import torch
from attr.validators import max_len
from rouge import Rouge
from torch.utils.data import DataLoader
from datasets import tqdm
from src.solution_generates import generate


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
        for x_batch, y_batch in tqdm(loader):
            for i in range(len(x_batch)):
                x_output = generate(model, init_text=x_batch[i], tokenizer=tokenizer, max_len=(len(x_batch[i].split()) + len(y_batch[i].split())))
                x_arr= x_output.split()[len(x_batch[i].split()):]
                results_x.append(" ".join(x_arr) if len(x_arr) >= 1 else " ")
                results_y.append(y_batch[i])
    rouge.add_batch(predictions=results_x, references=results_y)
    res = rouge.compute()
    return res

def evaluate_rouge_gpt(generator, loader: DataLoader, rouge) -> dict:
    results_x = []
    results_y = []
    for x_batch, y_batch in tqdm(loader):
        for i in range(len(x_batch)):
            # print("x_batch[i]=", x_batch[i])
            x_output = generator(
                x_batch[i],
                max_new_tokens=len(y_batch[i].split()),  # итоговая длина (включая prompt)
                num_return_sequences=1,
                # do_sample=True,  # стохастическая генерация
                top_p=0.9,  # nucleus sampling
                temperature=1.0,
            )
            # print("x_output=", x_output)
            x_arr= x_output[0]['generated_text'].split()[len(x_batch[i].split()):]
            # if len(x_arr) ==0:
            #     print("x_arr=", x_arr, "y_batch[i]=", y_batch[i])
            r = " ".join(x_arr) if len(x_arr) >= 1 else " "
            if len(r) == 0:
                print("r=", r)
            results_x.append(r)
            results_y.append(y_batch[i])
            if len(y_batch[i]) == 0:
                print("y_batch[i]=", y_batch[i])
    print("len(results_x)=", len(results_x), "len(results_y)=", len(results_y))
    rouge.add_batch(predictions=results_x, references=results_y)
    res = rouge.compute()
    return res
