import torch
from transformers import AutoModelForNextSentencePrediction


def generate(model, init_text, tokenizer, max_len):
    model.eval()
    tokens = tokenizer.encode(init_text)
    x = torch.tensor(tokens)
    x = torch.unsqueeze(x, 0)
    with torch.no_grad():
        x_out = model(x)
        y = torch.argmax(x_out, dim=1).item()
        while (len(tokenizer.decode(tokens).strip().split()) < max_len) and (y != tokenizer.eos_token) and (y != tokens[-1]):
            tokens.append(y)
            x = torch.tensor(tokens)
            x = torch.unsqueeze(x, 0)
            x_out = model(x)
            y = torch.argmax(x_out, dim=1).item()
    result = tokenizer.decode(tokens)
    return result

def generate2(model: AutoModelForNextSentencePrediction, init_text, tokenizer, max_len):
    model.eval()
    tokens = tokenizer.encode(init_text, return_tensors="pt")
    res_ids = model.generate(tokens, max_length=max_len, early_stopping=True)
    results = tokenizer.decode(res_ids[0], skip_special_tokens=True)
    return results
    # x = torch.tensor(tokens)
    # x = torch.unsqueeze(x, 0)
    # with torch.no_grad():
    #     x_out = model(x)
    #     y = torch.argmax(x_out, dim=1).item()
    #     while (len(tokenizer.decode(tokens).strip().split()) < max_len) and (y != tokenizer.eos_token) and (y != tokens[-1]):
    #         tokens.append(y)
    #         x = torch.tensor(tokens)
    #         x = torch.unsqueeze(x, 0)
    #         x_out = model(x)
    #         y = torch.argmax(x_out, dim=1).item()
    # result = tokenizer.decode(tokens)
