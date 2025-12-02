import torch

def generate(model, init_text, tokenizer, max_len):
    model.eval()
    tokens = tokenizer.encode(init_text)
    x = torch.tensor(tokens)
    print("x=", x)
    x = torch.unsqueeze(x, 0)
    print("x=", x)
    with torch.no_grad():
        x_out = model(x)
        y = torch.argmax(x_out, dim=1).item()
        print("y=", y)
        while (len(tokens) < max_len) and (y != tokenizer.eos_token) and (y != tokens[-1]):
            tokens.append(y)
            x = torch.tensor(tokens)
            x = torch.unsqueeze(x, 0)
            x_out = model(x)
            y = torch.argmax(x_out, dim=1).item()
    result = tokenizer.decode(tokens)
    return result