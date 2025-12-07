import evaluate
import torch
import yaml
from datasets import tqdm
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from src.solution_data_utils import split_data, clean_text
from src.solution_evaluates import evaluate_rouge, evaluate_model, evaluate_rouge_gpt
from src.solution_datasets import PredictWordDataset, ValuateDataset
from src.solution_model import Predictor
from src.solution_savepoints import save_to_file
from src.solution_generates import generate


def main():
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    print(config)
    texts = []
    model_name = config['model_name']

    # with open("data/tweets.txt") as f:
    #     while s:=f.readline():
    #         text = clean_text(s) +'\n'
    #         texts.append(text)
    # with open('data/cleaned_tweets.txt', "w") as f:
    #         f.writelines(texts)
    # print(texts[:5])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("tokenizer.eos_token=", tokenizer.eos_token)
    print("AutoTokenizer.vocab_size", tokenizer.vocab_size)

    with open('data/cleaned_tweets.txt') as f:
        texts = f.read().splitlines()
    texts= list(filter(lambda x: len(x.split()) >=4, texts)) # беру твиты с длиной больше 3 слов
    if config["text_crop"] >  0:
        texts = texts[:config["text_crop"]]
    print(texts[:5])

    train_texts, test_texts = split_data(texts, train_size=0.6, test_size=0.2)
    max_len = max((len(text) for text in train_texts))
    print("max_len=", max_len)

    val_texts, test_texts = split_data(test_texts, train_size=0.5, test_size=0.5)
    print("len(train_texts)=", len(train_texts), " len(test_texts)=", len(test_texts))
    train_ds = PredictWordDataset(train_texts, tokenizer)
    val_ds = PredictWordDataset(val_texts, tokenizer)
    test_ds = ValuateDataset(test_texts)
    print(len(train_texts), "->", len(train_ds))

    print(len(train_texts), len(test_texts))

    train_loader = DataLoader(train_ds, batch_size=config.get('batch_size', 64), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.get('batch_size', 64))
    test_loader = DataLoader(test_ds, batch_size=config.get('batch_size', 64))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_conf = config['model']
    print("model_conf=", model_conf)
    model = Predictor(vocab_size=tokenizer.vocab_size, embedding_dim=model_conf["embedding_dim"],
                      hidden_dim=model_conf["hidden_dim"], n_layers=model_conf["n_layers"],
                      dropout=model_conf["dropout"], device=device)
    model.to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print("parameters=", parameters)

    optimizer = torch.optim.Adam(model.parameters(), lr=model_conf["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    # rouge = Rouge()
    rouge = evaluate.load("rouge")
    rouge.add_batch(predictions=["1 2 3"], references=["1 2 3"])
    print("rouge.compute()=", rouge.compute())

    # Основной цикл обучения
    n_epochs = config['n_epochs']

    for epoch in range(n_epochs):
        train_loss = 0.
        for x_batch, y_batch in tqdm(train_loader):
            model.train()
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch + 1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_acc:.2%}")

        save_to_file(model=model, optimizer=optimizer, epoch=epoch, loss=val_loss)


        eval_res = evaluate_rouge(model, test_loader, tokenizer,rouge, device=device)
        print(f"epoch {epoch + 1} rouge=", eval_res)

        generated_line = generate(model, "this is first my next word prediction", tokenizer, max_len=10, device=device)
        print("generated_line=", generated_line)

    tr_model = AutoModelForCausalLM.from_pretrained(model_name)
    # tr_model.generation_config.pad_token_id = tokenizer.eos_token_id
    generator = pipeline(
        task="text-generation",
        model=tr_model,
        tokenizer=tokenizer,
        device=0,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt = "this is first my next word prediction"
    result = generator(
    prompt,
    # max_length=80,       # итоговая длина (включая prompt)
    max_new_tokens=10,
    num_return_sequences=1,
    do_sample=True,      # стохастическая генерация
    top_p=0.95,          # nucleus sampling
    temperature=0.8,
    pad_token_id=tokenizer.pad_token_id,
)
    print("generated_line", result)
    res = evaluate_rouge_gpt(generator, test_loader, rouge)
    print(f"rouge {model_name}=", res)

if __name__ == "__main__":
    main()
