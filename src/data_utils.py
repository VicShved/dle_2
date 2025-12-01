import re
import emoji
import pandas as pd
from torch.utils.data import Dataset, random_split, DataLoader, Subset


def clean_text(clean):
    clean = clean.lower()
    clean = re.sub('\s+', ' ', clean)
    clean = " ".join(filter(lambda x: x[0]!="@", clean.split()))
    clean = re.sub('<[^<]+?>', '', clean)
    clean = re.sub('http\S+', '', clean)
    clean = emoji.replace_emoji(clean, replace='')
    clean = re.sub('[^a-zA-Z0-9\s]', '', clean)
    clean = clean.strip()
    return clean

def split_dataset(dataset: Dataset, train_f=0.8, val_f=0.1) -> list:
    train_size = int(len(dataset) * train_f)
    val_size = int(len(dataset) * val_f)
    test_size = len(dataset) - train_size - val_size
    return random_split(dataset=dataset, lengths=(train_size, val_size, test_size))
