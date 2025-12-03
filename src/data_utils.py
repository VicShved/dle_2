import re
import emoji
import pandas as pd
from torch.utils.data import Dataset, random_split, DataLoader, Subset
from sklearn.model_selection import train_test_split


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

def split_data(dataset: list, train_size=0.8, test_size=0.1) -> tuple:
    train, test = train_test_split(dataset, train_size=train_size, test_size=test_size, random_state=42)
    return train, test
