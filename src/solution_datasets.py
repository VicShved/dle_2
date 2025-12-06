from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

class PredictWordDataset(Dataset):
    def __init__(self, texts, tokenizer, min_len=4):
        self._samples = []
        self.min_len = min_len
        self.max_len = 0
        for line in texts:
            token_ids = tokenizer.encode(line.replace("\n", "")) #
            if len(token_ids) < min_len:
                continue
            self.max_len = max(self.max_len, len(token_ids))
            for i in range(min_len - 1, len(token_ids)):
                self._samples.append((token_ids[:i], token_ids[i]))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        x, y = self._samples[index]
        return F.pad(torch.tensor(x), (0, self.max_len - len(x) - 1), value=0), torch.tensor(y)


class ValuateDataset(Dataset):
    def __init__(self, texts, min_len=4):
        self._samples = []
        for line in texts:
            len_x = len(line) *3//4
            self._samples.append((line[:len_x], line[len_x:]))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        x, y = self._samples[index]
        return x, y
