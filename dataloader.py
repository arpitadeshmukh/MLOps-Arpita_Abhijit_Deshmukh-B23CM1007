import gzip
import json
import random
import requests
import torch
from typing import List, Tuple, Dict

def load_reviews(url: str, max_read=10000, sample_size=1000) -> List[str]:
    reviews = []
    response = requests.get(url, stream=True)

    with gzip.open(response.raw, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_read:
                break
            reviews.append(json.loads(line)["review_text"])

    return random.sample(reviews, min(sample_size, len(reviews)))

def build_splits(genre_urls: Dict[str, str],
                 train_per_genre: int,
                 test_per_genre: int) -> Tuple[list, list, list, list]:

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for genre, url in genre_urls.items():
        reviews = load_reviews(url, sample_size=train_per_genre + test_per_genre)
        random.shuffle(reviews)

        train = reviews[:train_per_genre]
        test = reviews[train_per_genre:]

        train_texts.extend(train)
        train_labels.extend([genre] * len(train))

        test_texts.extend(test)
        test_labels.extend([genre] * len(test))

    return train_texts, train_labels, test_texts, test_labels


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
