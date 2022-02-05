from transformers import ElectraTokenizer
from torch.utils.data import Dataset
import torch
import tqdm


class DataSet(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.mbti_dict = {
            0: {"e": 0, "i": 1},
            1: {"s": 0, "n": 1},
            2: {"t": 0, "f": 1},
            3: {"j": 0, "p": 1},
        }
        self.labels = []  # ex: [[0,1,0,1]]
        self.articles = []
        self.attention_masks = []

        # Data preprocessing
        for row in tqdm.tqdm(data):
            article = row["article"]
            writer = row["writer"]

            # article preprocessing
            tmp_article = []
            tmp_attention = []
            for article in article:
                encoded_dict = tokenizer.encode_plus(
                    article,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                )
                tmp_article.append(encoded_dict["input_ids"])
                tmp_attention.append(encoded_dict["attention_mask"])

            # MBTI labelling
            tmp_label = []
            for idx in range(4):
                mbti = writer[idx]
                tmp_label.append(self.mbti_dict[idx][mbti])
            self.labels.append(tmp_label)
            self.articles.append(tmp_article)
            self.attention_masks.append(tmp_attention)

        # list to torch
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        article = self.articles[idx]
        attention_mask = self.attention_masks[idx]
        label = self.labels[idx]
        return article, attention_mask, label


if __name__ == "__main__":
    import pandas as pd

    tokenizer = ElectraTokenizer.from_pretrained(
        "monologg/koelectra-base-v3-discriminator"
    )
    data_path = "./example.csv"
    data = pd.read_csv(data_path)
    tmp_dataset = DataSet(data, tokenizer, 200)
    print(tmp_dataset.__getitem__(3))
