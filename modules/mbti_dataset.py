from transformers import ElectraTokenizer
from torch.utils.data import Dataset
import torch
import json
import os
import tqdm


class DataSet(Dataset):
    def __init__(self, path, data_jsons, tokenizer, max_length):
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
        for data_name in tqdm.tqdm(data_jsons):
            data_path = path + "/" + data_name
            with open(data_path, "r") as f:
                data = json.load(f)

            for article_num in data:
                article_info = data[article_num]
                articles = article_info["article"]
                writers = article_info["writer"]

                # article preprocessing
                tmp_article = []
                tmp_attention = []
                for article in articles:
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
                for writer in writers:
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
    path = "./data"
    data_jsons = os.listdir(path)
    tokenizer = ElectraTokenizer.from_pretrained(
        "monologg/koelectra-base-v3-discriminator"
    )
    tmp_dataset = DataSet(path, data_jsons, tokenizer, 200)
    print(tmp_dataset.__getitem__(3))
