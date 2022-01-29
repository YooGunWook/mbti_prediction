from torch.utils.data import Dataset
import torch
import kss
import json
import os
import tqdm


class DataSet(Dataset):
    def __init__(self, path, data_jsons, tokenizer):
        self.mbti_dict = {
            0: {"e": 0, "i": 1},
            1: {"s": 0, "n": 1},
            2: {"t": 0, "f": 1},
            3: {"j": 0, "p": 1},
        }
        self.labels = []
        self.sentences = []
        self.attention_masks = []

        # Data preprocessing
        for data_name in tqdm.tqdm(data_jsons):
            data_path = path + "/" + data_name
            with open(data_path, "r") as f:
                data = json.load(f)

            for article_num in data:
                tmp_label = []
                article_info = data[article_num]
                article = article_info["article"]
                writer = article_info["writer"].split("")

                # MBTI labelling
                for idx in range(4):
                    tmp_label.append(self.mbti_dict[idx][writer[idx]])
                self.labels.append(tmp_label)

                # article preprocessing
                ## need to discuss about sentence length

        # list to torch
        self.labels = torch.tensor(self.labels)

        def __len__(self):
            return len(self.label)

        def __getitem__(self, idx):
            sentence = self.sentences[idx]
            attention_mask = self.attention_masks[idx]
            label = self.labels[idx]
            return sentence, attention_mask, label


if __name__ == "__main__":
    path = "./data"
    data_jsons = os.listdir(path)
