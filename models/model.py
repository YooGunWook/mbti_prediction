import torch
from torch import nn
import transformers


class MBTIClassifier(nn.Module):
    def __init__(self, model_name):
        super(MBTIClassifier, self).__init__()
        self.model = transformers.AutoModel.from_pretrained(model_name)
        self.linear1 = nn.Linear(
            self.model.config.hidden_size, self.model.config.hidden_size // 8
        )
        self.linear2 = nn.Linear(
            self.model.config.hidden_size // 8, self.model.config.hidden_size // 32
        )
        self.layernorm2 = nn.LayerNorm(self.model.config.hidden_size // 32, eps=1e-6)
        self.linear3 = nn.Linear(
            self.model.config.hidden_size // 32, self.model.config.hidden_size // 64
        )
        self.layernorm3 = nn.LayerNorm(self.model.config.hidden_size // 64, eps=1e-6)
        self.linear4 = nn.Linear(self.model.config.hidden_size // 64, 4)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, sent, attention_mask, cls_pos=0):
        pooled_out = []

        # [CLS] based document embedding
        for idx in range(sent.shape[0]):
            output = self.model(input_ids=sent[idx], attention_mask=attention_mask[idx])
            tmp_out = torch.cat(
                tuple(i[cls_pos].unsqueeze(0) for i in output[-1]), dim=0
            )
            tmp_out = torch.mean(tmp_out, dim=0)
            pooled_out.append(tmp_out.unsqueeze(0))

        # first layer
        pooled_out = torch.cat(pooled_out, dim=0)
        pooled_out = self.linear1(pooled_out)
        pooled_out = self.relu(pooled_out)
        pooled_out = self.dropout(pooled_out)

        # second layer
        pooled_out = self.linear2(pooled_out)
        pooled_out = self.relu(pooled_out)
        pooled_out = self.dropout(pooled_out)
        pooled_out = self.layernorm2(pooled_out)

        # third layer
        pooled_out = self.linear3(pooled_out)
        pooled_out = self.relu(pooled_out)
        pooled_out = self.dropout(pooled_out)
        pooled_out = self.layernorm3(pooled_out)

        # fourth layer
        pooled_out = self.linear4(pooled_out)
        res = self.sigmoid(pooled_out)

        return res


if __name__ == "__main__":
    model = MBTIClassifier("monologg/koelectra-base-v3-discriminator")
