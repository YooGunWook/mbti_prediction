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

    def _init_weight_(self):
        # initializing weight
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.xavier_uniform_(self.linear_3.weight)
        nn.init.xavier_uniform_(self.linear_4.weight)

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
    tokenizer = transformers.ElectraTokenizer.from_pretrained(
        "monologg/koelectra-base-v3-discriminator"
    )
    sents = ["안녕하세요. 저는 강아지입니다.", "오늘은 뭐 하는 날인가요?"]
    fin_inputs = []
    fin_attentions = []
    for _ in range(3):
        tmp_inputs = []
        tmp_attentions = []
        for sent in sents:
            encoded_dict = tokenizer.encode_plus(
                sent,
                add_special_tokens=True,
                max_length=15,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
            )
            tmp_inputs.append(encoded_dict["input_ids"])
            tmp_attentions.append(encoded_dict["attention_mask"])
        fin_inputs.append(tmp_inputs)
        fin_attentions.append(tmp_attentions)

    fin_inputs = torch.tensor(fin_inputs)
    fin_attentions = torch.tensor(fin_attentions)
    print(model(fin_inputs, fin_attentions))
