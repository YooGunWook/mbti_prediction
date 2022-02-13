from modules import mbti_dataset
from modules import trainer
from models import model as mbti_model

from transformers import ElectraTokenizer
from torch.utils.data import DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from transformers import (get_linear_schedule_with_warmup, AdamW)

import torch
import glob
import tqdm
import csv
import ast
import yaml
import random
import numpy as np

TRAIN_CONFIG_PATH = "./config/config.yml"

################ config ##############
with open(TRAIN_CONFIG_PATH) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

RANDOM_SEED = config["SEED"]["random_seed"]
SHUFFLE = config["DATALOADER"]["shuffle"]
BATCH_SIZE = config["TRAIN"]["batch_size"]
LEARNING_RATE = config["TRAIN"]["learning_rate"]
EPOCHS = config["TRAIN"]["epochs"]
LAMBDA = config["TRAIN"]["lambda"]
EPS = [float(val) for val in config["TRAIN"]["eps"]]
WEIGHT_DECAY= config["TRAIN"]["weight_decay"]
NUM_WARMUP_STEP = config["TRAIN"]["num_warmup_step"]
THRESHOLD = config["TRAIN"]["threshold"]
EARLY_STOP = config["TRAIN"]["early_stop"]
######################################

# Set random seed, for prevent randomness
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def main():

    # check gpu
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(0))
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    datas_path = glob.glob("./data/*.csv")
    article = []
    target = []
    for data_path in tqdm.tqdm(datas_path):
        with open(data_path, "r") as f:
            data = csv.reader(f)
            for idx, line in enumerate(data):
                if idx == 0:
                    continue
                article.append(ast.literal_eval(line[2]))
                target.append(line[3])

    tokenizer = ElectraTokenizer.from_pretrained(
        "monologg/koelectra-base-v3-discriminator"
    )
    
    # train test split
    X_train, X_valid, y_train, y_valid = train_test_split(
        article, target, stratify=target, random_state=3307, test_size=0.2
    )

    # train valid split
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_valid, y_valid, stratify=y_valid, random_state=3307, test_size=0.2
    )

    # train, valid dataset
    train_dataset = mbti_dataset.DataSet(X_train, y_train, tokenizer, 200)
    valid_dataset = mbti_dataset.DataSet(X_valid, y_valid, tokenizer, 200)
    test_dataset = mbti_dataset.DataSet(X_test, y_test, tokenizer, 200)
    
    trainloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, drop_last=True)
    validloader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, drop_last=True)
    testloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, drop_last=True)
    
    # model init
    model = mbti_model.MBTIClassifier("monologg/koelectra-base-v3-discriminator")
    optimizer = AdamW(
        model.parameters(), lr=LEARNING_RATE, eps=EPS, weight_decay=WEIGHT_DECAY
    )
    total_steps = len(trainloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=NUM_WARMUP_STEP,
        num_training_steps=total_steps,
    )
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    model_trainer = trainer(model, device, loss_fn, optimizer, scheduler, THRESHOLD, EARLY_STOP)
    
    for epoch in EPOCHS:
        print(f"{'Epoch':^7} | {'Batch step':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'type':^9} | {'Elapsed':^9}")
        model_trainer.train(trainloader, validloader, epoch)
    
    
if __name__ == "__main__":
    main()
