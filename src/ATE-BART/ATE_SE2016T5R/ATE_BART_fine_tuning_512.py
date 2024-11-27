import os
import sys

sys.path.insert(0, os.path.dirname(os.getcwd())) 

import time
import gc
import json

import numpy as np
import pandas as pd

from transformers import BartTokenizer, BartModel
from transformers import logging

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from InputDataset import InputDataset

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence

import matplotlib.pyplot as plt

from torch import cuda

from ate_bart_models.ATE_BART_Dropout_Linear import ATE_BART_Dropout_Linear

device = 'cuda' if cuda.is_available() else 'cpu'
logging.set_verbosity_error() 

print(torch.cuda.get_device_name(0))
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory // 1024 ** 3} GB")

def clear_memory():
    torch.cuda.empty_cache()

    with torch.no_grad():
        torch.cuda.empty_cache()

    gc.collect()

DATASET = 'ATE_SemEval16_Restaurants_train.json'

df = pd.json_normalize(json.load(open(DATASET)))

df.head()

TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4

EPOCHS = 2

LEARNING_RATE = 1e-5

TRAIN_SPLIT = 0.8
SEQ_LEN = 512

NO_RUNS = 10

BART_FINE_TUNED_OUTPUT = '../../../results/ATE/SemEval16 - Task 5 - Restaurants/models/bart_fine_tuned_512.pth'

MODEL_OUTPUT = '../../../results/ATE/SemEval16 - Task 5 - Restaurants/models/bart_pre_trained_dropout_linear_512.pth'
STATS_OUTPUT = '../../../results/ATE/SemEval16 - Task 5 - Restaurants/stats/bart_pre_trained_dropout_linear_512.csv'

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

best_accuracy = 0.0

def create_mini_batch(samples):
    ids_tensors = [s[1] for s in samples]
    ids_tensors[0] = torch.nn.ConstantPad1d((0, SEQ_LEN - len(ids_tensors[0])), 0)(ids_tensors[0])
    ids_tensors = pad_sequence(ids_tensors, batch_first=True).to(device)

    tags_tensors = [s[2] for s in samples]
    tags_tensors[0] = torch.nn.ConstantPad1d((0, SEQ_LEN - len(tags_tensors[0])), 0)(tags_tensors[0])
    tags_tensors = pad_sequence(tags_tensors, batch_first=True).to(device)
    
    masks_tensors = torch.zeros(ids_tensors.shape, dtype=torch.long).to(device)
    masks_tensors = masks_tensors.masked_fill(ids_tensors != 0, 1).to(device)
    
    return ids_tensors, tags_tensors, masks_tensors

def train(epoch, model, loss_fn, optimizer, dataloader):
    model.train()

    dataloader_len = len(dataloader)

    losses = []

    for _,data in enumerate(dataloader, 0):
        optimizer.zero_grad()

        ids_tensors, tags_tensors, masks_tensors = data

        outputs = model(ids_tensors, masks_tensors)

        loss = loss_fn(outputs.view(-1, 3), tags_tensors.view(-1))

        losses.append(loss.item())
        
        if _ % (dataloader_len // 10) == 0:
            print(f"Epoch: {epoch}/{EPOCHS}, Batch: {_}/{dataloader_len}, Loss: {loss.item()}")
        
        loss.backward()
        
        optimizer.step()
    
    return losses

def validation(model, dataloader, loss_fn):
    model.eval()
    
    fin_targets=[]
    fin_outputs=[]

    losses = []

    with torch.no_grad():
        for _, data in enumerate(dataloader, 0):
            ids_tensors, tags_tensors, masks_tensors = data
            ids_tensors = ids_tensors.to(device)
            tags_tensors = tags_tensors.to(device)
            masks_tensors = masks_tensors.to(device)

            outputs = model(ids_tensors, masks_tensors)
            
            _, predictions = torch.max(outputs, dim=2)

            losses.append(loss_fn(outputs.view(-1, 3), tags_tensors.view(-1)).item())

            fin_outputs += list([int(p) for pred in predictions for p in pred])
            fin_targets += list([int(tag) for tags_tensor in tags_tensors for tag in tags_tensor])

    return fin_outputs, fin_targets, losses

results = pd.DataFrame(columns=['accuracy','precision_score_micro','precision_score_macro','recall_score_micro','recall_score_macro','f1_score_micro','f1_score_macro', 'execution_time'])

for i in range(NO_RUNS):
    # clear cache cuda
    torch.cuda.empty_cache()
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()

    start_time = time.time()

    print(f"Run {i + 1}/{NO_RUNS}")

    train_dataset = df.sample(frac=TRAIN_SPLIT)
    test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    training_set = InputDataset(train_dataset, tokenizer)
    testing_set = InputDataset(test_dataset, tokenizer)

    train_dataloader = DataLoader(
        training_set,
        sampler = RandomSampler(train_dataset),
        batch_size = TRAIN_BATCH_SIZE,
        drop_last = True,
        collate_fn=create_mini_batch
    )

    validation_dataloader = DataLoader(
        testing_set,
        sampler = SequentialSampler(testing_set),
        batch_size = VALID_BATCH_SIZE,
        drop_last = True,
        collate_fn=create_mini_batch
    )

    model = ATE_BART_Dropout_Linear(BartModel.from_pretrained('facebook/bart-base'), bart_hidden_size=768, dropout=0.3, no_out_labels=3, device=device).to(device)

    optimizer = torch.optim.Adam(params = model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_losses = []
    for epoch in range(EPOCHS):
        losses = train(epoch, model, loss_fn, optimizer, train_dataloader)

        train_losses += losses
    
    plt.title(f'Train Loss for run {i + 1}/{NO_RUNS}')
    plt.plot(train_losses)
    plt.savefig(f'../../../results/ATE/SemEval16 - Task 5 - Restaurants/plots/bart_pt_do_linear/train_loss_run_{i + 1}.png')

    plt.clf()

    outputs, targets, valid_losses = validation(model, validation_dataloader, torch.nn.CrossEntropyLoss())
    
    accuracy = accuracy_score(targets, outputs)
    precision_score_micro = precision_score(targets, outputs, average='micro')
    precision_score_macro = precision_score(targets, outputs, average='macro')
    recall_score_micro = recall_score(targets, outputs, average='micro')
    recall_score_macro = recall_score(targets, outputs, average='macro')
    f1_score_micro = f1_score(targets, outputs, average='micro')
    f1_score_macro = f1_score(targets, outputs, average='macro')

    execution_time = time.time() - start_time

    results.loc[i] = [accuracy,precision_score_micro,precision_score_macro,recall_score_micro,recall_score_macro,f1_score_micro,f1_score_macro, execution_time]

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.bart, BART_FINE_TUNED_OUTPUT)
        torch.save(model, MODEL_OUTPUT)

    del train_dataset
    del test_dataset
    del training_set
    del testing_set
    del model
    del loss_fn
    del optimizer
    del outputs
    del targets

print(results.head())

results.to_csv(STATS_OUTPUT)