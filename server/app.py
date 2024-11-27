import sys
import os

import time

sys.path.insert(0, os.path.dirname(os.getcwd()))
sys.path.insert(0, os.path.dirname(os.path.join(os.getcwd(), '..', 'src/')))
sys.path.insert(0, os.path.dirname(os.path.join(os.getcwd(), '..', 'src/ATE/ate_models')))
sys.path.insert(0, os.path.dirname(os.path.join(os.getcwd(), '..', 'src/ABSA/absa_models')))

from flask import Flask, request, jsonify

import torch 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer
import pandas as pd

from ABSA import InputDataset as ABSAInputDataset
from ATE import InputDataset as ATEInputDataset

import spacy

app = Flask(__name__)

device = 'cpu'

# ate_model = torch.load('../results/ATE/MAMS/models/bert_pre_trained_dropout_linear_512.pth').to(device)
# absa_model = torch.load('../results/ABSA/MAMS/models/bert_pre_trained_dropout_linear_512.pth').to(device)

# ate_model = torch.load('../results/ATE/MAMS/models/bert_pre_trained_dropout_bilstm_linear_512.pth').to(device)
# absa_model = torch.load('../results/ABSA/MAMS/models/bert_pre_trained_dropout_bilstm_linear_512.pth').to(device)

# ate_model = torch.load('../results/ATE/MAMS/models/bert_pre_trained_dropout_cnn_bilstm_linear_512.pth', map_location=torch.device('cpu'))
# absa_model = torch.load('../results/ABSA/MAMS/models/bert_pre_trained_dropout_cnn_bilstm_linear_512.pth', map_location=torch.device('cpu'))

# ate_model = torch.load('../results/ATE/MAMS/models/bert_fine_tuned_dropout_linear_512.pth').to(device)
# absa_model = torch.load('../results/ABSA/MAMS/models/bert_fine_tuned_dropout_linear_512.pth').to(device)

ate_model = torch.load('../results/ATE/MAMS/models/bert_fine_tuned_dropout_bilstm_linear_512.pth').to(device)
absa_model = torch.load('../results/ABSA/MAMS/models/bert_fine_tuned_dropout_bilstm_linear_512.pth').to(device)

# ate_model = torch.load('../results/ATE/MAMS/models/bert_fine_tuned_dropout_cnn_bilstm_linear_512.pth').to(device)
# absa_model = torch.load('../results/ABSA/MAMS/models/bert_fine_tuned_dropout_cnn_bilstm_linear_512.pth').to(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
nlp = spacy.load('en_core_web_lg')

SEQ_LEN = 512

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

def validation(dataloader, model):
    fin_outputs = []
    fin_targets = []

    with torch.no_grad():
        for _, data in enumerate(dataloader, 0):
            ids_tensors, tags_tensors, masks_tensors = data
            ids_tensors = ids_tensors.to(device)
            tags_tensors = tags_tensors.to(device)
            masks_tensors = masks_tensors.to(device)

            outputs = model(ids_tensors, masks_tensors)
            
            _, predictions = torch.max(outputs, dim=2)

            fin_outputs += list([int(p) for pred in predictions for p in pred])
            fin_targets += list([int(tag) for tags_tensor in tags_tensors for tag in tags_tensor])

    return fin_outputs, fin_targets

def ate_extraction(df):
    ate_dataset = ATEInputDataset.InputDataset(df, tokenizer)

    ate_dataloader = DataLoader(
        ate_dataset,
        sampler = SequentialSampler(ate_dataset),
        batch_size = 1,
        drop_last = True,
        collate_fn=create_mini_batch
    )

    fin_outputs, _ = validation(ate_dataloader, ate_model)

    return fin_outputs

def sa_extraction(df):
    absa_dataset = ABSAInputDataset.InputDataset(df, tokenizer)

    absa_dataloader = DataLoader(
        absa_dataset,
        sampler = SequentialSampler(absa_dataset),
        batch_size = 1,
        drop_last = True,
        collate_fn=create_mini_batch
    )

    fin_outputs, _ = validation(absa_dataloader, absa_model)

    return fin_outputs

@app.route('/predict', methods=["POST"])
def absa():
    start = time.time()

    input_json = request.get_json(force=True)

    df = pd.DataFrame(columns=['text','tokens','iob_aspect_tags', 'absa_tags'])

    sentences = input_json['sentences']

    for text in sentences:
        doc = nlp(text)

        df.loc[len(df)] = [text, [token.text for token in doc], [0 for token in doc], [0 for token in doc]]
    
    ate_outputs = ate_extraction(df)
    sa_outputs = sa_extraction(df)

    tokens = df.iloc[0]['tokens']

    dictToReturn = {
        'tokens': tokens,
        'ate_outputs': ate_outputs[:len(tokens)],
        'sa_outputs': sa_outputs[:len(tokens)]
    }

    response = jsonify(dictToReturn)
    response.headers.add('Access-Control-Allow-Origin', '*')

    end = time.time()
    print('Execution time: ', end - start)

    return response
