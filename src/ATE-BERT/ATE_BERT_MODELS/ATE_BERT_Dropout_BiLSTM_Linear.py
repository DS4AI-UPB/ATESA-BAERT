import torch

class ATE_BERT_Dropout_BiLSTM_Linear(torch.nn.Module):
    def __init__(self, bert, dropout, bilstm_in_features, no_out_labels, device='cpu'):
        super(ATE_BERT_Dropout_BiLSTM_Linear, self).__init__()

        self.device = device

        self.bert = bert.to(device)

        self.dropout = torch.nn.Dropout(dropout)

        self.lstm = torch.nn.LSTM(self.bert.config.hidden_size, bilstm_in_features // 2, bidirectional=True, batch_first=True)

        self.linear = torch.nn.Linear(bilstm_in_features, no_out_labels)

    def forward(self, ids_tensors, masks_tensors):
        bert_output, _ = self.bert(input_ids=ids_tensors, attention_mask=masks_tensors, return_dict=False)
        
        dropout_output = self.dropout(bert_output)

        lstm_output, _ = self.lstm(dropout_output)

        output = self.linear(lstm_output)

        return output
