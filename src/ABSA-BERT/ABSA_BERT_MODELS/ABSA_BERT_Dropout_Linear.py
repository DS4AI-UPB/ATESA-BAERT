import torch

class ABSA_BERT_Dropout_Linear(torch.nn.Module):
    def __init__(self, bert, dropout, no_out_labels, device='cpu'):
        super(ABSA_BERT_Dropout_Linear, self).__init__()

        self.device = device

        self.bert = bert.to(device)

        self.dropout = torch.nn.Dropout(dropout)

        self.linear = torch.nn.Linear(self.bert.config.hidden_size, no_out_labels)

    def forward(self, ids_tensors, masks_tensors):
        bert_outputs, _ = self.bert(input_ids=ids_tensors, attention_mask=masks_tensors, return_dict=False)

        dropout_output = self.dropout(bert_outputs)

        output = self.linear(dropout_output)

        return output
