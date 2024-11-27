import torch

class ABSA_BART_Dropout_BiLSTM_Linear(torch.nn.Module):
    def __init__(self, bart, bart_hidden_size, bilstm_in_features, no_out_labels,  
                dropout=0.3, device='cpu'):
        super(ABSA_BART_Dropout_BiLSTM_Linear, self).__init__()          
        
        self.device = device

        self.bart = bart

        self.dropout = torch.nn.Dropout(dropout)

        self.lstm = torch.nn.LSTM(bart_hidden_size, bilstm_in_features // 2, bidirectional=True, batch_first=True)

        self.fc = torch.nn.Linear(bilstm_in_features, no_out_labels)
    
    def forward(self, ids, mask):
        embedded = self.bart(ids, attention_mask=mask, return_dict=False)

        dropout_output = self.dropout(embedded[0]).to(self.device)

        lstm_output, _ = self.lstm(dropout_output)
                
        fc_out = self.fc(lstm_output)

        return fc_out