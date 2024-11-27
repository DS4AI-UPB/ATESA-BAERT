import torch

class ABSA_BART_Dropout_Linear(torch.nn.Module):
    def __init__(self, bart, bart_hidden_size, no_out_labels, dropout=0.3, device='cpu'):
        super(ABSA_BART_Dropout_Linear, self).__init__()          
        
        self.device = device

        self.bart = bart

        self.dropout = torch.nn.Dropout(dropout)

        self.fc = torch.nn.Linear(bart_hidden_size, no_out_labels)
    
    def forward(self, ids, mask):
        embedded = self.bart(ids, attention_mask=mask, return_dict=False)

        dropout_output = self.dropout(embedded[0]).to(self.device)
                
        fc_out = self.fc(dropout_output)

        return fc_out