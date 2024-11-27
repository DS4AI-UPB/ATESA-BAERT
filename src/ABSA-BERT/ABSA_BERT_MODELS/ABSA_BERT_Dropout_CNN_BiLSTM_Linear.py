import torch

class ABSA_BERT_Dropout_CNN_BiLSTM_Linear(torch.nn.Module):
    def __init__(self, bert, bert_seq_len, bilstm_in_features, no_out_labels,  
                conv_out_channels, conv_kernel_size, dropout=0.3, conv_dilation=1, conv_padding=0, conv_stride=1, device='cpu'):
        super(ABSA_BERT_Dropout_CNN_BiLSTM_Linear, self).__init__()          
        
        self.device = device

        self.bert = bert

        self.dropout = torch.nn.Dropout(dropout)

        self.convlayer = torch.nn.Conv1d(self.bert.config.hidden_size, conv_out_channels, conv_kernel_size, dilation=conv_dilation, padding=conv_padding)

        conv_l_out_size = (bert_seq_len + 2 * conv_padding - conv_dilation * (conv_kernel_size - 1) - 1) // conv_stride + 1

        self.lstm = torch.nn.LSTM(conv_l_out_size, bilstm_in_features // 2, bidirectional=True, batch_first=True)

        self.fc = torch.nn.Linear(bilstm_in_features, no_out_labels)
    
    def forward(self, ids, mask):
        last_hidden_state, _ = self.bert(ids, attention_mask=mask, return_dict=False)

        embedded = self.dropout(last_hidden_state.to(self.device)).to(self.device).permute(0, 2, 1)

        conv_output = self.convlayer(embedded).to(self.device)

        lstm_output, _ = self.lstm(conv_output)
                
        fc_out = self.fc(lstm_output)

        return fc_out