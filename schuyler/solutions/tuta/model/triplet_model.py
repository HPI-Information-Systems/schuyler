
    

class CtcHead(nn.Module):
    def __init__(self, config):
        super(CtcHead, self).__init__()
        self.uniform_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = act.ACT_FCN[config.hidden_act]
        self.tanh = nn.Tanh()
        self.predict_linear = nn.Linear(config.hidden_size, config.num_ctc_type)
        self.loss = nn.CrossEntropyLoss()

        self.aggregator = config.aggregator
        self.aggr_funcs = {"sum": self.token_sum, 
                           "avg": self.token_avg}
    
    def token_sum(self, token_states, indicator):
        """take the sum of token encodings (not including [SEP]s) as cell encodings """
        x_mask = indicator.unsqueeze(1)                      # [batch_size, 1, seq_len]
        y_mask = x_mask.transpose(-1, -2)                    # [batch_size, seq_len, 1]
        mask_matrix = y_mask.eq(x_mask).float()              # [batch_size, seq_len, seq_len]
        sum_states = mask_matrix.matmul(token_states)        # [batch_size, seq_len, hidden_size]
        return sum_states
    
    def token_avg(self, token_states, indicator):
        """take the average of token encodings (not including [SEP]s) as cell encodings """
        x_mask = indicator.unsqueeze(1)                      # [batch_size, 1, seq_len]
        y_mask = x_mask.transpose(-1, -2)                    # [batch_size, seq_len, 1]
        mask_matrix = y_mask.eq(x_mask).float()              # [batch_size, seq_len, seq_len]
        sum_matrix = torch.sum(mask_matrix, dim=-1)
        mask_matrix = mask_matrix.true_divide(sum_matrix.unsqueeze(-1))
        cell_states = mask_matrix.matmul(token_states)  # [batch_size, seq_len, hidden_size]
        return cell_states

    def forward(self, encoded_states, indicator, ctc_label):
        # get cell encodings from token sequence
        cell_states = self.aggr_funcs[self.aggregator](encoded_states, indicator)

        ctc_label = ctc_label.contiguous().view(-1)
        cell_states = cell_states.contiguous().view(ctc_label.size()[0], -1)
        ctc_logits = cell_states[ctc_label > -1, :]       # [batch_total_cell_num, hidden_size]
        ctc_label = ctc_label[ctc_label > -1]

        # separator
        sep_logits = self.uniform_linear(ctc_logits[0::2, :])
        sep_logits = self.tanh(sep_logits)
        sep_logits = self.predict_linear(sep_logits)
        sep_predict = sep_logits.argmax(dim=-1)
        sep_labels = ctc_label[0: : 2]
        # sep_correct = torch.sum(sep_predict.eq(sep_labels).float())
        sep_loss = self.loss(sep_logits, sep_labels)
        # sep_count = torch.tensor(sep_logits.size()[0] + 1e-6)

        # token-sum
        tok_logits = self.uniform_linear(ctc_logits[1::2, :])
        tok_logits = self.tanh(tok_logits)
        tok_logits = self.predict_linear(tok_logits)
        tok_predict = tok_logits.argmax(dim=-1)  
        tok_labels = ctc_label[1: : 2]                                  # [batch-variant copied num]
        # tok_correct = torch.sum(tok_predict.eq(tok_labels).float())   # scalar
        tok_loss = self.loss(tok_logits, tok_labels)                    # scalar
        # tok_count = torch.tensor(tok_logits.size()[0] + 1e-6)         # 1d tensor
        # return (sep_loss, sep_correct, sep_count), (tok_loss, tok_correct, tok_count)
        return (sep_loss, sep_predict, sep_labels), (tok_loss, tok_predict, tok_labels)

    
class TtcHead(nn.Module):
    """Fine-tuning head for the task of table type classification."""

    def __init__(self, config):
        super(TtcHead, self).__init__()
        self.uniform_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = act.ACT_FCN[config.hidden_act]
        # self.tanh = nn.Tanh()
        self.predict_linear = nn.Linear(config.hidden_size, config.num_table_types)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, encoded_states, ttc_label, return_prediction=True):
        """Predict table types with the transformed CLS, then compute loss against the ttc_label. 
        
        Args:
            encoded_states <float> [batch-size, seq-len, hidden-size]: representation of the last hidden layer.
            ttc_label <int> [batch-size]: type of the table.
        Returns: 
            loss <float> []: computed cross-entropy loss.
            ttc_logits <float> [batch-size, num_table_types]: logits over table types,
            *prediction <int> [batch-size]: predicted table type.
        """
        transformed_states = self.uniform_linear(encoded_states)   # [batch-size, seq-len, hidden-size]
        table_state = transformed_states[:, 0, :]                  # [batch-size, hidden-size]
        table_state = self.act_fn(table_state)                     # [batch-size, hidden-size]
        ttc_logits = self.predict_linear(table_state)              # [batch-size, hidden-size]
        loss = self.loss(ttc_logits, ttc_label)                    # []
        
        if return_prediction == True:
            prediction = ttc_logits.argmax(dim=-1)                 # [batch-size]
            return loss, prediction
        
        return loss, ttc_logits
