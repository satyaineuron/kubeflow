import torch
import torch.nn as nn

class BERTSentiment(nn.Module):
    def __init__(self,
                 transformer,
                 output_dim,
                 freeze):

        super().__init__()

        self.transformer = transformer
        # We will not use embedding layer for our text instead we'll be using the pre-trained transformer model
        hidden_dim = transformer.config.hidden_size

        self.fc = nn.Linear(hidden_dim, output_dim)

        # we wrap the transformer in a no_grad to ensure no gradients are calculated over this part of the model
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, ids: torch.Tensor):

        # ids = [batch_size, sequence_length]
        # arg output_attentions to True to return the attention matrix in the output
        output = self.transformer(ids, output_attentions=True)
        # The resulting output is a dictionary containing the last hidden state of the model (hidden) and the attention scores (attention)

        hidden = output.last_hidden_state

        # The [CLS] token representation from the last layer of the transformer is extracted from output tensor
        attention = output.attentions[-1]

        cls_hidden = hidden[:, 0, :]

        # passed through the fully connected layer
        prediction = self.fc(torch.tanh(cls_hidden))

        return prediction
