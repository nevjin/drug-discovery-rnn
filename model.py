import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.functional import softmax

class RNN(torch.nn.Module):
    def __init__(self, rnn_config):
        super(RNN, self).__init__()

        self.embedding_layer = nn.Embedding(
            num_embeddings=rnn_config['num_embeddings'],
            embedding_dim=rnn_config['embedding_dim'],
            padding_idx=rnn_config['num_embeddings'] - 1
        )
        if rnn_config['rnn_type'] == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=rnn_config['input_size'],
                hidden_size=rnn_config['hidden_size'],
                num_layers=rnn_config['num_layers'],
                batch_first=True,
                dropout=rnn_config['dropout']
            )
        elif rnn_config['rnn_type'] == 'GRU':
            self.rnn = nn.GRU(
                input_size=rnn_config['input_size'],
                hidden_size=rnn_config['hidden_size'],
                num_layers=rnn_config['num_layers'],
                batch_first=True,
                dropout=rnn_config['dropout']
            )
        else:
            raise ValueError(
                "rnn_type should be either 'LSTM' or 'GRU'."
            )

        self.linear = nn.Linear(
            rnn_config['hidden_size'], rnn_config['num_embeddings'] - 2
        )

    def forward(self, data, lengths):
        embeddings = self.embedding_layer(data)

        embeddings = pack_padded_sequence(
            input=embeddings,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False
        )

        embeddings, _ = self.rnn(embeddings)

        embeddings = self.linear(embeddings.data)

        return embeddings

    def sample(self, batch_size, vocab, device, max_length=140, return_log_probs=False):

        start_int = vocab.vocab['<sos>']

        sos = torch.full(
            (batch_size, 1), start_int, dtype=torch.long, device=device
        )
        output = [sos]
        log_probs = []

        x = self.embedding_layer(sos)
        x, hidden = self.rnn(x)

        finish = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for _ in range(max_length):

            logits = self.linear(x)
            probs = softmax(logits, dim=-1)
            sampled = torch.multinomial(probs.squeeze(1), 1)

            if return_log_probs:
                step_log_probs = torch.log(probs.gather(2, sampled.unsqueeze(-1)).squeeze(-1))
                log_probs.append(step_log_probs)

            output.append(sampled)

            eos_sampled = (sampled == vocab.vocab['<eos>']).squeeze(-1)
            finish = torch.logical_or(finish, eos_sampled)
            if torch.all(finish):
                break

            x = self.embedding_layer(sampled)
            x, hidden = self.rnn(x, hidden)

        output = torch.cat(output, dim=-1)

        if return_log_probs:
            log_probs = torch.stack(log_probs, dim=1)
            return output, log_probs

        return output

    def sample_cpu(self, vocab):
        output = []

        start_int = vocab.vocab['<sos>']

        sos = torch.tensor(
            start_int,
            dtype=torch.long
        ).unsqueeze(dim=0
                    ).unsqueeze(dim=0)

        x = self.embedding_layer(sos)
        x, hidden = self.rnn(x)
        x = self.linear(x)
        x = softmax(x, dim=-1)
        x = torch.multinomial(x.squeeze(), 1)
        output.append(x.item())

        while output[-1] != vocab.vocab['<eos>']:
            x = x.unsqueeze(dim=0)
            x = self.embedding_layer(x)
            x, hidden = self.rnn(x, hidden)
            x = self.linear(x)
            x = softmax(x, dim=-1)
            x = torch.multinomial(x.squeeze(), 1)
            output.append(x.item())

        output = [vocab.int2tocken[x] for x in output]

        output.pop()

        output = vocab.combine_list(output)

        return output