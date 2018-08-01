import torch
import numpy as np
import torch.nn.functional as F


def masked_mean(x, m=None, dim=-1):
    """
        mean pooling when there're paddings
        input:  tensor: batch x time x h
                mask:   batch x time
        output: tensor: batch x h
    """
    if m is None:
        return torch.mean(x, dim=dim)
    mask_sum = torch.sum(m, dim=-1)  # batch
    res = torch.sum(x, dim=1)  # batch x h
    res = res / (mask_sum.unsqueeze(-1) + 1e-6)
    return res


class LayerNorm(torch.nn.Module):

    def __init__(self, input_dim):
        super(LayerNorm, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(input_dim))
        self.beta = torch.nn.Parameter(torch.zeros(input_dim))
        self.eps = 1e-6

    def forward(self, x, mask):
        # x:        nbatch x hidden
        # mask:     nbatch
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True) + self.eps)
        output = self.gamma * (x - mean) / (std + self.eps) + self.beta
        return output * mask.unsqueeze(1)


class Embedding(torch.nn.Module):
    '''
    inputs: x:          batch x seq (x is post-padded by 0s)
    outputs:embedding:  batch x seq x emb
            mask:       batch x seq
    '''

    def __init__(self, embedding_size, vocab_size, enable_cuda=False):
        super(Embedding, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.enable_cuda = enable_cuda
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)
        self.init_weights()

    def init_weights(self):
        init_embedding_matrix = self.embedding_init()
        self.embedding_layer.weight = torch.nn.Parameter(init_embedding_matrix)

    def embedding_init(self):
        # Embeddings
        word_embedding_init = np.random.uniform(low=-0.05, high=0.05, size=(self.vocab_size, self.embedding_size))
        word_embedding_init[0, :] = 0
        word_embedding_init = torch.from_numpy(word_embedding_init).float()
        if self.enable_cuda:
            word_embedding_init = word_embedding_init.cuda()
        return word_embedding_init

    def compute_mask(self, x):
        mask = torch.ne(x, 0).float()
        return mask

    def embed(self, words):
        masked_embed_weight = self.embedding_layer.weight
        padding_idx = self.embedding_layer.padding_idx
        X = self.embedding_layer._backend.Embedding.apply(
            words, masked_embed_weight,
            padding_idx, self.embedding_layer.max_norm, self.embedding_layer.norm_type,
            self.embedding_layer.scale_grad_by_freq, self.embedding_layer.sparse)
        return X

    def forward(self, x):
        embeddings = self.embed(x)  # batch x time x emb
        mask = self.compute_mask(x)  # batch x time
        return embeddings, mask


class LSTMCell(torch.nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_layernorm=False, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.use_layernorm = use_layernorm
        self.weight_ih = torch.nn.Parameter(torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = torch.nn.Parameter(torch.FloatTensor(hidden_size, 4 * hidden_size))
        if use_bias:
            self.bias_f = torch.nn.Parameter(torch.FloatTensor(hidden_size))
            self.bias_iog = torch.nn.Parameter(torch.FloatTensor(3 * hidden_size))
        else:
            self.register_parameter('bias', None)
        if self.use_layernorm:
            self.layernorm_i = LayerNorm(input_dim=self.hidden_size * 4)
            self.layernorm_h = LayerNorm(input_dim=self.hidden_size * 4)
            self.layernorm_c = LayerNorm(input_dim=self.hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.orthogonal(self.weight_hh.data)
        torch.nn.init.xavier_uniform(self.weight_ih.data, gain=1)
        if self.use_bias:
            self.bias_f.data.fill_(1.0)
            self.bias_iog.data.fill_(0.0)

    def get_init_hidden(self, bsz, use_cuda):
        h_0 = torch.autograd.Variable(torch.FloatTensor(bsz, self.hidden_size).zero_())
        c_0 = torch.autograd.Variable(torch.FloatTensor(bsz, self.hidden_size).zero_())

        if use_cuda:
            h_0, c_0 = h_0.cuda(), c_0.cuda()

        return h_0, c_0

    def forward(self, input_, mask_, h_0=None, c_0=None, dropped_h_0=None):
        """
        Args:
            input_:     A (batch, input_size) tensor containing input features.
            mask_:      (batch)
            hx:         A tuple (h_0, c_0), which contains the initial hidden
                        and cell state, where the size of both states is
                        (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """
        if h_0 is None or c_0 is None:
            h_init, c_init = self.get_init_hidden(input_.size(0), use_cuda=input_.is_cuda)
            if h_0 is None:
                h_0 = h_init

            if c_0 is None:
                c_0 = c_init

        if dropped_h_0 is None:
            dropped_h_0 = h_0

        # if (mask_.data == 0).all():
        #     return h_0, c_0
        wh = torch.mm(dropped_h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        if self.use_layernorm:
            wi = self.layernorm_i(wi, mask_)
            wh = self.layernorm_h(wh, mask_)
        pre_act = wi + wh
        if self.use_bias:
            pre_act = pre_act + torch.cat([self.bias_f, self.bias_iog]).unsqueeze(0)

        f, i, o, g = torch.split(pre_act, split_size=self.hidden_size, dim=1)
        expand_mask_ = mask_.unsqueeze(1)  # batch x None
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        c_1 = c_1 * expand_mask_ + c_0 * (1 - expand_mask_)
        if self.use_layernorm:
            h_1 = torch.sigmoid(o) * torch.tanh(self.layernorm_c(c_1, mask_))
        else:
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        h_1 = h_1 * expand_mask_ + h_0 * (1 - expand_mask_)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class FastUniLSTM(torch.nn.Module):
    """
    Adapted from https://github.com/facebookresearch/DrQA/
    now supports:   different rnn size for each layer
                    all zero rows in batch (from time distributed layer, by reshaping certain dimension)
    """

    def __init__(self, ninp, nhids, dropout_between_rnn_layers=0.):
        super(FastUniLSTM, self).__init__()
        self.ninp = ninp
        self.nhids = nhids
        self.nlayers = len(self.nhids)
        self.dropout_between_rnn_layers = dropout_between_rnn_layers
        self.stack_rnns()

    def stack_rnns(self):
        rnns = [torch.nn.LSTM(self.ninp if i == 0 else self.nhids[i - 1],
                              self.nhids[i],
                              num_layers=1,
                              bidirectional=False) for i in range(self.nlayers)]
        self.rnns = torch.nn.ModuleList(rnns)

    def forward(self, x, mask):

        def pad_(tensor, n):
            if n > 0:
                zero_pad = torch.autograd.Variable(torch.zeros((n,) + tensor.size()[1:]))
                if x.is_cuda:
                    zero_pad = zero_pad.cuda()
                tensor = torch.cat([tensor, zero_pad])
            return tensor

        """
        inputs: x:          batch x time x inp
                mask:       batch x time
        output: encoding:   batch x time x hidden[-1]
        """
        # Compute sorted sequence lengths
        batch_size = x.size(0)
        lengths = mask.data.eq(1).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # remove non-zero rows, and remember how many zeros
        n_nonzero = np.count_nonzero(lengths)
        n_zero = batch_size - n_nonzero
        if n_zero != 0:
            lengths = lengths[:n_nonzero]
            x = x[:n_nonzero]

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.nlayers):
            rnn_input = outputs[-1]

            # dropout between rnn layers
            if self.dropout_between_rnn_layers > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_between_rnn_layers,
                                          training=self.training)
                rnn_input = torch.nn.utils.rnn.PackedSequence(dropout_input,
                                                              rnn_input.batch_sizes)
            seq, last = self.rnns[i](rnn_input)
            outputs.append(seq)
            if i == self.nlayers - 1:
                # last layer
                last_state = last[0]  # (num_layers * num_directions, batch, hidden_size)
                last_state = last_state[0]  # batch x hidden_size

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = torch.nn.utils.rnn.pad_packed_sequence(o)[0]
        output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)  # batch x time x enc

        # re-padding
        output = pad_(output, n_zero)
        last_state = pad_(last_state, n_zero)

        output = output.index_select(0, idx_unsort)
        last_state = last_state.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != mask.size(1):
            padding = torch.zeros(output.size(0),
                                  mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, torch.autograd.Variable(padding)], 1)

        output = output.contiguous() * mask.unsqueeze(-1)
        return output, last_state, mask

