import logging
import numpy as np

import torch
import torch.nn.functional as F

from helpers.layers import Embedding, masked_mean, LSTMCell, FastUniLSTM

logger = logging.getLogger(__name__)


class LSTM_DQN(torch.nn.Module):
    model_name = 'lstm_dqn'

    def __init__(self, model_config, word_vocab, verb_map, noun_map, enable_cuda=False):
        super(LSTM_DQN, self).__init__()
        self.model_config = model_config
        self.enable_cuda = enable_cuda
        self.word_vocab_size = len(word_vocab)
        self.id2word = word_vocab
        self.n_actions = len(verb_map)
        self.n_objects = len(noun_map)
        self.read_config()
        self._def_layers()
        self.init_weights()
        # self.print_parameters()

    def print_parameters(self):
        amount = 0
        for p in self.parameters():
            amount += np.prod(p.size())
        print("total number of parameters: %s" % (amount))
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        amount = 0
        for p in parameters:
            amount += np.prod(p.size())
        print("number of trainable parameters: %s" % (amount))

    def read_config(self):
        # model config
        config = self.model_config[self.model_name]
        self.embedding_size = config['embedding_size']
        self.encoder_rnn_hidden_size = config['encoder_rnn_hidden_size']
        self.action_scorer_hidden_dim = config['action_scorer_hidden_dim']
        self.dropout_between_rnn_layers = config['dropout_between_rnn_layers']

    def _def_layers(self):

        # word embeddings
        self.word_embedding = Embedding(embedding_size=self.embedding_size,
                                        vocab_size=self.word_vocab_size,
                                        enable_cuda=self.enable_cuda)

        # lstm encoder
        self.encoder = FastUniLSTM(ninp=self.embedding_size,
                                   nhids=self.encoder_rnn_hidden_size,
                                   dropout_between_rnn_layers=self.dropout_between_rnn_layers)

        # Recurrent network for temporal dependencies (a.k.a history).

        self.action_scorer_shared_recurrent = LSTMCell(input_size=self.encoder_rnn_hidden_size[-1],
                                                       hidden_size=self.action_scorer_hidden_dim)

        self.action_scorer_shared = torch.nn.Linear(self.encoder_rnn_hidden_size[-1], self.action_scorer_hidden_dim)
        self.action_scorer_action = torch.nn.Linear(self.action_scorer_hidden_dim, self.n_actions, bias=False)
        self.action_scorer_object = torch.nn.Linear(self.action_scorer_hidden_dim, self.n_objects, bias=False)
        self.fake_recurrent_mask = None

    def init_weights(self):
        torch.nn.init.xavier_uniform(self.action_scorer_shared.weight.data, gain=1)
        torch.nn.init.xavier_uniform(self.action_scorer_action.weight.data, gain=1)
        torch.nn.init.xavier_uniform(self.action_scorer_object.weight.data, gain=1)
        self.action_scorer_shared.bias.data.fill_(0)

    def representation_generator(self, _input_words):
        embeddings, mask = self.word_embedding.forward(_input_words)  # batch x time x emb
        encoding_sequence, _, _ = self.encoder.forward(embeddings, mask)  # batch x time x h
        mean_encoding = masked_mean(encoding_sequence, mask)  # batch x h
        return mean_encoding

    def recurrent_action_scorer(self, state_representation, last_hidden=None, last_cell=None):
        # state representation: batch x input
        # last hidden / last cell: batch x hid
        if self.fake_recurrent_mask is None or self.fake_recurrent_mask.size(0) != state_representation.size(0):
            self.fake_recurrent_mask = torch.autograd.Variable(torch.ones(state_representation.size(0),))
            if self.enable_cuda:
                self.fake_recurrent_mask = self.fake_recurrent_mask.cuda()

        new_h, new_c = self.action_scorer_shared_recurrent.forward(state_representation, self.fake_recurrent_mask, last_hidden, last_cell)
        action_rank = self.action_scorer_action.forward(new_h)  # batch x n_action
        object_rank = self.action_scorer_object.forward(new_h)  # batch x n_object
        return action_rank, object_rank, new_h, new_c

    def action_scorer(self, state_representation):
        hidden = self.action_scorer_shared.forward(state_representation)  # batch x hid
        hidden = F.relu(hidden)  # batch x hid
        action_rank = self.action_scorer_action.forward(hidden)  # batch x n_action
        object_rank = self.action_scorer_object.forward(hidden)  # batch x n_object
        return action_rank, object_rank
