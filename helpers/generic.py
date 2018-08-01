import os
import numpy as np
from os.path import join as pjoin
from nltk.tokenize import word_tokenize as wt

import torch
from torch.autograd import Variable

from textworld.utils import maybe_mkdir


class SlidingAverage(object):
    def __init__(self, name, steps=100):
        self.name = name
        self.steps = steps
        self.t = 0
        self.ns = []
        self.avgs = []

    def add(self, n):
        self.ns.append(n)
        if len(self.ns) > self.steps:
            self.ns.pop(0)
        self.t += 1
        if self.t % self.steps == 0:
            self.avgs.append(self.value)

    @property
    def value(self):
        if len(self.ns) == 0: return 0
        return sum(self.ns) / len(self.ns)

    def __str__(self):
        return "%s=%.4f" % (self.name, self.value)

    def __gt__(self, value): return self.value > value
    def __lt__(self, value): return self.value < value

    def state_dict(self):
        return {'t': self.t,
                'ns': tuple(self.ns),
                'avgs': tuple(self.avgs)}

    def load_state_dict(self, state):
        self.t = state["t"]
        self.ns = list(state["ns"])
        self.avgs = list(state["avgs"])


def to_np(x):
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy()


def to_pt(np_matrix, enable_cuda=False, type='long'):
    if type == 'long':
        if enable_cuda:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.LongTensor).cuda())
        else:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.LongTensor))
    elif type == 'float':
        if enable_cuda:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.FloatTensor).cuda())
        else:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.FloatTensor))


def get_experiment_dir(config):
    env_id = config['general']['env_id']
    exps_dir = config['general']['experiments_dir']
    exp_tag = config['general']['experiment_tag']
    exp_dir = pjoin(exps_dir, env_id + "_" + exp_tag)
    return maybe_mkdir(exp_dir)


def dict2list(id2w_dict):
    res = []
    for item in id2w_dict:
        res.append(id2w_dict[item])
    return res


def _words_to_ids(words, word2id):
    ids = []
    for word in words:
        try:
            ids.append(word2id[word])
        except KeyError:
            ids.append(1)
    return ids


def preproc(s, str_type='None', lower_case=False):
    s = s.replace("\n", ' ')
    if s.strip() == "":
        return ["nothing"]
    if str_type == 'description':
        s = s.split("=-")[1]
    elif str_type == 'inventory':
        s = s.split("carrying")[1]
        if s[0] == ':':
            s = s[1:]
    elif str_type == 'feedback':
        if "Welcome to Textworld" in s:
            s = s.split("Welcome to Textworld")[1]
        if "-=" in s:
            s = s.split("-=")[0]
    s = s.strip()
    if len(s) == 0:
        return ["nothing"]
    tokens = wt(s)
    if lower_case:
        tokens = [t.lower() for t in tokens]
    return tokens


def max_len(list_of_list):
    return max(map(len, list_of_list))


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    '''
    FROM KERAS
    Pads each sequence to the same length:
    the length of the longest sequence.
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
