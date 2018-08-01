import logging
import numpy as np
from collections import namedtuple
import random
# from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F

from helpers.model import LSTM_DQN
from helpers.generic import to_np, to_pt, preproc, _words_to_ids, pad_sequences, max_len
logger = logging.getLogger(__name__)

import gym
import gym_textworld  # Register all textworld environments.


Transition = namedtuple('Transition', ('observation_id_list', 'v_idx', 'n_idx',
                                       'reward', 'mask', 'done',
                                       'next_observation_id_list',
                                       'observation_str'))


class ReplayMemory(object):

    def __init__(self, capacity=100000):
        # vanilla replay memory
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory(object):

    def __init__(self, capacity=100000, priority_fraction=0.0):
        # prioritized replay memory
        self.priority_fraction = priority_fraction
        self.alpha_capacity = int(capacity * priority_fraction)
        self.beta_capacity = capacity - self.alpha_capacity
        self.alpha_memory, self.beta_memory = [], []
        self.alpha_position, self.beta_position = 0, 0

    def push(self, is_prior=False, *args):
        """Saves a transition."""
        if is_prior:
            if len(self.alpha_memory) < self.alpha_capacity:
                self.alpha_memory.append(None)
            self.alpha_memory[self.alpha_position] = Transition(*args)
            self.alpha_position = (self.alpha_position + 1) % self.alpha_capacity
        else:
            if len(self.beta_memory) < self.beta_capacity:
                self.beta_memory.append(None)
            self.beta_memory[self.beta_position] = Transition(*args)
            self.beta_position = (self.beta_position + 1) % self.beta_capacity

    def sample(self, batch_size):
        from_alpha = min(int(self.priority_fraction * batch_size), len(self.alpha_memory))
        from_beta = min(batch_size - int(self.priority_fraction * batch_size), len(self.beta_memory))
        res = random.sample(self.alpha_memory, from_alpha) + random.sample(self.beta_memory, from_beta)
        random.shuffle(res)
        return res

    def __len__(self):
        return len(self.alpha_memory) + len(self.beta_memory)


class ObservationHistoryCache(object):

    def __init__(self, capacity=1):
        # vanilla replay memory
        self.capacity = capacity
        self.memory = []
        self.reset()

    def push(self, stuff):
        """stuff is list."""
        for i in range(1, self.capacity):
            self.memory[i - 1] = self.memory[i]
        self.memory[-1] = stuff

    def get_all(self):
        res = []
        for b in range(len(self.memory[-1])):
            tmp = []
            for i in range(self.capacity):
                if self.memory[i] == []:
                    continue
                tmp += self.memory[i][b]
            res.append(tmp)
        return res

    def reset(self):
        self.memory = []
        for i in range(self.capacity):
            self.memory.append([])

    def __len__(self):
        return len(self.memory)


class RLAgent(object):
    def __init__(self, config, word_vocab, verb_map, noun_map, replay_memory_capacity=100000, replay_memory_priority_fraction=0.0, load_pretrained=False):
        # print('Creating RL agent...')
        self.use_dropout_exploration = True  # TODO: move to config.
        self.config = config
        self.use_cuda = config['general']['use_cuda']
        self.word_vocab = word_vocab
        self.verb_map = verb_map
        self.noun_map = noun_map
        self.word2id = {}
        for i, w in enumerate(word_vocab):
            self.word2id[w] = i
        self.model = LSTM_DQN(model_config=config["model"],
                              word_vocab=self.word_vocab,
                              verb_map=verb_map,
                              noun_map=noun_map,
                              enable_cuda=self.use_cuda)
        if load_pretrained:
            self.load_pretrained_model(config["model"]['global']['pretrained_model_save_path'])
        if self.use_cuda:
            self.model.cuda()
        if replay_memory_priority_fraction > 0.0:
            self.replay_memory = PrioritizedReplayMemory(replay_memory_capacity, priority_fraction=replay_memory_priority_fraction)
        else:
            self.replay_memory = ReplayMemory(replay_memory_capacity)
        self.observation_cache_capacity = config['general']['observation_cache_capacity']
        self.observation_cache = ObservationHistoryCache(self.observation_cache_capacity)

    def load_pretrained_model(self, load_from):
        # load model, if there is any
        print("loading best model------------------------------------------------------------------\n")
        try:
            save_f = open(load_from, 'rb')
            self.model = torch.load(save_f)
        except:
            print("failed...lol")

    def reset(self, infos):
        self.rewards = []
        self.dones = []
        self.intermediate_rewards = []
        self.revisit_counting_rewards = []
        self.observation_cache.reset()

    def get_chosen_strings(self, v_idx, n_idx):
        v_idx_np = to_np(v_idx)
        n_idx_np = to_np(n_idx)
        res_str = []
        for i in range(n_idx_np.shape[0]):
            v, n = self.verb_map[v_idx_np[i]], self.noun_map[n_idx_np[i]]
            res_str.append(self.word_vocab[v] + " " + self.word_vocab[n])
        return res_str

    def choose_random_command(self, verb_rank, noun_rank):
        batch_size = verb_rank.size(0)
        vr, nr = to_np(verb_rank), to_np(noun_rank)

        v_idx, n_idx = [], []
        for i in range(batch_size):
            v_idx.append(np.random.choice(len(vr[i]), 1)[0])
            n_idx.append(np.random.choice(len(nr[i]), 1)[0])
        v_qvalue, n_qvalue = [], []
        for i in range(batch_size):
            v_qvalue.append(verb_rank[i][v_idx[i]])
            n_qvalue.append(noun_rank[i][n_idx[i]])
        v_qvalue, n_qvalue = torch.cat(v_qvalue), torch.cat(n_qvalue)
        v_idx, n_idx = to_pt(np.array(v_idx), self.use_cuda), to_pt(np.array(n_idx), self.use_cuda)
        return v_qvalue, v_idx, n_qvalue, n_idx

    def choose_maxQ_command(self, verb_rank, noun_rank):
        batch_size = verb_rank.size(0)
        vr, nr = to_np(verb_rank), to_np(noun_rank)
        v_idx = np.argmax(vr, -1)
        n_idx = np.argmax(nr, -1)
        v_qvalue, n_qvalue = [], []
        for i in range(batch_size):
            v_qvalue.append(verb_rank[i][v_idx[i]])
            n_qvalue.append(noun_rank[i][n_idx[i]])
        v_qvalue, n_qvalue = torch.cat(v_qvalue), torch.cat(n_qvalue)
        v_idx, n_idx = to_pt(v_idx, self.use_cuda), to_pt(n_idx, self.use_cuda)
        return v_qvalue, v_idx, n_qvalue, n_idx

    def get_ranks(self, input_description):
        state_representation = self.model.representation_generator(input_description)
        verb_rank, noun_rank = self.model.action_scorer(state_representation)  # batch x n_verb, batch x n_noun

        return state_representation, verb_rank, noun_rank

    def generate_one_command(self, input_description, epsilon=0.2):
        state_representation, verb_rank, noun_rank = self.get_ranks(input_description)  # batch x n_verb, batch x n_noun
        state_representation = state_representation.detach()

        v_qvalue_maxq, v_idx_maxq, n_qvalue_maxq, n_idx_maxq = self.choose_maxQ_command(verb_rank, noun_rank)
        v_qvalue_random, v_idx_random, n_qvalue_random, n_idx_random = self.choose_random_command(verb_rank, noun_rank)

        # random number for epsilon greedy
        rand_num = np.random.uniform(low=0.0, high=1.0, size=(input_description.size(0),))
        less_than_epsilon = (rand_num < epsilon).astype("float32")  # batch
        greater_than_epsilon = 1.0 - less_than_epsilon
        less_than_epsilon = to_pt(less_than_epsilon, self.use_cuda, type='float')
        greater_than_epsilon = to_pt(greater_than_epsilon, self.use_cuda, type='float')
        less_than_epsilon, greater_than_epsilon = less_than_epsilon.long(), greater_than_epsilon.long()
        v_idx = less_than_epsilon * v_idx_random + greater_than_epsilon * v_idx_maxq
        n_idx = less_than_epsilon * n_idx_random + greater_than_epsilon * n_idx_maxq
        v_idx, n_idx = v_idx.detach(), n_idx.detach()

        chosen_strings = self.get_chosen_strings(v_idx, n_idx)

        return v_idx, n_idx, chosen_strings, state_representation

    def get_game_step_info(self, ob, infos, prev_actions=None):
        # concat d/i/q/f/pf together as one string

        inventory_strings = [info["inventory"] for info in infos]
        inventory_token_list = [preproc(item, str_type='inventory', lower_case=True) for item in inventory_strings]
        inventory_id_list = [_words_to_ids(tokens, self.word2id) for tokens in inventory_token_list]

        feedback_strings = [info["command_feedback"] for info in infos]
        feedback_token_list = [preproc(item, str_type='feedback', lower_case=True) for item in feedback_strings]
        feedback_id_list = [_words_to_ids(tokens, self.word2id) for tokens in feedback_token_list]

        quest_strings = [info["objective"] for info in infos]
        quest_token_list = [preproc(item, str_type='None', lower_case=True) for item in quest_strings]
        quest_id_list = [_words_to_ids(tokens, self.word2id) for tokens in quest_token_list]

        if prev_actions is not None:
            prev_action_token_list = [preproc(item, str_type='None', lower_case=True) for item in prev_actions]
            prev_action_id_list = [_words_to_ids(tokens, self.word2id) for tokens in prev_action_token_list]
        else:
            prev_action_id_list = [[] for _ in infos]

        description_strings = [info["description"] for info in infos]
        description_token_list = [preproc(item, str_type='description', lower_case=True) for item in description_strings]
        for i, d in enumerate(description_token_list):
            if len(d) == 0:
                description_token_list[i] = ["end"]  # hack here, if empty description, insert word "end"
        description_id_list = [_words_to_ids(tokens, self.word2id) for tokens in description_token_list]
        description_id_list = [_d + _i + _q + _f + _pa for (_d, _i, _q, _f, _pa) in zip(description_id_list, inventory_id_list, quest_id_list, feedback_id_list, prev_action_id_list)]

        self.observation_cache.push(description_id_list)
        description_with_history_id_list = self.observation_cache.get_all()

        input_description = pad_sequences(description_with_history_id_list, maxlen=max_len(description_with_history_id_list), padding='post').astype('int32')
        input_description = to_pt(input_description, self.use_cuda)

        return input_description, description_with_history_id_list

    def get_observation_strings(self, infos):
        # concat game_id_d/i/d together as one string
        game_file_names = [info["game_file"] for info in infos]
        inventory_strings = [info["inventory"] for info in infos]
        description_strings = [info["description"] for info in infos]
        observation_strings = [_n + _d + _i for (_n, _d, _i) in zip(game_file_names, description_strings, inventory_strings)]

        return observation_strings

    def compute_reward(self, revisit_counting_lambda=0.0, revisit_counting=True):
        if len(self.dones) == 1:
            mask = [1.0 for _ in self.dones[-1]]
        else:
            assert len(self.dones) > 1
            mask = [1.0 if not self.dones[-2][i] else 0.0 for i in range(len(self.dones[-1]))]
        mask = np.array(mask, dtype='float32')
        mask_pt = to_pt(mask, self.use_cuda, type='float')

        # self.rewards: list of list, max_game_length x batch_size
        rewards = np.array(self.rewards[-1], dtype='float32')  # batch
        if revisit_counting:
            if len(self.revisit_counting_rewards) > 0:
                rewards += np.array(self.revisit_counting_rewards[-1], dtype='float32') * revisit_counting_lambda
        rewards_pt = to_pt(rewards, self.use_cuda, type='float')

        return rewards, rewards_pt, mask, mask_pt

    def update(self, replay_batch_size, discount_gamma=0.0):

        if len(self.replay_memory) < replay_batch_size:
            return None
        transitions = self.replay_memory.sample(replay_batch_size)
        batch = Transition(*zip(*transitions))

        observation_id_list = pad_sequences(batch.observation_id_list, maxlen=max_len(batch.observation_id_list), padding='post').astype('int32')
        input_observation = to_pt(observation_id_list, self.use_cuda)
        next_observation_id_list = pad_sequences(batch.next_observation_id_list, maxlen=max_len(batch.next_observation_id_list), padding='post').astype('int32')
        next_input_observation = to_pt(next_observation_id_list, self.use_cuda)
        v_idx = torch.stack(batch.v_idx, 0)  # batch x 1
        n_idx = torch.stack(batch.n_idx, 0)  # batch x 1

        _, verb_rank, noun_rank = self.get_ranks(input_observation)

        v_qvalue, n_qvalue = verb_rank.gather(1, v_idx).squeeze(-1), noun_rank.gather(1, n_idx).squeeze(-1)  # batch
        q_value = torch.mean(torch.stack([v_qvalue, n_qvalue], -1), -1)  # batch

        _, next_verb_rank, next_noun_rank = self.get_ranks(next_input_observation)  # batch x n_verb, batch x n_noun
        next_v_qvalue, _, next_n_qvalue, _ = self.choose_maxQ_command(next_verb_rank, next_noun_rank)
        next_q_value = torch.mean(torch.stack([next_v_qvalue, next_n_qvalue], -1), -1)  # batch
        next_q_value = next_q_value.detach()

        rewards = torch.cat(batch.reward)  # batch
        not_done = 1.0 - np.array(batch.done, dtype='float32')  # batch
        not_done = to_pt(not_done, self.use_cuda, type='float')
        rewards = rewards + not_done * next_q_value * discount_gamma  # batch
        mask = torch.cat(batch.mask)  # batch
        loss = F.smooth_l1_loss(q_value * mask, rewards * mask)
        return loss

    def finish(self):
        # Game has finished.
        # this function does nothing, bust compute values that to be printed out
        self.final_rewards = np.array(self.rewards[-1], dtype='float32')  # batch
        self.final_counting_rewards = np.sum(np.array(self.revisit_counting_rewards), 0)  # batch
        dones = []
        for d in self.dones:
            d = np.array([float(dd) for dd in d], dtype='float32')
            dones.append(d)
        dones = np.array(dones)
        step_used = 1.0 - dones
        self.step_used_before_done = np.sum(step_used, 0)  # batch

        self.final_intermediate_rewards = []
        intermediate_rewards = np.array(self.intermediate_rewards)  # step x batch
        intermediate_rewards = np.transpose(intermediate_rewards, (1, 0))  # batch x step
        for i in range(intermediate_rewards.shape[0]):
            self.final_intermediate_rewards.append(np.sum(intermediate_rewards[i][:int(self.step_used_before_done[i]) + 1]))
        self.final_intermediate_rewards = np.array(self.final_intermediate_rewards)

    def reset_binarized_counter(self, batch_size):
        self.binarized_counter_dict = [{} for _ in range(batch_size)]

    def get_binarized_count(self, observation_strings, update=True):
        batch_size = len(observation_strings)
        count_rewards = []
        for i in range(batch_size):
            concat_string = observation_strings[i]
            if concat_string not in self.binarized_counter_dict[i]:
                self.binarized_counter_dict[i][concat_string] = 0.0
            if update:
                self.binarized_counter_dict[i][concat_string] += 1.0
            r = self.binarized_counter_dict[i][concat_string]
            r = float(r == 1.0)
            count_rewards.append(r)
        return count_rewards

    def state_dict(self):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
