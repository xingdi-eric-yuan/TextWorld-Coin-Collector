import logging
import os
import numpy as np
import argparse
import warnings
import yaml
from os.path import join as pjoin
import sys
sys.path.append(sys.path[0] + "/..")

import torch
from tensorboardX import SummaryWriter
from agent import RLAgent

from helpers.generic import SlidingAverage, to_np
from helpers.generic import get_experiment_dir, dict2list
from helpers.setup_logger import setup_logging, log_git_commit
from test_agent import test
logger = logging.getLogger(__name__)

import gym
import gym_textworld  # Register all textworld environments.

import textworld


def train(config):
    # train env
    print('Setting up TextWorld environment...')
    batch_size = config['training']['scheduling']['batch_size']
    env_id = gym_textworld.make_batch(env_id=config['general']['env_id'],
                                      batch_size=batch_size,
                                      parallel=True)
    env = gym.make(env_id)
    env.seed(config['general']['random_seed'])

    # valid and test env
    run_test = config['general']['run_test']
    if run_test:
        test_batch_size = config['training']['scheduling']['test_batch_size']
        # valid
        valid_env_name = config['general']['valid_env_id']

        valid_env_id = gym_textworld.make_batch(env_id=valid_env_name, batch_size=test_batch_size, parallel=True)
        valid_env = gym.make(valid_env_id)
        valid_env.seed(config['general']['random_seed'])

        # test
        test_env_name_list = config['general']['test_env_id']
        assert isinstance(test_env_name_list, list)

        test_env_id_list = [gym_textworld.make_batch(env_id=item, batch_size=test_batch_size, parallel=True) for item in test_env_name_list]
        test_env_list = [gym.make(test_env_id) for test_env_id in test_env_id_list]
        for i in range(len(test_env_list)):
            test_env_list[i].seed(config['general']['random_seed'])
    print('Done.')

    # Set the random seed manually for reproducibility.
    np.random.seed(config['general']['random_seed'])
    torch.manual_seed(config['general']['random_seed'])
    if torch.cuda.is_available():
        if not config['general']['use_cuda']:
            logger.warning("WARNING: CUDA device detected but 'use_cuda: false' found in config.yaml")
        else:
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed(config['general']['random_seed'])
    else:
        config['general']['use_cuda'] = False  # Disable CUDA.
    revisit_counting = config['general']['revisit_counting']
    replay_batch_size = config['general']['replay_batch_size']
    history_size = config['general']['history_size']
    update_from = config['general']['update_from']
    replay_memory_capacity = config['general']['replay_memory_capacity']
    replay_memory_priority_fraction = config['general']['replay_memory_priority_fraction']

    word_vocab = dict2list(env.observation_space.id2w)
    word2id = {}
    for i, w in enumerate(word_vocab):
        word2id[w] = i

    # collect all nouns
    # verb_list, object_name_list = get_verb_and_object_name_lists(env)
    verb_list = ["go", "take"]
    object_name_list = ["east", "west", "north", "south", "coin"]
    verb_map = [word2id[w] for w in verb_list if w in word2id]
    noun_map = [word2id[w] for w in object_name_list if w in word2id]
    agent = RLAgent(config, word_vocab, verb_map, noun_map,
                    replay_memory_capacity=replay_memory_capacity, replay_memory_priority_fraction=replay_memory_priority_fraction)

    init_learning_rate = config['training']['optimizer']['learning_rate']
    exp_dir = get_experiment_dir(config)
    summary = SummaryWriter(exp_dir)

    parameters = filter(lambda p: p.requires_grad, agent.model.parameters())
    if config['training']['optimizer']['step_rule'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=init_learning_rate)
    elif config['training']['optimizer']['step_rule'] == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=init_learning_rate)

    log_every = 100
    reward_avg = SlidingAverage('reward avg', steps=log_every)
    step_avg = SlidingAverage('step avg', steps=log_every)
    loss_avg = SlidingAverage('loss avg', steps=log_every)

    # save & reload checkpoint only in 0th agent
    best_avg_reward = -10000
    best_avg_step = 10000

    # step penalty
    discount_gamma = config['general']['discount_gamma']
    provide_prev_action = config['general']['provide_prev_action']

    # epsilon greedy
    epsilon_anneal_epochs = config['general']['epsilon_anneal_epochs']
    epsilon_anneal_from = config['general']['epsilon_anneal_from']
    epsilon_anneal_to = config['general']['epsilon_anneal_to']

    # counting reward
    revisit_counting_lambda_anneal_epochs = config['general']['revisit_counting_lambda_anneal_epochs']
    revisit_counting_lambda_anneal_from = config['general']['revisit_counting_lambda_anneal_from']
    revisit_counting_lambda_anneal_to = config['general']['revisit_counting_lambda_anneal_to']

    epsilon = epsilon_anneal_from
    revisit_counting_lambda = revisit_counting_lambda_anneal_from
    if revisit_counting == "cumulative":
        agent.reset_cumulative_counter()
    for epoch in range(config['training']['scheduling']['epoch']):

        agent.model.train()
        obs, infos = env.reset()
        agent.reset(infos)
        print_command_string, print_rewards = [[] for _ in infos], [[] for _ in infos]
        print_interm_rewards = [[] for _ in infos]
        print_rc_rewards = [[] for _ in infos]

        dones = [False] * batch_size
        rewards = None
        avg_loss_in_this_game = []

        curr_observation_strings = agent.get_observation_strings(infos)
        if revisit_counting == "episodic":
            agent.reset_binarized_counter(batch_size)
            revisit_counting_rewards = agent.get_binarized_count(curr_observation_strings)
        elif revisit_counting == "cumulative":
            # do not reset counter here
            revisit_counting_rewards = agent.get_cumulative_count(curr_observation_strings)
        else:
            # revisit_counting is None
            pass

        current_game_step = 0
        prev_actions = ["" for _ in range(batch_size)] if provide_prev_action else None
        input_description, description_id_list = agent.get_game_step_info(obs, infos, prev_actions)
        curr_ras_hidden, curr_ras_cell = None, None  # ras: recurrent action scorer
        memory_cache = [[] for _ in range(batch_size)]
        solved = [0 for _ in range(batch_size)]

        while not all(dones):
            agent.model.train()
            v_idx, n_idx, chosen_strings, curr_ras_hidden, curr_ras_cell = agent.generate_one_command(input_description, curr_ras_hidden, curr_ras_cell, epsilon=epsilon)
            obs, rewards, dones, infos = env.step(chosen_strings)
            curr_observation_strings = agent.get_observation_strings(infos)
            if provide_prev_action:
                prev_actions = chosen_strings
            # counting
            if revisit_counting == "episodic":
                revisit_counting_rewards = agent.get_binarized_count(curr_observation_strings, update=True)
            elif revisit_counting == "cumulative":
                revisit_counting_rewards = agent.get_cumulative_count(curr_observation_strings, update=True)
            else:
                revisit_counting_rewards = [0.0 for b in range(batch_size)]
            agent.revisit_counting_rewards.append(revisit_counting_rewards)
            revisit_counting_rewards = [float(format(item, ".3f")) for item in revisit_counting_rewards]

            for i in range(len(infos)):
                print_command_string[i].append(chosen_strings[i])
                print_rewards[i].append(rewards[i])
                print_interm_rewards[i].append(infos[i]["intermediate_reward"])
                print_rc_rewards[i].append(revisit_counting_rewards[i])
            if type(dones) is bool:
                dones = [dones] * batch_size
            agent.rewards.append(rewards)
            agent.dones.append(dones)
            agent.intermediate_rewards.append([info["intermediate_reward"] for info in infos])
            # computer rewards, and push into replay memory
            rewards_np, rewards_pt, mask_np, mask_pt, memory_mask = agent.compute_reward(revisit_counting_lambda=revisit_counting_lambda, revisit_counting=revisit_counting)

            curr_description_id_list = description_id_list
            input_description, description_id_list = agent.get_game_step_info(obs, infos, prev_actions)

            for b in range(batch_size):
                if memory_mask[b] == 0:
                    continue
                if dones[b] == 1 and rewards[b] == 0:
                    # last possible step
                    is_final = True
                else:
                    is_final = mask_np[b] == 0
                if rewards[b] > 0.0:
                    solved[b] = 1
                # replay memory
                memory_cache[b].append((curr_description_id_list[b], v_idx[b], n_idx[b], rewards_pt[b], mask_pt[b], dones[b], is_final, curr_observation_strings[b]))

            if current_game_step > 0 and current_game_step % config["general"]["update_per_k_game_steps"] == 0:
                policy_loss = agent.update(replay_batch_size, history_size, update_from, discount_gamma=discount_gamma)
                if policy_loss is None:
                    continue
                loss = policy_loss
                # Backpropagate
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), config['training']['optimizer']['clip_grad_norm'])
                optimizer.step()  # apply gradients
                avg_loss_in_this_game.append(to_np(policy_loss))
            current_game_step += 1

        for i, mc in enumerate(memory_cache):
            for item in mc:
                if replay_memory_priority_fraction == 0.0:
                    # vanilla replay memory
                    agent.replay_memory.push(*item)
                else:
                    # prioritized replay memory
                    agent.replay_memory.push(solved[i], *item)

        agent.finish()
        avg_loss_in_this_game = np.mean(avg_loss_in_this_game)
        reward_avg.add(agent.final_rewards.mean())
        step_avg.add(agent.step_used_before_done.mean())
        loss_avg.add(avg_loss_in_this_game)
        # annealing
        if epoch < epsilon_anneal_epochs:
            epsilon -= (epsilon_anneal_from - epsilon_anneal_to) / float(epsilon_anneal_epochs)
        if epoch < revisit_counting_lambda_anneal_epochs:
            revisit_counting_lambda -= (revisit_counting_lambda_anneal_from - revisit_counting_lambda_anneal_to) / float(revisit_counting_lambda_anneal_epochs)

        # Tensorboard logging #
        # (1) Log some numbers
        if (epoch + 1) % config["training"]["scheduling"]["logging_frequency"] == 0:
            summary.add_scalar('avg_reward', reward_avg.value, epoch + 1)
            summary.add_scalar('curr_reward', agent.final_rewards.mean(), epoch + 1)
            summary.add_scalar('curr_interm_reward', agent.final_intermediate_rewards.mean(), epoch + 1)
            summary.add_scalar('curr_counting_reward', agent.final_counting_rewards.mean(), epoch + 1)
            summary.add_scalar('avg_step', step_avg.value, epoch + 1)
            summary.add_scalar('curr_step', agent.step_used_before_done.mean(), epoch + 1)
            summary.add_scalar('loss_avg', loss_avg.value, epoch + 1)
            summary.add_scalar('curr_loss', avg_loss_in_this_game, epoch + 1)

        msg = 'E#{:03d}, R={:.3f}/{:.3f}/IR{:.3f}/CR{:.3f}, S={:.3f}/{:.3f}, L={:.3f}/{:.3f}, epsilon={:.4f}, lambda_counting={:.4f}'
        msg = msg.format(epoch,
                         np.mean(reward_avg.value), agent.final_rewards.mean(), agent.final_intermediate_rewards.mean(), agent.final_counting_rewards.mean(),
                         np.mean(step_avg.value), agent.step_used_before_done.mean(),
                         np.mean(loss_avg.value), avg_loss_in_this_game,
                         epsilon, revisit_counting_lambda)
        if (epoch + 1) % config["training"]["scheduling"]["logging_frequency"] == 0:
            print("=========================================================")
            for prt_cmd, prt_rew, prt_int_rew, prt_rc_rew in zip(print_command_string, print_rewards, print_interm_rewards, print_rc_rewards):
                print("------------------------------")
                print(prt_cmd)
                print(prt_rew)
                print(prt_int_rew)
                print(prt_rc_rew)
        print(msg)
        # test on a different set of games
        if run_test and (epoch + 1) % config["training"]["scheduling"]["logging_frequency"] == 0:
            valid_R, valid_IR, valid_S = test(config, valid_env, agent, test_batch_size, word2id)
            summary.add_scalar('valid_reward', valid_R, epoch + 1)
            summary.add_scalar('valid_interm_reward', valid_IR, epoch + 1)
            summary.add_scalar('valid_step', valid_S, epoch + 1)

            # save & reload checkpoint by best valid performance
            model_checkpoint_path = config['training']['scheduling']['model_checkpoint_path']
            if valid_R > best_avg_reward or (valid_R == best_avg_reward and valid_S < best_avg_step):
                best_avg_reward = valid_R
                best_avg_step = valid_S
                torch.save(agent.model.state_dict(), model_checkpoint_path)
                print("========= saved checkpoint =========")
                for test_id in range(len(test_env_list)):
                    R, IR, S = test(config, test_env_list[test_id], agent, test_batch_size, word2id)
                    summary.add_scalar('test_reward_' + str(test_id), R, epoch + 1)
                    summary.add_scalar('test_interm_reward_' + str(test_id), IR, epoch + 1)
                    summary.add_scalar('test_step_' + str(test_id), S, epoch + 1)


if __name__ == '__main__':
    for _p in ['saved_models']:
        if not os.path.exists(_p):
            os.mkdir(_p)
    parser = argparse.ArgumentParser(description="train network.")
    parser.add_argument("-c", "--config_dir", default='config', help="the default config directory")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-vv", "--very-verbose", help="print out warnings", action="store_true")
    args = parser.parse_args()

    if args.very_verbose:
        args.verbose = args.very_verbose
        warnings.simplefilter("default", textworld.TextworldGenerationWarning)

    # Read config from yaml file.
    config_file = pjoin(args.config_dir, 'config.yaml')
    with open(config_file) as reader:
        config = yaml.safe_load(reader)

    default_logs_path = get_experiment_dir(config)
    setup_logging(default_config_path=pjoin(args.config_dir, 'logging_config.yaml'),
                  default_level=logging.INFO, add_time_stamp=True,
                  default_logs_path=default_logs_path)
    log_git_commit(logger)

    train(config=config)
