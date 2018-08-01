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
from agent import RLAgent

from helpers.generic import get_experiment_dir

from helpers.setup_logger import setup_logging, log_git_commit
logger = logging.getLogger(__name__)

import gym
import gym_textworld  # Register all textworld environments.

import textworld


def test(config, env, agent, batch_size, word2id):

    agent.model.eval()

    obs, infos = env.reset()
    agent.reset(infos)
    print_command_string, print_rewards = [[] for _ in infos], [[] for _ in infos]
    print_interm_rewards = [[] for _ in infos]

    provide_prev_action = config['general']['provide_prev_action']

    dones = [False] * batch_size
    rewards = None
    prev_actions = ["" for _ in range(batch_size)] if provide_prev_action else None
    input_description, _ = agent.get_game_step_info(obs, infos, prev_actions)

    while not all(dones):

        v_idx, n_idx, chosen_strings, state_representation = agent.generate_one_command(input_description, epsilon=0.0)
        obs, rewards, dones, infos = env.step(chosen_strings)
        if provide_prev_action:
            prev_actions = chosen_strings

        for i in range(len(infos)):
            print_command_string[i].append(chosen_strings[i])
            print_rewards[i].append(rewards[i])
            print_interm_rewards[i].append(infos[i]["intermediate_reward"])
        if type(dones) is bool:
            dones = [dones] * batch_size
        agent.rewards.append(rewards)
        agent.dones.append(dones)
        agent.intermediate_rewards.append([info["intermediate_reward"] for info in infos])

        input_description, _ = agent.get_game_step_info(obs, infos, prev_actions)

    agent.finish()
    R = agent.final_rewards.mean()
    S = agent.step_used_before_done.mean()
    IR = agent.final_intermediate_rewards.mean()

    msg = '====EVAL==== R={:.3f}, IR={:.3f}, S={:.3f}'
    msg = msg.format(R, IR, S)
    print(msg)
    print("\n")
    return R, IR, S


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

    r, s = test(config=config)
    print("$$$", r, s)
