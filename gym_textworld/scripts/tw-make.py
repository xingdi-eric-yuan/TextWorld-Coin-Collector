#!/usr/bin/env python
import argparse
import warnings
import multiprocessing

from tqdm import tqdm

import textworld

import gym
import gym_textworld  # Register all textworld environments.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id",
                        help="Gym-Textworld Environment for which to generate"
                             " the games. Should follow this pattern"
                             " twcc_[easy|medium|hard]_level[int]_gamesize[int]_step[int]_seed[int]_[train|validation|test]")
    parser.add_argument("--env_seed", type=int,
                        help="Random seed for generating the games.")
    parser.add_argument("--nb-processes", type=int,
                        help="Number of games to generate in parallel."
                             " Default: as many as there are CPU cores.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Activate verbose mode.")
    parser.add_argument("-vv", "--very-verbose", action="store_true",
                        help="Verbose mode + print warning messages.")
    return parser.parse_args()


def _generate_game(env_id, skip):
    env = gym.make(env_id)
    env.unwrapped.skip(skip)
    env.unwrapped._next_game()
    env.close()


def main():
    args = parse_args()

    if args.very_verbose:
        args.verbose = args.very_verbose
        warnings.simplefilter("default", textworld.TextworldGenerationWarning)

    if args.nb_processes is None:
        args.nb_processes = multiprocessing.cpu_count()

    env_id = gym_textworld.make(args.env_id)
    env = gym.make(env_id)
    env.seed(args.env_seed)
    nb_games = env.unwrapped.n_games
    skip_list = range(nb_games)

    desc = "Generating games for {}".format(args.env_id)
    pbar = tqdm(total=nb_games, desc=desc)

    print("Using {} processes.".format(args.nb_processes))
    if args.nb_processes > 1:
        pool = multiprocessing.Pool(args.nb_processes)
        for skip in skip_list:
            pool.apply_async(_generate_game, (env_id, skip), callback=lambda _: pbar.update())

        pool.close()
        pool.join()
        pbar.close()

    else:
        for skip in skip_list:
            _generate_game(env_id, skip)
            pbar.update()


if __name__ == "__main__":
    main()
