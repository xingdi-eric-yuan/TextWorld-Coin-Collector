#!/usr/bin/env python
import argparse

import textworld
from textworld.generator import Game


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("games", metavar="game", nargs="+",
                        help="TextWorld generated games.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Activate verbose mode.")
    return parser.parse_args()


def main():
    args = parse_args()

    seen = {}
    seen_solution = {}
    for game_path in args.games:
        game_path = game_path.replace(".ulx", ".json")
        game = Game.load(game_path)
        solution = tuple(game.quests[0].commands)

        if game in seen:
            print("Duplicate found:")
            print("  > {}".format(game_path))
            print("  > {}".format(seen[game]))
            print("-----")
            continue

        seen[game] = game_path

        if solution in seen_solution:
            print("Duplicate *solution* found:")
            print("  > {}".format(game_path))
            print("  > {}".format(seen_solution[solution]))
            print("-----")
            continue

        seen_solution[solution] = game_path


if __name__ == "__main__":
    main()
