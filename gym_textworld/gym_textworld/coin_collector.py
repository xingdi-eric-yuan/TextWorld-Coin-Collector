"""
Register all environments related to the Coin Collector benchmark.
"""
import re

from gym.envs.registration import register


PATTERN = re.compile(r"twcc_(easy|medium|hard)_level(\d+)_gamesize(\d+)_step(\d+)_seed(\d+)_(train|validation|test)")
SEED_OFFSETS = {'train': 20180514, 'validation': 81020619, 'test': 41508102}
MODE2LEVEL = {"easy": 0, "medium": 100, "hard": 200}


def make(env_id):
    match = re.match(PATTERN, env_id)
    if not match:
        msg = "env_id should match the following pattern:\n{}"
        raise ValueError(msg.format(PATTERN.pattern))

    mode = match.group(1)
    level = int(match.group(2))
    n_games = int(match.group(3))
    max_steps = int(match.group(4))
    random_seed = int(match.group(5))
    split = match.group(6)

    game_generator_seed = SEED_OFFSETS[split] + MODE2LEVEL[mode] * 10000 + level * 1000 + n_games * 100 + random_seed * 10 + max_steps
    env_id = env_id + "-v0"
    register(
        id=env_id,
        entry_point='gym_textworld.envs:CoinCollectorLevel',
        max_episode_steps=max_steps,
        kwargs={
            'n_games': n_games,
            'level': MODE2LEVEL[mode] + level,
            'game_generator_seed': game_generator_seed,
            'request_infos': [
                "objective",
                "description",
                "inventory",
                "command_feedback",
                "intermediate_reward",
                "admissible_commands"
            ],
            'grammar_flags': {
                "theme": "house",
                "only_last_action": True,
            }
        }
    )
    return env_id
