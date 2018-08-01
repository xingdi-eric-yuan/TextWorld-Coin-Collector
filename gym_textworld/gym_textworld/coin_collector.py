"""
Register all environments related to the Coin Collector benchmark.
"""
from gym.envs.registration import register


MAX_LEVEL = 40
for level in list(range(1, MAX_LEVEL + 1)) + list(range(101, 100 + MAX_LEVEL + 1)) + list(range(201, 200 + MAX_LEVEL + 1)):
    for _n_games in [1, 2, 3, 5, 10, 30, 50, 100, 500, 1000, 5000]:
        for _game_id in range(10):
            for max_steps in [30, 50, 100, 200]:
                for theme in ["house", "basic"]:
                    register(
                        id='tw-coin-collector_level{}_gamesize{}_step{}_seed{}_{}-v0'.format(level, _n_games, max_steps, _game_id, theme),
                        entry_point='gym_textworld.envs:CoinCollectorLevel',
                        max_episode_steps=max_steps,
                        kwargs={
                            'n_games': _n_games,
                            'level': level,
                            'game_generator_seed': 20180514 + level * 1000 + _n_games * 100 + _game_id * 10 + max_steps,
                            'request_infos': [
                                "objective",
                                "description",
                                "inventory",
                                "command_feedback",
                                "intermediate_reward",
                                "admissible_commands"
                            ],
                            'grammar_flags': {
                                "theme": theme,
                                "only_last_action": True,
                            }
                        }
                    )

                    # Validation
                    register(
                        id='tw-coin-collector_level{}_gamesize{}_step{}_seed{}_{}-validation-v0'.format(level, _n_games, max_steps, _game_id, theme),
                        entry_point='gym_textworld.envs:CoinCollectorLevel',
                        max_episode_steps=max_steps,
                        kwargs={
                            'n_games': _n_games,
                            'level': level,
                            'game_generator_seed': 81020619 + level * 1000 + _n_games * 100 + _game_id * 10 + max_steps,
                            'request_infos': [
                                "objective",
                                "description",
                                "inventory",
                                "command_feedback",
                                "intermediate_reward",
                                "admissible_commands"
                            ],
                            'grammar_flags': {
                                "theme": theme,
                                "only_last_action": True,
                            }
                        }
                    )

                    # Test
                    register(
                        id='tw-coin-collector_level{}_gamesize{}_step{}_seed{}_{}-test-v0'.format(level, _n_games, max_steps, _game_id, theme),
                        entry_point='gym_textworld.envs:CoinCollectorLevel',
                        max_episode_steps=max_steps,
                        kwargs={
                            'n_games': _n_games,
                            'level': level,
                            'game_generator_seed': 41508102 + level * 1000 + _n_games * 100 + _game_id * 10 + max_steps,
                            'request_infos': [
                                "objective",
                                "description",
                                "inventory",
                                "command_feedback",
                                "intermediate_reward",
                                "admissible_commands"
                            ],
                            'grammar_flags': {
                                "theme": theme,
                                "only_last_action": True,
                            }
                        }
                    )

