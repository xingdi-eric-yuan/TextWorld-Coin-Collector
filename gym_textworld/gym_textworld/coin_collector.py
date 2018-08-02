"""
Register all environments related to the Coin Collector benchmark.
"""
from gym.envs.registration import register
mode_to_level = {"easy": 0, "medium": 100, "hard": 200}


MAX_LEVEL = 40
for mode in ["easy", "medium", "hard"]:
    for level in list(range(1, MAX_LEVEL + 1)):
        for n_games in [1, 2, 3, 5, 10, 30, 50, 100, 500, 1000]:
            for random_seed in range(10):
                for max_steps in [50, 200]:
                    register(
                        id='twcc_{}_level{}_gamesize{}_step{}_seed{}_train'.format(mode, level, n_games, max_steps, random_seed),
                        entry_point='gym_textworld.envs:CoinCollectorLevel',
                        max_episode_steps=max_steps,
                        kwargs={
                            'n_games': n_games,
                            'level': mode_to_level[mode] + level,
                            'game_generator_seed': 20180514 + mode_to_level[mode] * 10000 + level * 1000 + n_games * 100 + random_seed * 10 + max_steps,
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

                    # Validation
                    register(
                        id='twcc_{}_level{}_gamesize{}_step{}_seed{}_validation'.format(mode, level, n_games, max_steps, random_seed),
                        entry_point='gym_textworld.envs:CoinCollectorLevel',
                        max_episode_steps=max_steps,
                        kwargs={
                            'n_games': n_games,
                            'level': mode_to_level[mode] + level,
                            'game_generator_seed': 81020619 + mode_to_level[mode] * 10000 + level * 1000 + n_games * 100 + random_seed * 10 + max_steps,
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

                    # Test
                    register(
                        id='twcc_{}_level{}_gamesize{}_step{}_seed{}_test'.format(mode, level, n_games, max_steps, random_seed),
                        entry_point='gym_textworld.envs:CoinCollectorLevel',
                        max_episode_steps=max_steps,
                        kwargs={
                            'n_games': n_games,
                            'level': mode_to_level[mode] + level,
                            'game_generator_seed': 41508102 + mode_to_level[mode] * 10000 + level * 1000 + n_games * 100 + random_seed * 10 + max_steps,
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
