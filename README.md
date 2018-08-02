# TextWorld-Coin-Collector
--------------------------------------------------------------------------------
PyTorch implementation of papar [Counting to Explore and Generalize in Text-based Games][counting]

## Coin Collector
* <p align=center><img width="70%" src="hard_level10.png" /></p>
* Coin collector is a set of games, each game is a randomly connected chain of rooms, the agent's goal is to navigate through the path and pick up the coin.
* Modes: easy / medium / hard, amount of off-chain rooms.
* Levels: Length of optimal trajectory.
* Action space: `{go, take} × {north, south, east, west, coin}​`
* Environment ID: `twcc_[mode]_level[level]_gamesize[#game]_step[max step]_seed[random seed]_[split]`, please check [here][coin_collector] for more details.

## Requirements
* Python 3
* [PyTorch][pytorch_install].
* [TextWorld][textworld_install]

## Game Generation
* Install gym_textworld by `pip install gym_textworld`
* run `tw-make.py <env_id>` to generate games corresponding to games defined in config files.

## To Run
* LSTM-DQN: run `python lstm_dqn_baseline/train_single_generate_agent.py -c lstm_dqn_baseline/config/`.
* LSTM-DRQN: run `python lstm_drqn_baseline/train_single_generate_agent.py -c lstm_dqn_baseline/config/`.
* Configurations can be modified in the above two config files.

## LICENSE
[MIT][MIT]

## References
* [TextWorld: A Learning Environment for Text-based Games][textworld_paper]
* [Counting to Explore and Generalize in Text-based Games][counting]

[pytorch_install]: http://pytorch.org/
[textworld_install]: https://github.com/Microsoft/TextWorld/
[counting]: https://arxiv.org/abs/1806.11525/
[textworld_paper]: https://arxiv.org/abs/1806.11532/
[coin_collector]: https://github.com/xingdi-eric-yuan/TextWorld-Coin-Collector/blob/master/gym_textworld/gym_textworld/coin_collector.py/
[MIT]: https://github.com/xingdi-eric-yuan/TextWorld-Coin-Collector/blob/master/LICENSE/