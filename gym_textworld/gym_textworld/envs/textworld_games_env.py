import gym

import textworld

from gym_textworld.spaces import text_spaces


class TextworldGameEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, gamefile, ob_max_length, act_max_length, vocab=None, mode="word"):
        self.gamefile = gamefile
        self.game_env = textworld.play(gamefile)
        self.action_space = text_spaces.Char(max_length=act_max_length)
        self.observation_space = text_spaces.Char(max_length=ob_max_length,
                                                  extra_vocab=[".", ",", "\n"])
        # self.action_space = text_spaces.Word(max_length=8, vocab=vocab)
        # self.observation_space = text_spaces.Word(max_length=200, vocab=vocab)

    def step(self, action):
        action = [self.action_space.id2c[i] for i in action]
        text_command = "".join(action)  # Text command
        # text_command = action

        game_state, reward, done = self.game_env.step(text_command)

        observation = game_state.feedback
        # observation = self.observation_space.tokenize(observation)
        infos = {"game_state": game_state}
        return observation, reward, done, infos

    def seed(self, seed=None):
        self.game_env.seed(seed)
        return [seed]

    def reset(self):
        game_state = self.game_env.reset()
        observation = game_state.feedback
        # observation = self.observation_space.tokenize(observation)
        return observation

    def render(self, mode='human'):
        self.game_env.render(mode=mode)

    def close(self):
        if self.game_env is not None:
            self.game_env.close()

        self.game_env = None
