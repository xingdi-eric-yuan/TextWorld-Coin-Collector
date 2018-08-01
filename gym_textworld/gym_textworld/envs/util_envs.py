import numpy as np

import multiprocessing as mp

import gym


def _child(id, pipe):
    """
    Event loop run by the child processes
    """
    env = gym.make(id)

    try:
        while True:
            command = pipe.recv()
            # command is a tuple like ("call" | "get", "name.of.attr", extra args...)

            obj = env
            attrs = command[1].split(".")
            for attr in attrs[:-1]:
                obj = getattr(obj, attr)

            if command[0] == "call":
                fct = getattr(obj, attrs[-1])
                result = fct(*command[2])
            elif command[0] == "get":
                result = getattr(obj, attrs[-1])
            elif command[0] == "hasattr":
                result = hasattr(obj, attrs[-1])

            pipe.send(result)
    finally:
        pipe.close()


class _ChildEnv:
    """
    Wrapper for an env in a child process.
    """
    def __init__(self, id):
        self._pipe, child_pipe = mp.Pipe()
        self._process = mp.Process(target=_child, args=(id, child_pipe))
        self._process.start()

    def call(self, method, *args):
        self._pipe.send(("call", method, args))

    def get(self, attr):
        self._pipe.send(("get", attr))

    def hasattr(self, attr):
        self._pipe.send(("hasattr", attr))

    def result(self):
        return self._pipe.recv()

    def call_sync(self, *args):
        self.call(*args)
        return self.result()

    def get_sync(self, *args):
        self.get(*args)
        return self.result()

    def hasattr_sync(self, *args):
        self.hasattr(*args)
        return self.result()

    def close(self):
        self._pipe.close()
        self._process.join()


class ParallelBatchEnv(gym.Env):
    """ Environment to run multiple games in parallel.
    """
    def __init__(self, env_id, batch_size):
        """
        Parameters
        ----------
        env_id : list of str or str
            Environment IDs that will compose a batch. If only
            one env_id is provided, it will be repeated `batch_size` times.
        batch_size : int
            Number of environment to run in parallel.
        """
        self.env_ids = env_id if type(env_id) is list else [env_id] * batch_size
        self.batch_size = batch_size
        assert len(self.env_ids) == self.batch_size

        self.envs = []
        for id in self.env_ids:
            self.envs.append(_ChildEnv(id))

        self.observation_space = self.envs[0].get_sync("observation_space")
        self.action_space = self.envs[0].get_sync("action_space")

    def skip(self, ngames=1):
        for env in self.envs:
            env.call_sync("unwrapped.skip", ngames)

    def seed(self, seed=None):
        # Use different seed for each env to decorrelate
        # the examples in the batch.
        rng = np.random.RandomState(seed)
        seeds = list(rng.randint(65635, size=self.batch_size))
        for env, seed in zip(self.envs, seeds):
            env.call_sync("seed", seed)

        return seeds

    def reset(self):
        self.last = [None] * self.batch_size
        obs, infos = [], []
        for env in self.envs:
            env.call("reset")

        for env in self.envs:
            result = env.result()
            ob, info = result
            obs.append(ob)
            infos.append(info)

        return obs, infos

    def step(self, actions):
        results = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if self.last[i] is not None and self.last[i][2]:  # Game is done
                results.append(self.last[i])  # Copy last infos over.
            else:
                env.call("step", action)
                results.append(None)

        obs, rewards, dones, infos = [], [], [], []
        for i, (env, result) in enumerate(zip(self.envs, results)):
            if result is None:
                result = env.result()

            ob, reward, done, info = result

            obs.append(ob)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

            self.last[i] = result

        return obs, rewards, dones, infos

    def render(self, mode='human'):
        for env in self.envs:
            env.call("render", mode)

        renderings = []
        for env in self.envs:
            renderings.append(env.result())

        return renderings

    def close(self):
        for env in self.envs:
            env.close()


class BatchEnv(gym.Env):
    """ Environment to run multiple games independently.
    """
    def __init__(self, env_id, batch_size):
        """
        Parameters
        ----------
        env_id : list of str or str
            Environment IDs that will compose a batch. If only
            one env_id is provided, it will be repeated `batch_size` times.
        batch_size : int
            Number of independent environments to run.
        """
        self.env_ids = env_id if type(env_id) is list else [env_id] * batch_size
        self.batch_size = batch_size
        assert len(self.env_ids) == self.batch_size

        self.envs = [gym.make(self.env_ids[i]) for i in range(self.batch_size)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def skip(self, ngames=1):
        for env in self.envs:
            env.env.skip(ngames)

    def seed(self, seed=None):
        # Use different seed for each env to decorrelate
        # the examples in the batch.
        rng = np.random.RandomState(seed)
        seeds = list(rng.randint(65635, size=self.batch_size))
        for env, seed in zip(self.envs, seeds):
            env.seed(seed)

        return seeds

    def reset(self):
        self.last = [None] * self.batch_size
        obs, infos = [], []
        for env in self.envs:
            ob, info = env.reset()
            obs.append(ob)
            infos.append(info)

        return obs, infos

    def step(self, actions):
        obs, rewards, dones, infos = [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if self.last[i] is not None and self.last[i][2]:  # Game is done
                ob, reward, done, info = self.last[i]  # Copy last infos over.
            else:
                ob, reward, done, info = env.step(action)

            obs.append(ob)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            self.last[i] = ob, reward, done, info

        return obs, rewards, dones, infos

    def render(self, mode='human'):
        renderings = []
        for env in self.envs:
            rendering = env.render(mode=mode)
            renderings.append(rendering)

        return renderings

    def close(self):
        for env in self.envs:
            env.close()
