from gym.envs.registration import register, spec


def make_infinite_shuffled_iterator(iterable, rng):
    """
    Yield each element of `iterable` one by one, then shuffle the elements
    and start yielding from the start.
    """
    elements = []
    for e in iterable:
        elements.append(e)
        yield e

    while True:
        rng.shuffle(elements)
        for e in elements:
            yield e


def make_batch(env_id, batch_size, parallel=False):
    """ Make an environment that runs multiple games independently.

    Parameters
    ----------
    env_id : str
        Environment ID that will compose a batch.
    batch_size : int
        Number of independent environments to run.
    parallel : {True, False}, optional
        If True, the environment will be executed in different processes.
    """
    batch_env_id = "batch{}-".format(batch_size) + env_id
    env_spec = spec(env_id)
    entry_point = 'gym_textworld.envs:BatchEnv'
    if parallel:
        entry_point = 'gym_textworld.envs:ParallelBatchEnv'

    register(
        id=batch_env_id,
        entry_point=entry_point,
        max_episode_steps=env_spec.max_episode_steps,
        max_episode_seconds=env_spec.max_episode_seconds,
        nondeterministic=env_spec.nondeterministic,
        reward_threshold=env_spec.reward_threshold,
        trials=env_spec.trials,
        # Setting the 'vnc' tag avoid wrapping the env with a TimeLimit wrapper. See
        # https://github.com/openai/gym/blob/4c460ba6c8959dd8e0a03b13a1ca817da6d4074f/gym/envs/registration.py#L122
        tags={"vnc": "foo"},
        kwargs={'env_id': env_id, 'batch_size': batch_size}
    )

    return batch_env_id