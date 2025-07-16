import numpy as np
import gymnasium as gym
import pytest

from WindGym.wrappers import RecordEpisodeVals, AdversaryWrapper


# --------------------------------------------------------------------------- #
#  Minimal deterministic environment                                        #
# --------------------------------------------------------------------------- #
class PowerEnv(gym.Env):
    """
    *Very* small env that terminates after ``ep_len`` steps and returns
    constant power values.

    Only the keys used by ``RecordEpisodeVals`` are produced.

    Parameters
    ----------
    ep_len : int
        Number of steps before ``terminated=True``.
    power_agent : float
        Per-step "Power agent" value.
    power_base : float | None
        Per-step "Power baseline" value.  ``None`` disables that key.
    """

    def __init__(
        self, ep_len: int, power_agent: float, power_base: float | None = None
    ):
        super().__init__()
        self._len = ep_len
        self._pa = float(power_agent)
        self._pb = None if power_base is None else float(power_base)

        self.action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(1,), dtype=np.float32
        )
        self.t = 0

    # ---------- gym API ---------------------------------------------------- #
    def reset(self, *, seed=None, options=None):
        self.t = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        terminated = self.t >= self._len
        info = {"Power agent": self._pa}
        if self._pb is not None:
            info["Power baseline"] = self._pb
        return (np.zeros(1, dtype=np.float32), 0.0, terminated, False, info)


def vec_env(env_fns):
    """Helper: create a synchronous vector-env from callables."""
    return gym.vector.SyncVectorEnv(env_fns)


# --------------------------------------------------------------------------- #
#  Utility – run until *every* sub-env finishes one episode                 #
# --------------------------------------------------------------------------- #
def roll_episode(wrapped: RecordEpisodeVals, action_val=0.0) -> int:
    """
    Step *wrapped* until **all** contained envs have produced exactly one
    terminated/truncated flag.  Returns the number of vector-steps.
    """
    act = np.full((wrapped.num_envs, 1), action_val, dtype=np.float32)
    done = np.zeros(wrapped.num_envs, dtype=bool)
    steps = 0
    while not done.all():
        _, _, term, trunc, _ = wrapped.step(act)
        done |= np.logical_or(term, trunc)
        steps += 1
    return steps


# --------------------------------------------------------------------------- #
#  Tests                                                                    #
# --------------------------------------------------------------------------- #
def test_single_env_mean_power_queue() -> None:
    """
    One env, no baseline:

        mean_power = Σ(power_agent) / episode_len

    must be appended after the episode terminates.
    """
    env = vec_env([lambda: PowerEnv(ep_len=4, power_agent=10.0)])
    w = RecordEpisodeVals(env)

    w.reset()
    roll_episode(w)

    assert list(w.mean_power_queue) == [10.0]  # 4×10 / 4
    assert len(w.mean_power_queue_baseline) == 0


def test_multi_env_staggered_done() -> None:
    """
    Two envs with *different* episode lengths (2 & 3 steps).

    Expectations
    ------------
    * The first mean (5) is queued when env-0 ends,
      the second mean (2) when env-1 ends.
    * ``episode_powers`` are per-episode accumulators and *are* cleared by
      episode termination or ``reset()``.
    * The rolling mean queue is **not** cleared by ``reset()`` – it is meant
      to persist across many episodes for moving-average statistics.
    """
    env = vec_env(
        [
            lambda: PowerEnv(ep_len=2, power_agent=5.0),
            lambda: PowerEnv(ep_len=3, power_agent=2.0),
        ]
    )
    w = AdversaryWrapper(env)
    w.reset()

    roll_episode(w)
    assert list(w.mean_power_queue) == [5.0, 2.0]

    # per-episode accumulator should be reset to zero after episode termination
    assert w.episode_powers.sum() == 0  # CHANGED THIS LINE
    assert (
        w.episode_powers_baseline.sum() == 0
    )  # ADDED THIS LINE if you have baseline too

    w.reset()  # fresh episode(s)

    # … now cleared, while history queues persist
    assert w.episode_powers.sum() == 0
    assert list(w.mean_power_queue) == [5.0, 2.0]


def test_baseline_queue() -> None:
    """
    If ``"Power baseline"`` is emitted, the wrapper must also keep a separate
    rolling mean for it.
    """
    env = vec_env([lambda: PowerEnv(ep_len=3, power_agent=9.0, power_base=6.0)])
    w = RecordEpisodeVals(env)
    w.reset()
    roll_episode(w)

    assert list(w.mean_power_queue) == [9.0]
    assert list(w.mean_power_queue_baseline) == [6.0]


def test_reset_clears_per_episode_state_only() -> None:
    """
    ``reset()`` wipes **episode-local** state (`episode_powers`,
    `last_dones`) but *retains* the rolling history deques.
    """
    env = vec_env([lambda: PowerEnv(ep_len=2, power_agent=7.0)])
    w = AdversaryWrapper(env)

    w.reset()
    roll_episode(w)

    # Queues populated, accumulators non-zero
    assert w.mean_power_queue
    # After roll_episode, the episode should be terminated, so episode_powers should be reset
    assert w.episode_powers.sum() == 0  # CHANGED THIS LINE

    w.reset()

    # Per-episode accumulators cleared …
    assert w.episode_powers.sum() == 0
    # … but rolling history kept
    assert list(w.mean_power_queue) == [7.0]
