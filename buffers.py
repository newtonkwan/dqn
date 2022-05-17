# modified from https://github.com/henry-prior/jax-rl/blob/master/jax_rl/SAC.py
import jax
import numpy as np
from haiku import PRNGSequence
from jax import random


class ReplayBuffer:
    def __init__(
        self,
        state_dim: np.array,
        max_size: int = int(2e6),
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.empty((max_size, *state_dim))
        self.action = np.empty((max_size, 1), dtype=int)
        self.reward = np.empty((max_size, 1))
        self.next_state = np.empty((max_size, *state_dim))

    def reset(self):
        self.ptr = 0
        self.size = 0

    def add(self, state, action, next_state, reward):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, rng: PRNGSequence, batch_size: int):
        ind = random.randint(rng, (batch_size,), 0, self.size)
        # this randomly samples a starting location 
        return (
            jax.device_put(self.state[ind, :]),
            jax.device_put(self.action[ind, :]), 
            jax.device_put(self.reward[ind, :]),
            jax.device_put(self.next_state[ind, :]),
        )

    def add_batch(self, state, action, next_state, reward):
        batch_size, *state_dim = state.shape
        batch_size_1, action_dim = action.shape

        assert batch_size == batch_size_1
        assert np.array_equal(self.state.shape[1:], np.array(state_dim)) == True

        # first time is divmod(10, 1000)
        q, new_ptr = divmod(self.ptr + batch_size, self.max_size)

        # if q = 0 = False; q = 1 == True
        if q:
            self.ptr = 0

        new_ptr = min(self.ptr + batch_size, self.max_size)

        self.state[self.ptr : new_ptr] = state 
        self.action[self.ptr : new_ptr] = action
        self.reward[self.ptr : new_ptr] = reward.reshape(-1, 1)
        self.next_state[self.ptr : new_ptr] = next_state

        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)
        
    # TODO: add a size method 
    # TODO: add a fraction_filled method
    # TODO: add a def _preallocate function for sequence of items similar to the DeepMind version fom 
    # bsuite