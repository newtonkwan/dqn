# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""A simple JAX-based DQN implementation.
Reference: "Playing atari with deep reinforcement learning" (Mnih et al, 2015).
Link: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf.
"""

from typing import Any, Callable, NamedTuple, Sequence

from bsuite.baselines import base
from bsuite.baselines.utils import replay
from bsuite.environments import catch

import logging
import wandb
import hydra
import dm_env
from dm_env import specs
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import matplotlib.pyplot as plt
import buffers

def global_setup(args):
    '''Set up global variables.'''
    if args.wandb.log:
        wandb.init(
            entity=str(args.wandb.entity),
            project=str(args.wandb.project),
            group=str(args.wandb.group),
            name=str(args.wandb.name),
            config=vars(args),
        )
def env_setup(num_envs):
    '''Set up env variables.'''
    train_envs = [catch.Catch(seed=i) for i in range(num_envs)]
    eval_env = catch.Catch(seed=num_envs)
    return train_envs, eval_env

def next_timesteps(envs, actions):
    '''Take timesteps in each environment'''
    new_timesteps = []
    for env, action in zip(envs, actions):
      new_timesteps.append(env.step(action.squeeze()))
    #new_timesteps = [env.step(action) for env, action in zip(envs, actions)]

    return new_timesteps

class TrainingState(NamedTuple):
  """Holds the agent's training state."""
  params: hk.Params
  target_params: hk.Params
  opt_state: Any
  step: int

class DQN(base.Agent):
  """A simple DQN agent using JAX."""

  def __init__(
      self,
      obs_spec: specs.Array,
      action_spec: specs.DiscreteArray,
      network: Callable[[jnp.ndarray], jnp.ndarray],
      optimizer: optax.GradientTransformation,
      batch_size: int,
      epsilon: float,
      rng: hk.PRNGSequence,
      discount: float,
      replay_capacity: int,
      min_replay_size: int,
      sgd_period: int,
      target_update_period: int,
  ):
    # Transform the (impure) network into a pure function.
    network = hk.without_apply_rng(hk.transform(network))

    # Define loss function.
    def loss(params: hk.Params,
             target_params: hk.Params,
             transitions: Sequence[jnp.ndarray]) -> jnp.ndarray:
      """Computes the standard TD(0) Q-learning loss on batch of transitions."""
      # observation, action, reward, discount, observation
      o_tm1, a_tm1, r_t, o_t = transitions
      q_tm1 = network.apply(params, o_tm1)
      q_t = network.apply(target_params, o_t)
      batch_q_learning = jax.vmap(rlax.q_learning)
      a_tm1 = a_tm1.squeeze()
      r_t = r_t.squeeze()
      td_error = batch_q_learning(q_tm1, a_tm1, r_t, discount * jnp.ones_like(r_t), q_t)
      return jnp.mean(td_error**2)

    # Define update function.
    @jax.jit
    def sgd_step(state: TrainingState,
                 transitions: Sequence[jnp.ndarray]) -> TrainingState:
      """Performs an SGD step on a batch of transitions."""
      gradients = jax.grad(loss)(state.params, state.target_params, transitions)
      updates, new_opt_state = optimizer.update(gradients, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)

      return TrainingState(
          params=new_params,
          target_params=state.target_params,
          opt_state=new_opt_state,
          step=state.step + 1)

    # Initialize the networks and optimizer.
    dummy_observation = np.zeros((1, *obs_spec.shape), jnp.float32)
    initial_params = network.init(next(rng), dummy_observation)
    initial_target_params = network.init(next(rng), dummy_observation)
    initial_opt_state = optimizer.init(initial_params)

    # This carries the agent state relevant to training.
    self._state = TrainingState(
        params=initial_params,
        target_params=initial_target_params,
        opt_state=initial_opt_state,
        step=0)
    self._sgd_step = sgd_step
    self._forward = jax.jit(network.apply)
    self._replay = buffers.ReplayBuffer(
      state_dim = obs_spec.shape, 
      max_size = replay_capacity
      )

    # Store hyperparameters.
    self._num_actions = action_spec.num_values
    self._batch_size = batch_size
    self._sgd_period = sgd_period
    self._target_update_period = target_update_period
    self._epsilon = epsilon
    self._total_steps = 0
    self._min_replay_size = min_replay_size

  def select_action(self, key, timesteps: list) -> jnp.array:
    """Selects batched actions according to an epsilon-greedy policy."""
    key, subkey = jax.random.split(key)
    num_envs = len(timesteps)
    # mask = np.random.rand(num_envs, 1) < self._epsilon
    mask = jax.random.uniform(key, shape = (num_envs, 1)) < self._epsilon
    observations = jnp.stack([_t.observation for _t in timesteps])
    q_values = self._forward(self._state.params, observations)
    actions = jnp.where(
        mask==True, 
        # x = np.random.randint(self._num_actions), 
        x = jax.random.randint(subkey, shape=(),minval=0, maxval = self._num_actions, dtype=int), 
        y = jnp.argmax(q_values, axis=1, keepdims=True)) # keepdims=True preserves the correct shape 
    assert actions.shape == (num_envs, 1)
    return actions


  def select_action_eval(self, key, timestep: dm_env.TimeStep) -> base.Action:
    """Selects actions according to a greedy policy."""
    # Greedy policy, breaking ties uniformly at random.
    observation = timestep.observation[None, ...]
    q_values = self._forward(self._state.target_params, observation)
    # action = np.random.choice(np.flatnonzero(q_values == q_values.max()))
    action = jax.random.choice(key, np.flatnonzero(q_values == q_values.max())) # np.random.choice(np.flatnonzero(q_values == q_values.max()))
    return int(action)

  def update(
      self,
      key, 
      timesteps: list,
      actions: jnp.array, 
      new_timesteps: list,
  ):
    """Adds transition to replay and periodically does SGD."""
    # Add this transition to replay.
    observations = jnp.stack([_t.observation for _t in timesteps])
    new_observations = jnp.stack([_t.observation for _t in new_timesteps])
    rewards = jnp.stack([_t.reward for _t in new_timesteps])
    self._replay.add_batch(
        state = observations,
        action = actions,
        reward = rewards, 
        next_state = new_observations, 
        )

    self._total_steps += 1
    if self._total_steps % self._sgd_period != 0:
      return

    if self._replay.size < self._min_replay_size:
      return

    # Do a batch of SGD.
    transitions = self._replay.sample(rng = key, batch_size = self._batch_size)
    self._state = self._sgd_step(self._state, transitions)

    # Periodically update target parameters.
    if self._state.step % self._target_update_period == 0:
      self._state = self._state._replace(target_params=self._state.params)

def default_agent(args, 
                  obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray
                  ) -> base.Agent:
  """Initialize a DQN agent with default parameters."""

  def network(inputs: jnp.ndarray) -> jnp.ndarray:
    flat_inputs = hk.Flatten()(inputs)
    mlp = hk.nets.MLP([64, 64, action_spec.num_values])
    action_values = mlp(flat_inputs)
    return action_values

  return DQN(
      obs_spec=obs_spec,
      action_spec=action_spec,
      network=network,
      optimizer=optax.adam(args.learning_rate),
      batch_size=args.batch_size,
      discount=args.discount,
      replay_capacity=args.replay_capacity,
      min_replay_size=args.min_replay_size,
      sgd_period=args.sgd_period,
      target_update_period=args.target_update_period,
      epsilon=args.epsilon,
      rng=hk.PRNGSequence(args.seed),
  )
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

@hydra.main(config_path="conf", config_name="config")
def main(args):
    # setup WandB
    global_setup(args)
    key = jax.random.PRNGKey(args.seed)

    # Create train and test environments
    train_envs, eval_env = env_setup(
      num_envs = args.num_envs
      )

    # Initiliaze the agent
    agent = default_agent(
      args, 
      obs_spec = eval_env.observation_spec(), 
      action_spec = eval_env.action_spec()
      )

    print(f"Training agent for {args.train_episodes} episodes with {args.num_envs} environments ...")
    all_episode_returns = []
    for episode in range(args.train_episodes):
        # Run an episode.
        episode_return = 0. 

        # Timestep: (step_type, reward, discount, observation)
        # Create a list of initial timesteps for each environment 
        timesteps = [env.reset() for env in train_envs]

        # check if the episode in the first environment has terminated 
        # all of the episodes end after 10 steps, so this is okay. 
        while not timesteps[0].last():
            # Actions: 0, 1, 2 = (-1, 0, 1) = (go left, stay, go right)
            # Select actions from the agent's policy for each environment 
            # Expected shape: np.array (num_envs, 1)
            key, subkey, subsubkey = jax.random.split(key, num = 3)
            actions = agent.select_action(subkey, timesteps)

            # Take steps sequentially for each environment 
            new_timesteps = next_timesteps(train_envs, actions)

            # Tell the agent what just happened     
            agent.update(subsubkey, timesteps, actions, new_timesteps)

            # episode_returns += np.array([new_timestep.reward for new_timestep in new_timesteps])
            episode_return += np.mean(np.array([_t.reward for _t in new_timesteps]))
            timesteps = new_timesteps
        
        all_episode_returns.append(episode_return)
        smoothed_return = moving_average(all_episode_returns, 20)
        if args.wandb.log: 
            wandb.log({"train": float(smoothed_return[-1])})
        if episode % 5 == 0:
            print(f"Episode: {episode} with smoothed return: {episode_return}")

    # Evaluate the agent using the target network and greedy-policy
    print(f"Evaluating agent for {args.eval_episodes} episodes...")
    all_episode_returns = []
    # Timestep: (step_type, reward, discount, observation)
    for _ in range(args.eval_episodes):
        # Run an episode.
        episode_return = 0. 
        timestep = eval_env.reset()
        #print(all_episode_returns)
        while not timestep.last():
            key, subkey = jax.random.split(key)

            # Generate an action from the agent's policy.
            action = agent.select_action_eval(subkey, timestep)

            # Step the environment.
            new_timestep = eval_env.step(action)

            # Comment out during evaluation
            # Tell the agent about what just happened.
            # agent.update(timestep, action, new_timestep)

            # Book-keeping.
            episode_return += new_timestep.reward 
            timestep = new_timestep
        all_episode_returns.append(episode_return)
        smoothed_returns = moving_average(all_episode_returns, 10)
        if args.wandb.log: 
          wandb.log({"evaluation": float(smoothed_returns[-1])})
        if _ % 20 == 0:
            print(f"Episode: {_} with smoothed return {episode_return}")

if __name__ == "__main__":
    main() 