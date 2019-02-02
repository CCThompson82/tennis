import os
import sys
import numpy as np
from src.base_models.base_model import BaseModel


from src.base_networks.actor import Network as Actor
from src.base_networks.critic import Network as Critic
from src.buffers.experience_buffer import ExperienceBuffer
from src.utils.ornstein_uhlenbeck import OUNoise
from collections import OrderedDict
import torch
nn = torch.nn
fn = nn.functional

WORK_DIR = os.environ['ROOT_DIR']
sys.path.append(WORK_DIR)


class DDPG(object):
    def __init__(self, hyperparams, nb_features, nb_actions, params, seed):
        self.hyperparams = hyperparams
        self.actor = Actor(nb_features=nb_features,
                           nb_actions=nb_actions,
                           params=params,
                           seed=seed)
        self.actor_target = Actor(nb_features=nb_features,
                                  nb_actions=nb_actions,
                                  params=params,
                                  seed=seed)
        self.actor.to(self.actor.device)
        self.actor_target.to(self.actor_target.device)

        self.critic = Critic(nb_features=nb_features,
                             nb_actions=nb_actions,
                             params=params,
                             seed=seed)
        self.critic_target = Critic(nb_features=nb_features,
                                    nb_actions=nb_actions,
                                    params=params,
                                    seed=seed)
        self.critic.to(self.critic.device)
        self.critic_target.to(self.critic_target.device)

        # use soft update method to do a full hard-update
        self.soft_update(
            src_model=self.actor, dst_model=self.actor_target, tau=1.0)
        self.soft_update(
            src_model=self.critic, dst_model=self.critic_target, tau=1.0)

        # set up training optimizers
        self.critic_optimizer = torch.optim.Adam(
            params=self.critic.parameters(),
            lr=self.hyperparams['critic_init_learning_rate'])
        self.actor_optimizer = torch.optim.Adam(
            params=self.actor.parameters(),
            lr=self.hyperparams['actor_init_learning_rate'])

        # progress bar attr
        self.critic_loss_ = 0
        self.actor_loss_ = 0

    @staticmethod
    def soft_update(src_model, dst_model, tau):
        for dst_param, src_param in zip(dst_model.parameters(),
                                        src_model.parameters()):
            updated_param = tau*src_param.data + (1.0-tau)*dst_param.data
            dst_param.data.copy_(updated_param)


class Model(BaseModel):
    def __init__(self, model_config, env_config):
        super(Model, self).__init__(model_config=model_config,
                                    env_config=env_config)
        self.agents = [
            DDPG(hyperparams=self.hyperparams,
                 nb_features=env_config['nb_observations'],
                 nb_actions=env_config['nb_actions'],
                 params=self.params,
                 seed=self.hyperparams['random_seed']) for _ in
            range(env_config['nb_agents'])]

        # set up ExperienceReplay buffer
        self.memory = ExperienceBuffer(
            action_size=env_config['nb_actions'],
            max_buffer_size=self.params['experience_buffer']['max_tuples'],
            batch_size=self.hyperparams['batch_size'],
            seed=self.hyperparams['random_seed'])

        # utility for generating noise within actions
        self.noise = OUNoise(env_config['nb_actions'],
                             self.hyperparams['random_seed'],
                             theta=self.hyperparams['noise_theta'],
                             sigma=self.hyperparams['noise_sigma'])

    def terminate_training_status(self, episode_counts, **kwargs):
        return np.mean(episode_counts) >= self.hyperparams['max_episodes']

    @staticmethod
    def terminate_episode_status(max_reached_statuses, local_done_statuses):
        max_reached_status = np.any(max_reached_statuses)
        local_done_status = np.any(local_done_statuses)
        # NOTE: This environment should always return the same value for each
        # agent in both the max_reached and local_done arrays, be it all False
        # or all True.  This may not be the case for other parallel
        # environments.
        return np.any([max_reached_status, local_done_status])

    def progress_bar(self, step_counts, episode_counts, **kwargs):
        return OrderedDict([('step_count', int(np.mean(step_counts))),
                            # ('episode_count', int(np.mean(episode_counts))),
                            ('epsilon', self.epsilon(np.mean(episode_counts))),
                            ('buffer_size', '{}'.format(self.buffer_size)),
                            ('L_critic', '{:7.5f}'.format(
                                np.round(self.agents[0].critic_loss_, 4))),
                            ('L_actor', '{:5.3f}'.format(
                                self.agents[0].actor_loss_)),
                            ('R_critic', '{:7.5f}'.format(
                                np.round(self.agents[1].critic_loss_, 4))),
                            ('R_actor', '{:5.3f}'.format(
                                self.agents[1].actor_loss_)),
                            ('mean episode score', '{:4.3f}'.format(
                                self.mean_episode_score))])

    def get_next_actions(self, states, episode):

        actions = np.zeros([2, 2])
        for agent_idx, perspective_states in enumerate(states):
            state_tensor = torch.from_numpy(perspective_states).float().to(
                self.agents[agent_idx].actor.device)

            self.agents[agent_idx].actor.eval()
            with torch.no_grad():
                _actions = self.agents[agent_idx].actor.forward(
                    state_tensor).cpu().data.numpy()
            self.agents[agent_idx].actor.train()
            actions[agent_idx, :] = _actions

        # TODO: If mode is eval, epsilon is set to zero automatically and this
        #  conditional is never triggered to add noise
        if np.random.rand() <= self.epsilon(episode):
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def store_experience(self, states, actions, rewards, next_states,
                         episode_statuses):

        for idx in range(self.env_config['nb_agents']):
            self.memory.add(state=states[idx],
                            action=actions[idx],
                            reward=rewards[idx],
                            next_state=next_states[idx],
                            done=episode_statuses[idx])

    def check_training_status(self, step):
        """
        Check state of experience buffer or episode status and determine whether
        a training step should be run.

        Returns:
            bool (True if training step should be run)
        """
        status = (self.params['experience_buffer']['min_for_training'] <=
                  self.memory.__len__())
        train_step = step % self.hyperparams['train_frequency'] == 0
        return np.all([status, train_step])
    
    def train_model(self):
        """
        Performs a series of training steps
        """
        training_iterations = self.hyperparams.get('training_steps', 1)
        for step in range(training_iterations):
            self.execute_training_step()

    def execute_training_step(self):
        """
        Coordinates the gradient estimation and backpropagation of the object
        network
        Returns:

        """
        states, actions, rewards, next_states, dones = self.memory.sample()
        states_tensor = torch.from_numpy(states).float()
        actions_tensor = torch.from_numpy(actions).float()
        next_states_tensor = torch.from_numpy(next_states).float()
        rewards_tensor = torch.from_numpy(rewards).float()
        dones_tensor = torch.from_numpy(dones).float()

        for agent in self.agents:


            # train the critic
            future_return = agent.critic_target.forward(
                next_states_tensor, agent.actor_target.forward(
                    next_states_tensor))
            Q_target = rewards_tensor + (
                    self.hyperparams['gamma'] * future_return.cpu() *
                    (1-dones_tensor))

            agent.critic.train()
            Q_expected = agent.critic.forward(states_tensor, actions_tensor)
            critic_loss = fn.mse_loss(Q_expected, Q_target.to(agent.critic.device))

            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.hyperparams['clip_gradient']:
                nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
            agent.critic_optimizer.step()

            # train the actor
            # actions_pred = self.actor.forward(states_tensor)
            actions_pred = agent.actor.forward(states_tensor)
            agent.critic.eval()
            actor_loss = -agent.critic.forward(states_tensor, actions_pred).mean()
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()
            agent.critic.train()

            # record for progress bar
            agent.critic_loss_ = critic_loss.cpu().data.numpy()
            agent.actor_loss_ = actor_loss.cpu().detach().numpy()

            self.soft_update(src_model=agent.critic, dst_model=agent.critic_target,
                             tau=self.hyperparams['tau'])
            self.soft_update(src_model=agent.actor, dst_model=agent.actor_target,
                             tau=self.hyperparams['tau'])

        if np.random.random() < self.hyperparams['noise_reset_epsilon']:
            self.noise.reset()

    @property
    def mean_episode_score(self):
        try:
            arr = np.load(self.dir_util.results_filename)
        except FileNotFoundError:
            return 0
        if len(arr) < 100:
            return np.round(np.mean(arr), 3)

        return np.round(np.mean(arr[-100:, :]), 3)

    @property
    def best_episode_score(self):
        try:
            arr = np.load(self.dir_util.results_filename)
        except FileNotFoundError:
            return 0
        return np.round(np.max(arr), 3)

    @property
    def worst_episode_score(self):
        try:
            arr = np.load(self.dir_util.results_filename)
        except FileNotFoundError:
            return 0
        return np.round(np.min(arr), 3)

    @property
    def buffer_size(self):
        return self.memory.__len__()

    @staticmethod
    def soft_update(src_model, dst_model, tau):
        for dst_param, src_param in zip(dst_model.parameters(),
                                        src_model.parameters()):
            updated_param = tau*src_param.data + (1.0-tau)*dst_param.data
            dst_param.data.copy_(updated_param)

    def checkpoint_model(self, episode_count):
        for agent_idx, agent in enumerate(self.agents):
            self.checkpoint_agent(agent_idx, episode_count)

    def checkpoint_agent(self, agent_idx, episode_count):
        checkpoint_filename = os.path.join(
            self.dir_util.checkpoint_dir, '{}_actor_ckpt_{}.pth'.format(
                agent_idx, episode_count))

        torch.save(
            self.agents[agent_idx].actor.state_dict(), checkpoint_filename)
        checkpoint_filename = os.path.join(
            self.dir_util.checkpoint_dir, '{}_critic_ckpt_{}.pth'.format(
                agent_idx, episode_count))
        torch.save(
            self.agents[agent_idx].critic.state_dict(), checkpoint_filename)

    def epsilon(self, episode):
        if self.model_config['mode'] == 'eval':
            return 0.0

        epf = self.hyperparams['epsilon_root_factor']
        if episode == 0:
            epsilon = 1.0
        else:
            epsilon = (1.0/episode)**(1.0/epf)
        return np.round(epsilon, 3)

    def restore_checkpoint(self, ckpt_index):
        for agent_idx in range(self.env_config['nb_agents']):
            actor_filename = os.path.join(self.dir_util.checkpoint_dir, '{}_actor_ckpt_{}.pth'.format(
                    agent_idx, ckpt_index))
            state_dict = torch.load(actor_filename)
            self.agents[agent_idx].actor.load_state_dict(state_dict)
