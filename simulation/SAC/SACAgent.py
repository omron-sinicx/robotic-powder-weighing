import torch
import os
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

from DatasetHandler.ReplayMemory import ReplayMemory
from .Actor.ActorBasic import ActorBasic
from .Critic.CriticBasic import CriticBasic
from .Critic.CriticLSTM import CriticLSTM
from .Actor.ActorLSTM import ActorLSTM
from .EntropyTerm.EntropyTerm import EntropyTerm
from LearningCommonParts.TotalRewardService import TotalRewardService, ProgressChecker


class SACAgent(object):
    def __init__(self, env, userDefinedSettings):
        self.env = env
        self.userDefinedSettings = userDefinedSettings
        self.replay_buffer = ReplayMemory(env.STATE_DIM, env.ACTION_DIM, env.MAX_EPISODE_LENGTH, env.DOMAIN_PARAMETER_DIM, userDefinedSettings)
        if userDefinedSettings.LSTM_FLAG:
            self.critic = CriticLSTM(env.STATE_DIM, env.ACTION_DIM, env.DOMAIN_PARAMETER_DIM, userDefinedSettings)
            self.actor = ActorLSTM(env.STATE_DIM, env.ACTION_DIM, userDefinedSettings)
        else:
            self.critic = CriticBasic(env.STATE_DIM, env.ACTION_DIM, env.DOMAIN_PARAMETER_DIM, userDefinedSettings)
            self.actor = ActorBasic(env.STATE_DIM, env.ACTION_DIM, userDefinedSettings)
        self.entropyTerm = EntropyTerm(env.ACTION_DIM, userDefinedSettings)
        self.totalRewardService = TotalRewardService(self.userDefinedSettings)
        self.taskAchivementService = ProgressChecker(self.userDefinedSettings)

        self.current_episode_num = self.userDefinedSettings.current_episode_num

    def train(self, domain_num=None, expert_value_function=None, expert_policy=None):
        self.model_dir = os.path.join(self.userDefinedSettings.LOG_DIRECTORY, 'model', str(domain_num))
        self.summary_dir = os.path.join(self.userDefinedSettings.LOG_DIRECTORY, 'summary', str(domain_num))

        self.summaryWriter = SummaryWriter(log_dir=self.summary_dir)

        if self.current_episode_num != 0:
            self.load_model(self.model_dir)
            self.load_dataset()
        else:
            self.summary_writer_count = 0

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        for episode_num in range(self.current_episode_num, self.current_episode_num + self.userDefinedSettings.learning_episode_num):

            if episode_num < self.userDefinedSettings.policy_update_start_episode_num and expert_policy is not None:
                get_action_fn = expert_policy.get_action
            else:
                get_action_fn = self.actor.get_action

            self.env.domainInfo.set_parameters()
            state = self.env.reset()

            total_reward = 0.
            for step_num in range(self.env.MAX_EPISODE_LENGTH):
                policy_update_flag = self.is_update(episode_num)
                action, method_depend_info = get_action_fn(state, step=step_num, deterministic=False, random_action_flag=not policy_update_flag)
                next_state, reward, done, domain_parameter, task_achievement = self.env.step(action, get_task_achievement=True)

                self.replay_buffer.push(state, action, reward, next_state, done, method_depend_info, domain_parameter=domain_parameter, step=step_num)

                state = next_state
                total_reward += reward

                if policy_update_flag:
                    for _ in range(self.userDefinedSettings.updates_per_step):
                        self.update(self.userDefinedSettings.batch_size, expert_value_function=expert_value_function, episode_num=episode_num)

                if done:
                    break

            print('Episode: {:>5} | Episode Reward: {:>8.2f}| model updated!!'.format(episode_num, total_reward))
            self.save_model()
            self.summaryWriter.add_scalar('status/train reward', total_reward, episode_num)

        self.save_dataset(self.current_episode_num + self.userDefinedSettings.learning_episode_num)

    def save_dataset(self, batch_size):
        batch = self.replay_buffer.sample(sampling_method='all', get_debug_term_flag=True)
        torch.save(batch, os.path.join(self.model_dir, 'dataset.pth'))

    def load_dataset(self):
        batch_data = torch.load(os.path.join(self.model_dir, 'dataset.pth'))
        self.replay_buffer.add_from_other(batch_data=batch_data)

    def sample_dataset(self, replay_buffer, sample_episode_num=1):
        for episode_num in range(sample_episode_num):
            total_reward = 0.
            state = self.env.reset()
            for step_num in range(self.env.MAX_EPISODE_LENGTH):
                action, method_depend_info = self.actor.get_action(state, step=step_num, deterministic=True, random_action_flag=False)
                next_state, reward, done, domain_parameter = self.env.step(action)
                replay_buffer.push(state, action, reward, next_state, done, method_depend_info, domain_parameter=domain_parameter, step=step_num)
                total_reward += reward
                if done:
                    break
                state = next_state

    def is_update(self, episode_num):
        return len(self.replay_buffer) > self.userDefinedSettings.batch_size and episode_num > self.userDefinedSettings.policy_update_start_episode_num

    def update(self, batch_size, expert_value_function=None, episode_num=None):
        if self.userDefinedSettings.LSTM_FLAG:
            self.update_lstm(batch_size, expert_value_function=expert_value_function, episode_num=episode_num)
        else:
            self.update_basic(batch_size, expert_value_function=expert_value_function)

    def update_basic(self, batch_size, expert_value_function=None, episode_num=None):
        batch = self.replay_buffer.sample(batch_size)
        state, action, reward, next_state, done, lstm_info, domain_parameter = batch

        # Updating entropy term
        _, log_prob, std = self.actor.evaluate(state)
        predict_entropy = -log_prob
        entropy_loss = self.entropyTerm.update(predict_entropy.detach())
        self.summaryWriter.add_scalar('status/standard deviation', std.detach().mean().item(), self.summary_writer_count)
        self.summaryWriter.add_scalar('loss/entropy', entropy_loss.detach().item(), self.summary_writer_count)

        # Training Q Function
        new_next_action, next_log_prob, _ = self.actor.evaluate(next_state)
        q1_loss, q2_loss, predicted_q1, predicted_q2 = self.critic.update(state, action, reward, next_state, done,
                                                                          new_next_action.detach(),
                                                                          next_log_prob.detach(), self.entropyTerm.alpha.detach(),
                                                                          domain_parameter,
                                                                          expert_value_function=expert_value_function,
                                                                          episode_num=episode_num)
        self.summaryWriter.add_scalar('status/Q1', predicted_q1.detach().mean().item(), self.summary_writer_count)
        self.summaryWriter.add_scalar('status/Q2', predicted_q2.detach().mean().item(), self.summary_writer_count)
        self.summaryWriter.add_scalar('loss/Q1', q1_loss.detach().item(), self.summary_writer_count)
        self.summaryWriter.add_scalar('loss/Q2', q2_loss.detach().item(), self.summary_writer_count)

        # Training Policy Function
        new_action, log_prob, _ = self.actor.evaluate(state)
        q_value = self.critic.predict_q_value(state, new_action, domain_parameter)
        policy_loss = self.actor.update(self.entropyTerm.alpha.detach(), log_prob, q_value)
        self.summaryWriter.add_scalar('loss/policy', policy_loss.detach().item(), self.summary_writer_count)

        # Q value soft update
        self.critic.soft_update()

        # tensorboard horizonall value
        self.summary_writer_count += 1

    def update_lstm(self, batch_size, expert_value_function=None, episode_num=None):
        batch = self.replay_buffer.sample(batch_size)
        state, action, reward, next_state, done, lstm_term, domain_parameter = batch

        # Updating entropy term
        _, log_prob, std = self.actor.evaluate(state, lstm_term['last_action'], lstm_term['hidden_in'])
        predict_entropy = -log_prob
        entropy_loss = self.entropyTerm.update(predict_entropy.detach())
        self.summaryWriter.add_scalar('status/standard deviation', std.detach().mean().item(), self.summary_writer_count)
        print(self.summary_writer_count)
        self.summaryWriter.add_scalar('loss/entropy', entropy_loss.detach().item(), self.summary_writer_count)

        # Training Q Function
        new_next_action, next_log_prob, _ = self.actor.evaluate(next_state, action, lstm_term['hidden_out'])
        q1_loss, q2_loss, predicted_q1, predicted_q2 = self.critic.update(state, action, reward, next_state, done,
                                                                          lstm_term, new_next_action.detach(),
                                                                          next_log_prob.detach(), self.entropyTerm.alpha.detach(),
                                                                          domain_parameter, actor=self.actor, episode_num=episode_num)
        self.summaryWriter.add_scalar('status/Q1', predicted_q1.detach().mean().item(), self.summary_writer_count)
        self.summaryWriter.add_scalar('status/Q2', predicted_q2.detach().mean().item(), self.summary_writer_count)
        self.summaryWriter.add_scalar('loss/Q1', q1_loss.detach().item(), self.summary_writer_count)
        self.summaryWriter.add_scalar('loss/Q2', q2_loss.detach().item(), self.summary_writer_count)

        # Training Policy Function
        new_action, log_prob, _ = self.actor.evaluate(state, lstm_term['last_action'], lstm_term['hidden_in'])  # changed
        q_value = self.critic.predict_q_value(state, new_action, lstm_term['last_action'], lstm_term['hidden_in'], domain_parameter)
        policy_loss = self.actor.update(self.entropyTerm.alpha.detach(), log_prob, q_value)
        self.summaryWriter.add_scalar('loss/policy', policy_loss.detach().item(), self.summary_writer_count)

        # Q value soft update
        self.critic.soft_update()

        # tensorboard horizonall value
        self.summary_writer_count += 1

    def save_model(self, model_dir=None):
        if model_dir is not None:
            self.model_dir = model_dir
        torch.save(self.critic.soft_q_net1.state_dict(), os.path.join(self.model_dir, 'Q1.pth'))
        torch.save(self.critic.soft_q_net2.state_dict(), os.path.join(self.model_dir, 'Q2.pth'))
        torch.save(self.critic.target_soft_q_net1.state_dict(), os.path.join(self.model_dir, 'target_Q1.pth'))
        torch.save(self.critic.target_soft_q_net2.state_dict(), os.path.join(self.model_dir, 'target_Q2.pth'))
        torch.save(self.actor.policyNetwork.state_dict(), os.path.join(self.model_dir, 'Policy.pth'))
        torch.save(self.entropyTerm.log_alpha, os.path.join(self.model_dir, 'Entropy.pth'))
        torch.save(self.summary_writer_count, os.path.join(self.model_dir, 'summary_writer_count.pth'))

    def load_model(self, path=None, load_only_policy=False):
        if path is not None:
            self.model_dir = path

        if not load_only_policy:
            self.critic.soft_q_net1.load_state_dict(torch.load(os.path.join(self.model_dir, 'Q1.pth'), map_location=torch.device(self.userDefinedSettings.DEVICE)))
            self.critic.soft_q_net2.load_state_dict(torch.load(os.path.join(self.model_dir, 'Q2.pth'), map_location=torch.device(self.userDefinedSettings.DEVICE)))
            self.critic.target_soft_q_net1.load_state_dict(torch.load(os.path.join(self.model_dir, 'target_Q1.pth'), map_location=torch.device(self.userDefinedSettings.DEVICE)))
            self.critic.target_soft_q_net2.load_state_dict(torch.load(os.path.join(self.model_dir, 'target_Q2.pth'), map_location=torch.device(self.userDefinedSettings.DEVICE)))
            self.entropyTerm.log_alpha = torch.load(os.path.join(self.model_dir, 'Entropy.pth'))
            self.summary_writer_count = torch.load(os.path.join(self.model_dir, 'summary_writer_count.pth'))

        self.actor.policyNetwork.load_state_dict(torch.load(os.path.join(self.model_dir, 'Policy.pth'), map_location=torch.device(self.userDefinedSettings.DEVICE)))

    def test(self, model_path=None, policy=None, domain_num=None, test_num=5, render_flag=True, reward_show_flag=True):
        if model_path is not None:
            self.load_model(model_path)

        if policy is not None:
            actor = policy
        else:
            actor = self.actor

        total_reward_list = []
        task_achievement_list = []
        for episode_num in range(test_num):
            self.env.domainInfo.set_parameters()
            state = self.env.reset()
            total_reward = 0.

            if render_flag:
                self.env.render()
                time.sleep(1)

            for step_num in range(self.env.MAX_EPISODE_LENGTH):
                if render_flag is True:
                    self.env.render()

                action, _ = actor.get_action(state, step=step_num, deterministic=False)
                next_state, reward, done, _, task_achievement = self.env.step(action, get_task_achievement=True)
                state = next_state
                total_reward += reward
                if done:
                    break

            total_reward_list.append(total_reward)
            task_achievement_list.append(task_achievement)

            if reward_show_flag is True:
                print('Tests: {:>5} | Total Reward: {:>8.2f} | Task Achievement: {}'.format(episode_num, total_reward, task_achievement))

        if model_path is not None:
            print('Avarage: {:>8.2f}'.format(np.mean(total_reward_list)))

        return np.mean(total_reward_list), sum(task_achievement_list) / len(task_achievement_list)
