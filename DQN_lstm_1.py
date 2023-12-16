
from LSTM_approximator import Customlstm
from temp_environment import my_env



import gym

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

import matplotlib.pyplot as plt

import numpy as np
import torch as th




policy_kwargs = dict(activation_fn=th.nn.Identity,net_arch=[], features_extractor_class=Customlstm,
    features_extractor_kwargs=dict(features_dim=512))
def plot(argument,title):
    plt.plot(np.arange(argument.__len__()),argument)
    plt.title(title)
    plt.show()


path = 'F:/saved_moedls/'
tmp_fix = 37.7
initial_tmp = 36
humid_fix = 65
initial_humid = 30
tmp_out = 25
humid_out = 20

def evaluate_model(model,num_timestep,seed=None):
  env = my_env(tmp_fix,initial_tmp,humid_fix,initial_Humid,tmp_out,humid_out, number_of_timesteps=1000,seed=seed)
  obs = env.current_state()
  rewards = []

  for i in range(num_timestep):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done,_, info = env.step(action)
    # buffer_action = model.policy.scale_action(action)


    rewards.append(reward)

  return np.sum(rewards)

tmp_fix = 37.7
initial_Tmp = 36
humid_fix = 65
initial_Humid = 30
Tmp_out = 25
Humid_out = 20


# If the environment don't follow the interface, an error will be thrown
# check_env(env, warn=True)




all_rewards_episodes = []
all_rewards_generalized_episodes = []
for k in range(4):
    seed = None
    # env = gym.make(env_name)
    env = my_env(tmp_fix, initial_Tmp, humid_fix, initial_Humid, Tmp_out, Humid_out, number_of_timesteps=1000,
                 change_setting=False,
                 limit=False, seed=seed)
    all_rewards = []
    model = DQN("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs, gamma=1.000, learning_starts=1000,
                learning_rate=4 * 10 ** -4, batch_size=128, buffer_size=45000, train_freq=4, exploration_final_eps=0.05,
                max_grad_norm=10, gradient_steps=5,
                tau=0.2, target_update_interval=100)

    best_reward = evaluate_model(model, 1000, seed=seed)

    print("reward summation is :{}".format(best_reward))
    for i in range(200):

        model.learn(total_timesteps=2000,reset_num_timesteps=False,log_interval=2)
        temp_reward = evaluate_model(model,1000,seed=seed)
        all_rewards.append(temp_reward)
        model.save(path+"dqn_lstm_{}_{}".format(k,i))
        model.save_replay_buffer(path+"dqn_lstm_buffer_{}_{}".format(k,i))
        print("reward summation is :{}".format(temp_reward))
        if temp_reward > best_reward:
          best_reward = temp_reward
          model.save(path+"best_dqn_lstm_{}".format(k))
          print("model saved!")
    sorted = np.array(all_rewards).argsort()
    all_rewards_generalized = []
    all_rewards_episodes.append(all_rewards)
    np.save(path+'all_rewards_episodes_lstm.npy', all_rewards_episodes)

    for j in range(8):
        new_env = my_env(tmp_fix,initial_Tmp,humid_fix,initial_Humid,Tmp_out,Humid_out, number_of_timesteps=1000, change_setting=False,
                 limit=False, seed=j)
        rewards_generalized = []
        model = DQN.load(path+"dqn_lstm_{}_{}".format(k, sorted[-1]), new_env)
        model.load_replay_buffer(path+"dqn_lstm_buffer_{}_{}".format(k, sorted[-1]))
        best_reward = evaluate_model(model, 1000, seed=j % 4)
        rewards_generalized.append(best_reward)
        for i in range(149):
            model.learn(total_timesteps=1000, reset_num_timesteps=False, log_interval=2)
            temp_reward = evaluate_model(model, 1000,seed=j % 4)
            rewards_generalized.append(temp_reward)

            model.save(path+"dqn_lstm_generalized_{}_{}_{}".format(k, j, i))
            print("reward summation is :{}".format(temp_reward))
            if temp_reward > best_reward:
                best_reward = temp_reward
                model.save(path+"best_dqn_lstm_generalized_{}".format(k))
                print("model saved!")
        all_rewards_generalized.append(rewards_generalized)


    all_rewards_generalized_episodes.append(all_rewards_generalized)
    np.save(path+'all_rewards_generalized_episodes_lstm.npy', all_rewards_generalized_episodes)


all_episodes_concatenated = np.concatenate(all_rewards_generalized_episodes,axis=0)
ten_percent_quantile = np.max(all_episodes_concatenated,axis=1) - (np.max(all_episodes_concatenated,axis=1) - all_episodes_concatenated[:,0])/10
#
first_indexes = np.argmax(all_episodes_concatenated>=np.expand_dims(ten_percent_quantile, axis=1).repeat(all_episodes_concatenated.shape[1],axis=1),axis=1)
#
#
mean = np.mean(first_indexes)
var = np.var(first_indexes)
e



def plot_learning_curves(argument):
    fig, axs = plt.subplots(2, 2)
    x=np.arange(0,2*argument[0, :].__len__(),2)
    axs[0, 0].plot(x, argument[0, :])
    axs[0, 0].set_title('Evaluation rewards')
    axs[0, 1].plot(x, argument[1, :], 'tab:orange')
    axs[0, 1].set_title('Evaluation rewards')
    axs[1, 0].plot(x, argument[2, :], 'tab:green')
    axs[1, 0].set_title('Evaluation rewards')
    axs[1, 1].plot(x, argument[3, :], 'tab:red')
    axs[1, 1].set_title('Evaluation rewards')

    axs[0, 0].set_ylim([-200000, -5000])
    axs[0, 1].set_ylim([-200000, -5000])
    axs[1, 0].set_ylim([-200000, -5000])
    axs[1, 1].set_ylim([-200000, -5000])
    axs[0, 0].grid(axis="y")
    axs[0, 1].grid(axis="y")
    axs[1, 0].grid(axis="y")
    axs[1, 1].grid(axis="y")



