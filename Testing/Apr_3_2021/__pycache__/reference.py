import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import HER, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

PROCESSES_TO_USE = 1 # turns out you can't parallelize HER in sb3
NUM_EXPERIMENTS = 0 # unused for now
NUM_EPOCHS = 400 # one epoch = 5000 timesteps because I said so
NUM_TIMESTEPS = 5000
EVAL_EPS = 100 # How many episodes to use for eval
MODEL_CLASS = TD3

def evaluate_agent(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    rewards = []
    successes = []
    for i in range(num_episodes):
        episode_rewards = []
        episode_successes = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_successes.append(info[0]['is_success'])
            episode_rewards.append(reward)
        rewards.append(sum(episode_rewards))
        successes.append(np.sum(episode_successes) > 0)
    avg_eval_rew = np.mean(rewards)
    avg_eval_acc = np.mean(successes)
    print("EVAL AVG REWARD:", avg_eval_rew)
    print("EVAL SUCCESS RATE: ",np.round(avg_eval_acc,3))
    return avg_eval_rew, avg_eval_acc
    

env_id = 'FetchPickAndPlace-v1'
num_cpu = 10 # unused, but would be nice
vec_env = make_vec_env(env_id, n_envs = 1)

# Note that obs clipping is applied AFTER normalization.
# https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/common/vec_env/vec_normalize.html#VecNormalize.normalize_obs
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,clip_obs=5.)

eval_env = gym.make(env_id)

n_actions = eval_env.action_space.shape[0]
action_noise = NormalActionNoise(mean = np.zeros(n_actions), sigma = 0.2 * np.ones(n_actions))

model = HER('MlpPolicy', vec_env, MODEL_CLASS, 
    n_sampled_goal = 4,
    goal_selection_strategy = 'future',
    online_sampling = True,
    verbose = 1, 
    action_noise = action_noise,
    gamma = .978,
    tau = .95,
    buffer_size = int(1e7),
    batch_size= 512,
    learning_starts = 10000,
    train_freq = 1000,
    gradient_steps = 1000,
    policy_kwargs= dict(
        net_arch = [350,400,350],
        optimizer_kwargs = dict(weight_decay = 1.)
        )
    )


eval_rews = []
eval_accs = []
for i in range(0,NUM_EPOCHS):
    model = model.learn(total_timesteps = NUM_TIMESTEPS, reset_num_timesteps = False)
    eval_rew, eval_acc = evaluate_agent(model)
    eval_rews.append(eval_rew)
    eval_accs.append(eval_acc)
    if i % 50 == 0:
        fig = plt.figure()
        plt.subplot(2,1,1)
        plt.plot(eval_rews)
        plt.subplot(2,1,2)
        plt.plot(eval_accs)
        fig.savefig('pickplace_results_epoch_'+str(i)+'.png')





obs = env.reset()
for _ in range(10):
    action, _state = model.predict(obs, deterministic = True)
    obs, reward, done, info = env.step(action)
    print(info['is_success'])
    #env.render()
    if done:
        obs = env.reset()

env.close()