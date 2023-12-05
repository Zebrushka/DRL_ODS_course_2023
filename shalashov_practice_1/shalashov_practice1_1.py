import gym
import numpy as np
import random
import time
from gym.envs.toy_text.taxi import *
from IPython.display import clear_output
from matplotlib import pyplot as plt
from tqdm import tqdm
# %matplotlib inline

env = gym.make('Taxi-v3', render_mode='single_rgb_array')

random_state = 42

def render_frame(data):
    clear_output(wait=True)
    plt.imshow(data, interpolation='nearest')
    plt.show()
    time.sleep(0.05)
    clear_output(wait=True)


def plot_reward(data):
    clear_output(wait=True)
    plt.plot(data)
    plt.ylabel('Total reward')
    plt.show()


state_n = env.observation_space.n
action_n = env.action_space.n

print("n_states=%i, n_actions=%i" % (state_n, action_n))
print("states:  %s" % env.observation_space)
print("actions: %s" % env.action_space)

class CrossEntropyAgent():
    """
    state_n - размерность состояний
    action_n - размерность действий
    policy_smoothing - применение сглаживание по политике
    learning_rate - коэффицент сглаживания для политике
    laplace_smoothing - применение сглаживание по Лапласу
    laplace_coef - коэффициент сглаживания по Лапласу
    """


    def __init__(self, state_n, action_n, learning_rate, policy_smoothing=False, laplace_smoothing=False, laplace_coef=0):
        self.state_n = state_n
        self.action_n = action_n
        self.learning_rate = learning_rate
        self.policy_smoothing = policy_smoothing
        self.laplace_smoothing = laplace_smoothing
        self.laplace_coef = laplace_coef
        self.model = np.ones((self.state_n, self.action_n)) * (1.0 / self.action_n)

    def get_action(self, state):
        action = np.random.choice(np.arange(self.action_n), p=self.model[state])

        return int(action)

    def fit(self, elite_trajectories):

        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1

        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                if self.laplace_smoothing:
                    new_model[state] += self.laplace_coef
                    new_model[state] /= np.sum(new_model[state])
                else:
                    new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()

        if self.policy_smoothing:
            self.model = self.learning_rate*new_model + (1-self.learning_rate)*self.model
        else:
            self.model = new_model

        return None



def get_trajectory(env, agent, max_len=10000, visualize=False, logging=False):
    trajectory = {'states':[], 'actions':[], 'rewards':[]}

    state = env.reset()

    for _ in range(max_len):

        action = agent.get_action(state)
        obs, reward, done, _ =  env.step(action)

        # collect stats
        trajectory['states'].append(state)
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)

        state = obs

        if logging:
            print('action', action, 'current state', state)
            clear_output(wait=True)

        if visualize:
            render_frame(env.render())

        if done:
            break

    return trajectory


learning_rate = 0.05
laplace_coef = 1
trajectory_n = 1000
iteration_n = 150
q_param = 0.9


agent = CrossEntropyAgent(
    state_n=state_n,
    action_n=action_n,
    learning_rate=learning_rate,
    policy_smoothing=False,
    laplace_smoothing=True,
    laplace_coef=laplace_coef
)

rewards = []

for iteration in tqdm(range(iteration_n)):
    # policy eval
    trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
    total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]

    # пишем стату для графика наград
    rewards.append(np.mean(total_rewards))
    plot_reward(rewards)
    # логируем стату
    print('iter: ',iteration, ', total_rewards: ', np.mean(total_rewards))


    # policy improvement
    quantile = np.quantile(total_rewards, q_param)
    elite_trajectories = []
    for trajectory in trajectories:
        total_reward = np.sum(trajectory['rewards'])

        if total_reward > quantile:
            elite_trajectories.append(trajectory)

    agent.fit(elite_trajectories)



trajectory = get_trajectory(env, agent, visualize=True)

print('total reward:', sum(trajectory['rewards']))
print('model:')
print(agent.model)