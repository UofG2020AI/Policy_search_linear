import matplotlib.pyplot as plt
import numpy as np
import time
import itertools
import sys

def create_policy(func_approximator, epsilon, nA):
    """
    Creates an greedy policy with the exploration defined by the epsilon and nA parameters
    
    Input:
        func_approximator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Output:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(state):
        """
        Input:
            state: a 2D array with the position and velocity
        Output:
            A,q_values: 
        """
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = func_approximator.predict(state)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A,q_values  # return the potentially stochastic policy (which is due to the exploration)

    return policy_fn # return a handle to the function so we can call it in the future


def exec_policy(env, func_approximator, verbose=False):
    """
        A function for executing a policy given the funciton
        approximation (the exploration is zero)
    """

    # The policy is defined by our function approximator (of the utility)... let's get a hdnle to that function
    policy = create_policy(func_approximator, 0.0, env.action_space.n)
            
    # Reset the environment and pick the first action
    state = env.reset()
    rewards = []  
    states = []
    actions = []
    # One step in the environment
    for t in itertools.count():
        #env.render()

        # The policy is stochastic due to exploration 
        # i.e. the policy recommends not only one action but defines a 
        # distrbution , \pi(a|s)
        pi_action_state, q_values = policy(state)
        action_probs = np.max(pi_action_state)
        action = list(pi_action_state).index(action_probs)
        #print("Action (%s): %s" % (action_probs,action))

        # Execute action and observe where we end up incl reward
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        states.append(next_state)
        actions.append(action)
        if verbose:
            print("Step %d/199:\n" % (t), end="")
            print("\t state     : %s\n" % (state), end="")            
            print("\t q_approx  : %s\n" % (q_values.T), end="")
            print("\t pi(a|s)   : %s\n" % (pi_action_state), end="")            
            print("\t action    : %s\n" % (action), end="")
            print("\t next_state: %s\n" % (next_state), end="")
            print("\t reward    : %s\n" % (reward), end="")                        
        else:
            print("\rStep {}".format(t), end="")
       
        if done:
            break
            
        state = next_state
    return states, rewards, actions


from collections import namedtuple
# Keep track of some stats
EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
rewards = []
# Main Q-learner
def q_learning(env, func_approximator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
    """
    Q-Learning algorithm for Q-learning using Function Approximations.
    Finds the optimal greedy policy while following an explorative greedy policy.
    
    Args:
        env: OpenAI environment.
        func_approximator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        epsilon: Exploration strategy; chance the sample a random action. Float between 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    for i_episode in range(num_episodes):
        
        # Create a handle to the policy we're following (including exploration)a
        policy = create_policy(
            func_approximator, epsilon * epsilon_decay**i_episode, env.action_space.n)
        
        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = stats.episode_rewards[i_episode - 1]
        sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset()
                 
        # One step in the environment
        for t in itertools.count():
                        
            # Choose an action to take                        
            action_probs, q_vals = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            
            # Take a step
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # TD Update
            q_values_next = func_approximator.predict(next_state)
            
            # Q-Value TD Target
            td_target = reward + discount_factor * np.max(q_values_next)
              
            # Update the function approximator using our target
            func_approximator.update(state, action, td_target)
            
            print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, last_reward), end="")
                
            if done:
                break
                
            state = next_state

    return stats


import virl
def get_env():
    for i in range(10):
        for j in range(2):
            if j:
                noisy = 'True'
            else:
                noisy = 'False'
            problem_id = i
            yield problem_id, noisy, virl.Epidemic(problem_id=i, noisy=j)


def get_fig(states, rewards):
    fig, axes = plt.subplots(1, 2, figsize=(8 * 2, 6))
    labels = ['s[0]: susceptibles', 's[1]: infectious', 's[2]: quarantined', 's[3]: recovereds']
    states = np.array(states)
    for i in range(4):
        axes[0].plot(states[:,i], label=labels[i])
    axes[0].set_xlabel('weeks since start of epidemic')
    axes[0].set_ylabel('State s(t)')
    axes[0].legend()
    axes[1].plot(rewards)
    axes[1].set_title('Reward')
    axes[1].set_xlabel('weeks since start of epidemic')
    axes[1].set_ylabel('reward r(t)')
    return fig

class logging(object):
    def __init__(self, log_file='./virl_exec.log'):
        self.log_file = log_file

    def __call__(self, info, verbose=True):
        with open(self.log_file, 'a')as f:
            if not isinstance(info, str):
                info = str(info)
            f.write(info + '\n')
        if verbose:
            print(info)


def draw_total_rewards_with_problem_id(results, key=''):
    fig, axes = plt.subplots(1, 2, figsize=(8 * 2, 6))
    noisy_True = results[results['noisy']==True].reset_index()
    noisy_False = results[results['noisy']==False].reset_index()
    print(noisy_True,noisy_False)
    if len(noisy_True) == 0 or len(noisy_False) == 0:
        return 
    if len(noisy_True) > 10:
        for i in range(4):
            axes[0].plot(np.arange(10), list(noisy_False[noisy_False['action']==i]['Total_rewards']),label='action={}'.format(i))
            axes[1].plot(np.arange(10), list(noisy_False[noisy_True['action']==i]['Total_rewards']),label='action={}'.format(i))
    else:
        axes[0].plot(np.arange(10), list(noisy_False['Total_rewards']))
        axes[1].plot(np.arange(10), list(noisy_False['Total_rewards']))

    axes[0].set_title('{} noisy=False'.format(key))
    axes[0].set_xlabel('problem id')
    axes[0].set_ylabel('total rewards')
    axes[0].legend()
    axes[1].set_title('{} noisy=True'.format(key))
    axes[1].set_xlabel('problem id')
    axes[1].set_ylabel('total rewards')
    axes[1].legend()
    plt.savefig(dpi=300, fname='./results/{}.jpg'.format(key))
    plt.close()

if __name__ == '__main__':
    pass