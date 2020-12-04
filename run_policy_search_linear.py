import numpy as np
import pickle, os
import sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
import sklearn.pipeline
import virl


class RbfFunctionApproximator():
    """
    Q(s,a) function approximator. 

    it uses a specific form for Q(s,a) where seperate functions are fitteted for each 
    action (i.e. four Q_a(s) individual functions)

    We could have concatenated the feature maps with the action TODO TASK?

    """
 
    def __init__(self, env, eta0= 0.01, learning_rate= "constant"):
        #
        # Args:
        #   eta0: learning rate (initial), default 0.01
        #   learning_rate: the rule used to control the learning rate;
        #   see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html for details
        #        
        # We create a seperate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up and understand.
        #
        #
        self.eta0=eta0
        self.learning_rate=learning_rate
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler().fit(observation_examples)
        self.feature_transformer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ]).fit(observation_examples)
        self.models = []
        for _ in range(env.action_space.n):

            # You may want to inspect the SGDRegressor to fully understand what is going on
            # ... there are several interesting parameters you may want to change/tune.
            model = SGDRegressor(learning_rate=learning_rate, tol=1e-5, max_iter=1e5, eta0=eta0)
            
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)
    
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        s_scaled = self.scaler.transform([state])
        s_transformed = self.feature_transformer.transform(s_scaled)
        return s_transformed[0]
    
    def predict(self, s, a=None):
        """
        Makes Q(s,a) function predictions.
        
        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for
            
        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.
            
        """
        features = self.featurize_state(s)
        if a==None:
            return np.array([m.predict([features])[0] for m in self.models])
        else:            
            return self.models[a].predict([features])[0]
    
    def update(self, s, a, td_target):
        """
        Updates the approximator's parameters (i.e. the weights) for a given state and action towards
        the target y (which is the TD target).
        """
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [td_target]) # recall that we have a seperate funciton for each a 
        


from utils import (
    q_learning,
    exec_policy,
    get_fig,
    plt
)


if __name__ == '__main__':
    env = virl.Epidemic(stochastic=False, noisy=False)

    rbf_file = './rbf.pkl'
    if os.path.exists(rbf_file):
        with open(rbf_file, 'rb') as f:
            rbf_func = pickle.load(f)
        print('form file load RBF success.')
    else:
        rbf_func = RbfFunctionApproximator(env)
        # training
        states = q_learning(env, rbf_func, 1500, epsilon=0.05)
        # save the approximate function
        with open(rbf_file, 'wb')as f:
            pickle.dump(rbf_func, f)
    # make dir
    if not os.path.exists('./results/RBF'):
        os.mkdir('./results/RBF')
    for i in range(10):
        id = i
        for tf in range(2):
            env = virl.Epidemic(problem_id=id, noisy=tf)
            states, rewards, actions= exec_policy(env, rbf_func, verbose=False)
            fig = get_fig(states, rewards)
            if tf:
                tf = 'True'
            else:
                tf = 'False'
            plt.savefig(dpi=300, fname= './results/RBF/problem_id={}_noisy={}.jpg'.format(id, tf))
            print("\tproblem_id={} noisy={} Total rewards:{:.4f}".format(id, tf, sum(rewards)))
            plt.close()
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(10):
    env = virl.Epidemic(stochastic=True)
    states, rewards, actions= exec_policy(env, rbf_func, verbose=False)
    ax.plot(np.array(states)[:,1], label=f'draw {i}')
ax.set_xlabel('weeks since start of epidemic')
ax.set_ylabel('Number of Infectious persons')
ax.set_title('Simulation of 10 stochastic episodes with RBF policy')
ax.legend()
plt.savefig(dpi=300, fname='./results/RBF/stochastic.png')
plt.close()