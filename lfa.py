from constants import *
import numpy as np
from actions import Action
import matplotlib.pyplot as plt
import os

class LFAController(object):
    def __init__(self, lmbda=0.0):
        """
        Initialize a Linear Function Approximation controller for Easy21 Game.
        """
        # initiate eligibility traces to zero for all features.
        self.eligibility_trace = np.zeros(FEATURE_DIM)
        # lmbda represents the lambda parameter of the Sarsa controller.
        self.lmbda = lmbda
        # initiate weight vector to all zeros.
        self.weight = np.zeros(FEATURE_DIM)

        # set step size and epsilon
        self.step_size = 0.01
        self.epsilon = 0.05

        # set feature vector map and store (X.T * X)^-1 for updating weigths
        self._init_feature_map()
        
    def _init_feature_map(self):
        """
        Intitalizes the feature_map, which maps each state action pair to its feature
        vector representation.
        """
        self.feature_map = {}
        for d in range(MIN_DEALER_CARD_VALUE, MAX_DEALER_CARD_VALUE + 1):
            for p in range(MIN_PLAYER_CARD_SUM, MAX_PLAYER_CARD_SUM + 1):
                for a in range(NUM_ACTIONS):
                    self.feature_map[(d, p, a)] = self._compute_feature(d, p, a)

    def _compute_feature(self, d, p, a):
        """
        Computes feature vector for the provided state action values.

        Arguments:
            d (int): Dealer card value in the state.
            p (int): Player card sum in the state.
            a (Action): Action taken by the agent in the state.
        
        Returns:
            feat (np.array(FEATURE_DIM)): A numpy array representing the feature vector for the provided
                state action. 
        """
        feat = np.zeros(FEATURE_DIM)
        idx = lambda x : 12*x[0] + 2*x[1] + x[2]
        for i, db in enumerate(DEALER_BRACKETS):
            if d < db[0] or d > db[1]:
                continue
            for j, pb in enumerate(PLAYER_BRACKETS):
                if p < pb[0] or p > pb[1]:
                    continue
                feat[idx((i, j, a))][0] = 1.0
        
        return feat
    
    def get_action(self, state):
        """
        Return an action using the provided state and the state action value function following the
        epsilon-greedy approach.

        Agruments:
            state (State): The current state of the Easy21 game.
        
        Returns:
            action (Action): The action picked by the agent.
        """
        # if state is terminal return Action(0).
        if state.is_terminal():
            return Action(0)
        
        # get feature vector for both action
        feature_hit = self.get_state_action_vector(state, Action.HIT)
        feature_stick = self.get_state_action_vector(state, Action.STICK)

        # compute action value.
        value_hit = self.weight.T.dot(feature_hit)
        value_stick = self.weight.T.dot(feature_stick)

        # pick epsilon greedy action.
        if np.random.random() <= self.epsilon or value_hit == value_stick:
            return Action(np.random.choice(np.arange(2)))
        
        # return action with greater action value.
        return Action.HIT if value_hit > value_stick else Action.STICK
                        
    def update_policy(self, prev_state, prev_action, curr_state, curr_action, reward):
        """
        Updates the policy using the provided state transition from previous state to current state along with
        the transition reward.
        
        Arguments:
            prev_state (State): The last state of the Easy21 game.
            prev_action (Action): The action taken by the agent in prev_state.
            curr_state (State): The current state of the Easy21 game.
            curr_action (action): The action take by the agent in the curr_state
            reward (float): the reward received for taking the prev_action in prev_state
        """
        # get feature vector corresponding to state actions. 
        prev_state_action_vector = self.get_state_action_vector(prev_state, prev_action)
        curr_state_action_vector = self.get_state_action_vector(curr_state, curr_action)

        # compute TD error for state transition.
        td_error = reward + self.weight.T.dot(curr_state_action_vector) - self.weight.T.dot(prev_state_action_vector)

        # udpdate the eligibility trace of the previous state.
        self.eligibility_trace += prev_state_action_vector

        # update feature weights.
        self.weight += self.step_size*td_error*self.eligibility_trace
        self.eligibility_trace *= self.lmbda

    def get_state_action_vector(self, state, action):
        """
        Return feature vector corresponding to the provided state action pair.

        Arguments:
            state (State): The state of the Easy21 game.
            action (Action): Action taken by the agent in the state.
        """
        if state.is_terminal():
            return np.zeros(FEATURE_DIM)
        d, p, a = (state.dealer_card.get_num_value(), state.player_card_sum, int(action))
        if self.feature_map.get((d, p, a)) is None:
            raise ValueError('State-Action {} not present in feature map'.format((d, p, a)))
        return self.feature_map[(d, p, a)]

    def clear_eligibility_traces(self):
        """
        Clears eligibility traces after end of each episode.
        """
        self.eligibility_trace = np.zeros(FEATURE_DIM)
        
    def compute_mean_squared_error(self, optimal_state_action_value):
        """
        Returns the mean squared error of the state action value w.r.t provided optimal state action values.
        """
        all_state_actions = MAX_DEALER_CARD_VALUE*MAX_PLAYER_CARD_SUM*NUM_ACTIONS
        state_action_values = self._compute_state_action_values()
        return np.sum(np.square(optimal_state_action_value - state_action_values))/all_state_actions
    
    def _compute_state_action_values(self):
        """
        Computes state action values for all state and actions.

        Returns:
            state_action_value (numpy 3d array): A numpy 3d array containing state action values for all state
                and action pairs.
        """
        state_action_values = np.zeros(STATE_ACTIONS)
        for d in range(MIN_DEALER_CARD_VALUE, MAX_DEALER_CARD_VALUE + 1):
            for p in range(MIN_PLAYER_CARD_SUM, MAX_PLAYER_CARD_SUM + 1):
                for a in range(NUM_ACTIONS):
                    if self.feature_map.get((d, p, a)) is None:
                        raise ValueError('State-Action {} not present in feature map'.format((d, p, a)))
                    state_action_values[d][p][a] = self.weight.T.dot(self.feature_map[(d, p, a)])
        return state_action_values
    
    def plot_value_function(self, num_episode):
        """
        Plots the action value function and saves them as .png for the given episode number.

        Arguments:
            num_episode (int): The current episode number of the Easy21 game.
        """
        print('')
        print(f'Plotting value function for Sarsa(λ={self.lmbda})')

        state_action_values = self._compute_state_action_values()

        dealer_card_value = np.arange(MIN_DEALER_CARD_VALUE, MAX_DEALER_CARD_VALUE+1)
        player_card_sum = np.arange(MIN_PLAYER_CARD_SUM, MAX_PLAYER_CARD_SUM+1)

        # create grid of dealer card value and player card sum on X and Y axis respectively.
        X, Y = np.meshgrid(dealer_card_value, player_card_sum)
        # initiate z-axis as zeros to plot optimal value function.
        Z = np.zeros((21, 10))
        for i in range(21):
            for j in range(10):
                idx = X[i][j]
                idy = Y[i][j]
                Z[i][j] = np.max(state_action_values[idx][idy])
        # plot optimal value functions.
        plt.figure(figsize=FIG_SIZE)
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
        ax.set_xlabel('Dealer Showing')
        ax.set_ylabel('Player Sum')
        ax.set_zlabel('V*(s)')
        ax.set_title(f'V* [λ = {self.lmbda}, {num_episode} episodes]')
        #save figure
        if os.path.isdir(LFA_RESULT_PATH) is False:
            os.mkdir(LFA_RESULT_PATH)
        plt.savefig(f'{LFA_RESULT_PATH}/value_func_λ({self.lmbda})_episode({num_episode}).png')
        plt.close()
