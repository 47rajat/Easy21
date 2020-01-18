from controller import Easy21Controller
from constants import *
import numpy as np
import matplotlib.pyplot as plt
import os

class SarsaController(Easy21Controller):
    def __init__(self, lmbda = 0.0):
        """
        Initialize a SARSA Controller for Easy21 Game.
        
        Arguments:
            lmbda (float): Lambda parameter to be used for weighting the future returns.
        """
        # initiate base Easy21Controller class
        super().__init__()
        # initiate eligibility traces to zero for all state action values.
        self.eligibility_trace = np.zeros(STATE_ACTIONS)
        # lmbda represents the lambda parameter of the Sarsa controller.
        self.lmbda = lmbda
    
    def update_policy(self, prev_state, prev_action, curr_state, curr_action, reward):
        """
        Updates the policy using the provided state transition from previous state to current state along with
        the transition reward.

        Arguments:
            prev_state (State): The previous state in which the agent was.
            prev_action (Action): The action that was taken by the agent in previous state.
            curr_state (State): The current state in which the agent is.
            curr_action (Action): The action that will be taken by the agent in the current state.
            reward (float): The reward observed by the agent while transitioning from (previous state, previous action) to current state.
        """
        # compute indexes of previous and current state actions.
        pi, pj, pk = (prev_state.dealer_card.get_num_value(), prev_state.player_card_sum, int(prev_action))
        ci, cj, ck = (curr_state.dealer_card.get_num_value(), curr_state.player_card_sum, int(curr_action))
        
        # compute TD error for state transition.
        td_error = reward - self.state_action_value[pi][pj][pk]
        if curr_state.is_terminal() is False:
            td_error += self.state_action_value[ci][cj][ck]

        # update the eligibility trace of the previous state.
        self.eligibility_trace[pi][pj][pk] += 1

        # update the current state action count if not terminal state.
        self.state_action_count[pi][pj][pk] += 1

        # update state action values for all state and action pairs.
        for i in range(MIN_DEALER_CARD_VALUE, MAX_DEALER_CARD_VALUE + 1):
            for j in range(MIN_PLAYER_CARD_SUM, MAX_PLAYER_CARD_SUM + 1):
                for k in range(NUM_ACTIONS):
                    # ignoring updates for (state, actions) that have not been observed yet.
                    count = self.state_action_count[i][j][k]
                    if count == 0:
                        continue
                    # updating state action value and eligibility traces.
                    self.state_action_value[i][j][k] += ((td_error*self.eligibility_trace[i][j][k])/count)
                    self.eligibility_trace[i][j][k] *= self.lmbda
    def clear_eligibility_traces(self):
        """
        Clears eligibility traces after end of each episode.
        """
        self.eligibility_trace = np.zeros(STATE_ACTIONS)
        
    def compute_mean_squared_error(self, optimal_state_action_value):
        """
        Returns the mean squared error of the state action value w.r.t provided optimal state action values.

        Agruments:
            optimal_state_action_value (numpy 3d array): A numpy 3d array containing the optimal state action values
                for all state and action pairs.
        
        Returns:
            mean squared error (float): The mean sqaured error between the provided optimal state action 
                values and self computed action values. 
        """
        all_state_actions = MAX_DEALER_CARD_VALUE*MAX_PLAYER_CARD_SUM*NUM_ACTIONS
        return np.sum(np.square(optimal_state_action_value - self.state_action_value))/all_state_actions
    
    def plot_value_function(self, num_episode):
        """
        Plots the action value function and saves them as .png for the given episode number.

        Arguments:
            num_episode (int): The current episode number of the Easy21 game.
        """
        print('')
        print(f'Plotting value function for Sarsa(λ={self.lmbda})')

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
                Z[i][j] = np.max(self.state_action_value[idx][idy])
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
        if os.path.isdir(SARSA_RESULT_PATH) is False:
            os.mkdir(SARSA_RESULT_PATH)
        plt.savefig(f'{SARSA_RESULT_PATH}/value_func_λ({self.lmbda})_episode({num_episode}).png')
        plt.close()
