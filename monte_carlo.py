from controller import Easy21Controller
import numpy as np
from actions import Action
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from constants import *
import os

class MonteCarloController(Easy21Controller):
    def update_policy(self, state_actions, total_reward):
        """
        Updates the policy using the provided state_actions history and total reward for the episode.

        Arguments:
            state_actions (list of (State, Action)): List of state action pair containing the entire history 
                                                    of the episode
            total_reward (float): total reward obtained during the course of the episode.
        """
        for state, action in state_actions:
             # get index value.
             dealer_card_value = state.dealer_card.get_num_value()
             player_card_sum = state.player_card_sum

             # update state action count
             self.state_action_count[dealer_card_value][player_card_sum][action] += 1
             count = self.state_action_count[dealer_card_value][player_card_sum][action]

             # update action value function
             curr_val = self.state_action_value[dealer_card_value][player_card_sum][action]
             new_val = curr_val + ((total_reward - curr_val)/count)
             self.state_action_value[dealer_card_value][player_card_sum][action] = new_val
    
    def plot_value_function(self, num_episode):
        """
        Plots the value function and saves them as .png for the given episode number.

        Arguments:
            num_episode (int): The current episode number of the Easy21 game.
        """
        print('')
        print(f'Plotting monte-carlo result for {num_episode} episodes')

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
        ax.set_title(f'V* [{num_episode} episodes]')
        #save figure
        if os.path.isdir(MONTE_CARLO_RESULT_PATH) is False:
            os.mkdir(MONTE_CARLO_RESULT_PATH)
        plt.savefig(f'{MONTE_CARLO_RESULT_PATH}/value_func_episode({num_episode}).png')
        plt.close()