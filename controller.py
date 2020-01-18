import numpy as np
from constants import *
from actions import Action

class Easy21Controller(object):
    """
    Easy21Controller defines the base controller class for Easy21 Game.
    """
    def __init__(self):
        """
        Initialize a controller for Easy21 Game.
        """
        # intialize count for number of times a state has been encountered. This will be used to set epsilon
        # for epsilon greedy exploration.
        self.state_count = np.zeros(STATES)

        # intialize action value function as 3D array with all zeros. The dimension of the action value
        # function will be (dealer_card_value, palyer_card_sum, num_actions). There are two actions HIT or
        # STICK.
        self.state_action_value = np.zeros(STATE_ACTIONS)

        # intialize count for the number of time a station action pair has been encountered. This will be
        # used to update the action value function at the end of each episode.
        self.state_action_count = np.zeros(STATE_ACTIONS)

        # n_0 will be used in computation of epsilon.
        self.n_0 = EPSILON_0
    
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

        # update state count
        i = state.dealer_card.get_num_value()
        j = state.player_card_sum
        self.state_count[i][j] += 1
        # compute epsilon
        epsilon = self.n_0/(self.n_0 + self.state_count[i][j])

        # pick epsilon greedy action.
        if np.random.random() <= epsilon or self.state_action_value[i][j][0] == self.state_action_value[i][j][1]:
            return Action(np.random.choice(np.arange(NUM_ACTIONS)))
        
        # return action with greater action value.
        return Action(np.argmax(self.state_action_value[i][j]))

    def plot_value_function(self, num_episode):
        raise NotImplementedError("plot_value_function not implemented for controller")