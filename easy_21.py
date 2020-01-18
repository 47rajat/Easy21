from actions import Action
from card import Card
from state import State
from colors import Color
from constants import *

class Easy21(object):
    """
    Easy21 represents the environment for playing the Easy21 Game.
    """
    def initialize_game(self):
        """
        Initiate a new Easy21 Game.
        """
        dealer_card = Card(color=Color.BLACK)
        player_card = Card(color=Color.BLACK)
        return State(dealer_card, player_card.get_abs_num_value(), False)

    def step(self, state, action):
        """
        Executes one step in the Easy21 environment using the provdied state and action. It returns
        the next sampled state and reward.

        Arguments:
            state (State): The current state of the game.
            action (Action): The action the agent has taken in the current state.
        
        Returns:
            next_state (State): The next state to which the game transitions.
        """
        if action == Action.HIT:
            return self._execute_hit_action(state)
        elif action == Action.STICK:
            return self._execute_stick_action(state)
        else:
            raise Exception('Invalid ACTION requested, action: ', action)

    def _execute_hit_action(self, state):
        """
        Executes the hit action for the player.

        Arguments:
            state (State): The current state of the game.
        
        Returns:
            next_state (State): The next state to which the game transition.
        """
        # get next card for the player.
        card = Card()

        # compute player card sum.
        new_player_card_sum = state.player_card_sum + card.get_num_value()

        # check wheter the next state is terminal
        if new_player_card_sum > MAX_PLAYER_CARD_SUM or new_player_card_sum < MIN_PLAYER_CARD_SUM:
            return State(state.dealer_card, new_player_card_sum, True), -1
        
        return State(state.dealer_card, new_player_card_sum, False), 0

    def _execute_stick_action(self, state):
        """
        Executes the stick action for the player.

        Arguments:
            state (State): The current state of the game.
        
        Returns:
            next_state (State): The next state to which the game transition.
        """
        dealer_card_sum = state.dealer_card.get_num_value()

        # dealer keeps hitting till it's card sum is less than 17.
        while dealer_card_sum > MIN_DEALER_CARD_SUM and dealer_card_sum < MAX_DEALER_CARD_SUM:
            # sample a new card for the dealer
            card = Card()
            dealer_card_sum += card.get_num_value()
        
        next_state = State(state.dealer_card, state.player_card_sum, True)
        if dealer_card_sum > MAX_PLAYER_CARD_SUM or dealer_card_sum < MIN_PLAYER_CARD_SUM:
            return next_state, 1
        elif dealer_card_sum > state.player_card_sum:
            return next_state, -1
        elif dealer_card_sum < state.player_card_sum:
            return next_state, 1
        else:
            return next_state, 0
