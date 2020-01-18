class State(object):
    """
    Sate defines a state in the Easy21 Game. A state is defined by the dealer's first card
    and the sum of the cards of the player.
    """
    def __init__(self, dealer_card, player_card_sum, is_terminal):
        """
        Initiates a new state of Easy21 game.

        Arguments:
            dealer_card (Card): Represents the black card the dealer drew at the begining of the game.
            player_card_sum (int) : Represents the sum of player cards.
            is_terminal (bool): Represents wheter the state is terminal or not. 
        """
        self.dealer_card = dealer_card 
        self.player_card_sum = player_card_sum
        self.terminal = is_terminal # defines wheter the state is terminal or not.
    
    def is_terminal(self):
        """Returns wheter the state is terminal or not.
        """
        return self.terminal