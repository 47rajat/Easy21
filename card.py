import numpy as np
from constants import *
from colors import Color

class Card(object):
    """
    Card represents a card being used to playe the Easy21 game. A card has two attributes: number value
    (between 1 to 10) and a color (red or black).
    """
    def __init__(self, color=None):
        """
        Intitalizates a new card.
        """
        self._num_value = np.random.choice(np.arange(1, 11))
        
        # set color if provided, if not choose randomly.
        if color == None:
            color = self._get_color()
        self._color = color
    
    def _get_color(self):
        """
        Returns a card color chosen randomly
        """
        color =  np.random.choice(np.arange(NUM_CARD_TYPE), 
                            p=np.array([BLACK_PROBABILITY, RED_PROBABILITY]))
        return Color(color)
    
    def get_num_value(self):
        """
        Return the number value of the card depending on the color: +ve if black, -ve if red.
        """
        if self._color == Color.RED:
            return -1*self._num_value
        return self._num_value
    
    def get_abs_num_value(self):
        """
        Returns the number value of the card irrespective of the color.
        """
        return self._num_value

    def get_color(self):
        """
        Return the card color.
        """
        return self._color