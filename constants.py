# State-Action constants
STATES = (11, 22)
STATE_ACTIONS = (11, 22, 2)
NUM_ACTIONS = 2

MIN_DEALER_CARD_VALUE = 1
MAX_DEALER_CARD_VALUE = 10
MIN_DEALER_CARD_SUM = 0
MAX_DEALER_CARD_SUM = 17

MIN_PLAYER_CARD_SUM = 1
MAX_PLAYER_CARD_SUM = 21

# EPISILON_0 is used for the computation of epsilon for epsilon-greedy exploration.
EPSILON_0 = 100

# Card constants
NUM_CARD_TYPE = 2
BLACK_PROBABILITY = 2/3
RED_PROBABILITY = 1/3

# LFA controller constants
X_DIM = (NUM_ACTIONS*MAX_DEALER_CARD_VALUE*MAX_PLAYER_CARD_SUM, 36)
FEATURE_DIM = (36, 1)
DEALER_BRACKETS = [(1, 4), (4, 7), (7, 10)]
PLAYER_BRACKETS = [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)]

# Experiment Constants
NUM_MC_EPISODES = 1000000
NUM_SARSA_EPISODES = 10000
NUM_LFA_EPISODES = 10000

FIG_SIZE = (8,6)

PLOT_EPISODES = [1000, 10000, 100000, 500000, 1000000]

LAMBDA_VALUES = [i/10 for i in range(0, 11)]

MONTE_CARLO_RESULT_PATH = 'results/monte_carlo_controller'
SARSA_RESULT_PATH = 'results/sarsa_controller'
LFA_RESULT_PATH = 'results/lfa_controller'