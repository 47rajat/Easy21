from easy_21 import Easy21
from monte_carlo import MonteCarloController
from sarsa import SarsaController
from lfa import LFAController
from constants import *
import matplotlib.pyplot as plt

# set up environment and monte-carlo controller.
env = Easy21()
mc_controller = MonteCarloController()

print('Playing Easy 21 with Monte-Carlo controller....')
# play easy 21 using monte-carlo controller.
for e in range(NUM_MC_EPISODES):
    print(f'Episode {e+1:,}/{NUM_MC_EPISODES:,} done.', end='\r')
    state = env.initialize_game()
    episode_reward = 0
    # state_actions will store the episode history.
    state_actions = []
    
    # play game till termination.
    while state.is_terminal() == False:
        # get action from controller.
        action = mc_controller.get_action(state)

        # add state action to history.
        state_actions.append((state, action))

        # execute one step in the environment.
        state, reward = env.step(state, action)

        # update episode reward
        episode_reward += reward
    
    # update policy using episode history and reward.
    mc_controller.update_policy(state_actions, episode_reward)

    if (e+1) in PLOT_EPISODES:
        mc_controller.plot_value_function(e+1)
print('')
print('Monte-Carlo game done')
print('')

# play easy 21 using sarsa controller.
print('Playing Easy21 with SARSA controller....')

# mean_squared_error will store the mean squared error of value function after each episode.
mean_square_errors = {}

for lmbda in LAMBDA_VALUES:
    sarsa_controller = SarsaController(lmbda=lmbda)
    # initialize mean_square_error for lmbda.
    mean_square_errors[lmbda] = [0 for i in range(NUM_SARSA_EPISODES)]
    print(f'Starting game for λ = {lmbda}')
    for e in range(1, NUM_SARSA_EPISODES+1):
        print(f'Episode {e:,}/{NUM_SARSA_EPISODES:,} done.', end='\r')

        # clear eligibility traces.
        sarsa_controller.clear_eligibility_traces()

        # get initital state and action.
        prev_state = env.initialize_game()
        prev_action = sarsa_controller.get_action(prev_state)

        # play game till termination.
        while prev_state.is_terminal() == False:
            # execute one step in the environment.
            curr_state, reward = env.step(prev_state, prev_action)
            # get action for current state.
            curr_action = sarsa_controller.get_action(curr_state)

            # update policy based on prev and current state actions.
            sarsa_controller.update_policy(prev_state, prev_action, curr_state, curr_action, reward)

            # swap prev state-action with current state-action.
            prev_state, prev_action = curr_state, curr_action

        # compute mean square error of the sarsa value function with optimal (monte-carlo) value function.
        mean_square_errors[lmbda][e-1] = sarsa_controller.compute_mean_squared_error(mc_controller.state_action_value)

    # plot value function.
    sarsa_controller.plot_value_function(NUM_SARSA_EPISODES)
print('')
print('SARSA game done')
print()
# plot mean squared error.
print('Plotting mean squared error for Sarsa(λ)')
plt.figure(figsize=FIG_SIZE)
plt.style.use('ggplot')
for lmbda, errors in mean_square_errors.items():
    plt.plot(range(1, NUM_SARSA_EPISODES+1), errors, label=f'λ={lmbda}')
plt.xlabel('# episode')
plt.ylabel('Mean Squared Error')
plt.title('Mean Square Error of Sarsa(λ) value function')
plt.legend(loc='best')
plt.savefig(f'{SARSA_RESULT_PATH}/mean_squared_error_vs_episode.png')
plt.close()


# plot mean squared error per λ.
plt.figure(figsize=FIG_SIZE)
plt.style.use('ggplot')
x = [k for k in mean_square_errors.keys()]
x.sort()
y = [mean_square_errors[k][-1] for k in x]
plt.plot(x, y)
plt.xlabel('λ')
plt.ylabel('Mean Squared Error')
plt.title('Mean Square Error of Sarsa(λ) value function')
plt.savefig(f'{SARSA_RESULT_PATH}/mean_squared_error_vs_lambda.png')
plt.close()

print('Done!')
print('')

# play easy 21 using sarsa controller.
print('Playing Easy21 with LFA controller....')

# mean_squared_error will store the mean squared error of value function after each episode.
mean_square_errors = {}

for lmbda in LAMBDA_VALUES:
    lfa_controller = LFAController(lmbda=lmbda)
    # initialize mean_square_error for lmbda.
    mean_square_errors[lmbda] = [0 for i in range(NUM_LFA_EPISODES)]
    print(f'Starting game for λ = {lmbda}')
    for e in range(1, NUM_LFA_EPISODES+1):
        print(f'Episode {e:,}/{NUM_LFA_EPISODES:,} done.', end='\r')

        # clear eligibility traces.
        lfa_controller.clear_eligibility_traces()

        # get initital state and action.
        prev_state = env.initialize_game()
        prev_action = lfa_controller.get_action(prev_state)

        # play game till termination.
        while prev_state.is_terminal() == False:
            # execute one step in the environment.
            curr_state, reward = env.step(prev_state, prev_action)
            # get action for current state.
            curr_action = lfa_controller.get_action(curr_state)

            # update policy based on prev and current state actions.
            lfa_controller.update_policy(prev_state, prev_action, curr_state, curr_action, reward)

            # swap prev state-action with current state-action.
            prev_state, prev_action = curr_state, curr_action

        # compute mean square error of the lfa  value function with optimal (monte-carlo) value function.
        mean_square_errors[lmbda][e-1] = lfa_controller.compute_mean_squared_error(mc_controller.state_action_value)

    # plot value function.
    lfa_controller.plot_value_function(NUM_LFA_EPISODES)
print('')
print('LFA game done')
print()
# plot mean squared error per episode.
print('Plotting mean squared error for LFA Sarsa(λ)')
plt.figure(figsize=FIG_SIZE)
plt.style.use('ggplot')
for lmbda, errors in mean_square_errors.items():
    plt.plot(range(1, NUM_LFA_EPISODES+1), errors, label=f'λ={lmbda}')
plt.xlabel('# episode')
plt.ylabel('Mean Squared Error')
plt.title('Mean Square Error of Sarsa(λ) value function')
plt.legend(loc='best')
plt.savefig(f'{LFA_RESULT_PATH}/mean_squared_error_vs_episode.png')
plt.close()

# plot mean squared error per λ.
plt.figure(figsize=FIG_SIZE)
plt.style.use('ggplot')
x = [k for k in mean_square_errors.keys()]
x.sort()
y = [mean_square_errors[k][-1] for k in x]
plt.plot(x, y)
plt.xlabel('λ')
plt.ylabel('Mean Squared Error')
plt.title('Mean Square Error of Sarsa(λ) value function')
plt.savefig(f'{LFA_RESULT_PATH}/mean_squared_error_vs_lambda.png')
plt.close()

print('Done!')
