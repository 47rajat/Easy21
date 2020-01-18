# Easy21
This repository contains my solution of the [Easy21](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf) assignment from the [UCL course on RL](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html).

# Monte-Carlo Controller

V*(s), episodes = 1,000  |  V*(s), episodes = 100,000
:-------------------------:|:-------------------------:
![](/results/monte_carlo_controller/value_func_episode(10000).png)  |  ![](/results/monte_carlo_controller/value_func_episode(100000).png)
V*(s), episodes = 500,000      |  V*(s), episodes = 1,000,000 
![](/results/monte_carlo_controller/value_func_episode(500000).png)  |  ![](/results/monte_carlo_controller/value_func_episode(1000000).png)

# Sarsa(λ) Controller
 Value function (λ = 0, episodes = 10,000 )            |  Value function (λ = 1, episodes=10,000)
:-------------------------:|:-------------------------:
![](/results/sarsa_controller/value_func_λ(0.0)_episode(10000).png)  |  ![](/results/sarsa_controller/value_func_λ(1.0)_episode(10000).png)
MSE per episode            |  MSE per λ, episodes = 10,000 
![](/results/sarsa_controller/mean_squared_error_vs_episode.png)  |  ![](/results/sarsa_controller/mean_squared_error_vs_lambda.png)

# LFA Controller
 Value function (λ = 0, episodes = 10,000 )            |  Value function (λ = 1, episodes=10,000)
:-------------------------:|:-------------------------:
![](/results/lfa_controller/value_func_λ(0.0)_episode(10000).png)  |  ![](/results/lfa_controller/value_func_λ(1.0)_episode(10000).png)
MSE per episode            |  MSE per λ, episodes = 10,000 
![](/results/lfa_controller/mean_squared_error_vs_episode.png)  |  ![](/results/lfa_controller/mean_squared_error_vs_lambda.png)