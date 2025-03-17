import numpy as np
from TemporalDifference_QLearning import *
from Evaluation_Metrics import plot_convergence, plot_policy, plot_wealth_distribution

if __name__ == '__main__':
    # Initialize parameters
    time_horizon = 10
    initial_wealth = 10000

    # Asset parameters
    risky_return_up = 0.06
    risky_return_down = -0.02
    prob_up = 0.5
    risk_free_rate = 0.02

    # Define action space
    num_actions = 10
    action_options = np.linspace(0, 1, num_actions)
    action_options = np.round(action_options, 1)

    # Q-learning parameters
    episodes = 10000
    learning_rate = 0.01
    discount_factor = 0.1
    exploration_rate = 1

    # Initialize Q-function
    q_function = initialize_q_function(time_horizon, initial_wealth, risky_return_up, risky_return_down, risk_free_rate,
                                       action_options)

    # Train Q-learning and get convergence data
    q_function, convergence_data = train_q_learning(
        q_function, time_horizon, episodes, learning_rate, discount_factor, exploration_rate,
        num_actions, action_options, initial_wealth, risky_return_up, risky_return_down,
        prob_up, risk_free_rate, track_convergence=True  # Ensure convergence data is returned
    )

    # Test optimal policy for 10 random traces
    final_wealth_values = []
    for _ in range(10):
        wealth = initial_wealth
        print(f'Trace {_ + 1}:')
        for step in range(time_horizon):
            _, action = get_max_q_value(q_function[step][wealth])
            print(f'Time = {step}, Wealth = {wealth}, Action = {action}')
            if np.random.rand() < prob_up:
                wealth = int(wealth + wealth * action * risky_return_up + wealth * (1 - action) * risk_free_rate)
            else:
                wealth = int(wealth + wealth * action * risky_return_down + wealth * (1 - action) * risk_free_rate)
        final_wealth_values.append(wealth)  # Record final wealth

    # Generate and save plots
    plot_convergence(convergence_data, time_horizon=time_horizon, file_name="convergence_plot.png")
    plot_policy(q_function, time_horizon=time_horizon, file_name="policy_plot.png")
    plot_wealth_distribution(final_wealth_values, file_name="wealth_distribution.png")