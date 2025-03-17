import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("plots", exist_ok=True)


def plot_convergence(convergence_data, time_horizon, file_name="convergence_plot.png"):
    """
    Draw the Q-value convergence curve and save the image.

    Parameters:
    - convergence_data (list): Convergence data, the average Q-value at each time step.
    - time_horizon (int): Time horizon.
    - file_name (str): The filename to save the image.
    """
    plt.figure(figsize=(10, 6))
    for t in range(time_horizon):
        plt.plot(convergence_data[t], label=f"Time Step {t + 1}")
    plt.xlabel("Episodes")
    plt.ylabel("Average Q-Value")
    plt.title("Q-Value Convergence Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/{file_name}")
    plt.close()
    print(f"Q-value convergence curve plots/{file_name}")


def plot_policy(q_function, time_horizon, file_name="policy_plot.png"):
    """
    Plot the policy change graph and save the image.

    Parameters:
    - q_function (list): Q-function.
    - time_horizon (int): Time horizon.
    - file_name (str): The filename to save the image.
    """
    plt.figure(figsize=(10, 6))
    for t in range(time_horizon):
        wealth_values = sorted(q_function[t].keys())
        actions = [get_max_q_value(q_function[t][wealth])[1] for wealth in wealth_values]
        plt.plot(wealth_values, actions, label=f"Time Step {t + 1}")
    plt.xlabel("Wealth")
    plt.ylabel("Optimal Action")
    plt.title("Optimal Policy Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/{file_name}")
    plt.close()
    print(f"Policy change graph plots/{file_name}")


def plot_wealth_distribution(final_wealth_values, file_name="wealth_distribution.png"):
    """
    Plot the final wealth distribution and save the image.

    Parameters:
    - final_wealth_values (list): A list of final wealth values.
    - file_name (str): The filename to save the image.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(final_wealth_values, bins=50, edgecolor="black")
    plt.xlabel("Final Wealth")
    plt.ylabel("Frequency")
    plt.title("Final Wealth Distribution")
    plt.grid(True)
    plt.savefig(f"plots/{file_name}")
    plt.close()
    print(f"Final wealth distribution plots/{file_name}")


def get_max_q_value(q_values):
    """
    Find the maximum value and its corresponding action from a Q-value dictionary.

    Parameters:
    - q_values (dict): Q-value dictionary.

    Returns:
    - max_val (float): The maximum Q-value.
    - best_action (any): The corresponding action.
    """
    max_val = -np.inf
    best_action = None
    for act, val in q_values.items():
        if val > max_val:
            max_val = val
            best_action = act
    return max_val, best_action
