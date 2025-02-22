import gymnasium as gym
from pathlib import Path
import csv
import time
from pathlib import Path
import numpy as np


def load_policy(filename: Path) -> dict:
    with open(filename, 'r', newline='\n') as f:
        csv_reader = csv.reader(f, delimiter=',')

        policy = {}
        for line in csv_reader:
            if line[0].startswith('#'):
                continue

            state_str = line[0]
            state_data = state_str.split(" ")

            state = (float(state_data[0]), float(state_data[1]), float(state_data[2]), float(state_data[3]))
            action = int(line[1])
            policy[state] = action
        return policy

#def load_policy_str(filename: Path) -> dict:



def find_action(game_state: list[float], policy: dict) -> tuple[int, list]:
    max_dist = float("inf")
    state_selected = None
    for state in policy:
        state_np = np.array(state)

        l2_norm = np.linalg.norm(state_np - game_state, ord=2)

        if l2_norm < max_dist:
            max_dist = l2_norm
            state_selected = state

    return policy[state_selected], state_selected


def find_action_2(game_state: list[float], policy: dict) -> tuple[int, list]:

    game_state = tuple(game_state)
    return policy[game_state], game_state


if __name__ == '__main__':

    # load the policy
    policy_path = Path('/home/alex/qi3/cuberl/build_cpu/examples/rl/rl_example_13/experiments/2/policy.csv')
    policy = load_policy(policy_path)

    # create the environment
    version = 'v1'
    env_tag = f"CartPole-{version}"

    env = gym.make(id=env_tag, render_mode="human")

    state, _ = env.reset(seed=42)
    done = False
    total_reward = 0

    step_counter = 0
    step_reward = 0.0
    while step_counter < len(policy):

        state_tuple = state

        print(f'At state {state}: ')
        action, state_selected = find_action(state_tuple, policy)

        print(f'Matched state {state_selected} taking action {action}')

        observation, reward, done, truncated, info = env.step(action)

        step_reward += reward

        env.render()
        time.sleep(0.1)
        if truncated:
            done = True

        if done:
            print(f"Environment triggered done...")
            print(f"Reward achieved... {step_reward}")
            print(f"Resetting...")
            state, _ = env.reset(seed=42)
            step_reward = 0.0
            step_counter = len(policy) + 1
        else:
            total_reward += reward
            state = observation
        step_counter += 1

    print(f'Total reward {total_reward}')
