import gymnasium as gym


if __name__ == '__main__':

	env = gym.make('CartPole-v1')

	state = env.reset(42)
