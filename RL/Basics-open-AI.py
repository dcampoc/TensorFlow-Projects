'''
Getting information about the simulation at each time instant
'''

# env.step() function returns useful objects that can be used in the learning process
# The step function can return: "observations", "reward", "done": Boolean indicating if the environment needs to be reset
# "Info": diagnostic information used for debugging

import gym

# Make the environment, replace this string with any
# from the docs. (Some environments have dependencies)
env = gym.make('CartPole-v0')

# Reset the environment to default beginning
# Default observation variable
print("Initial Observation")
observation = env.reset()
print(observation)

print('\n')
for _ in range(2):


    # Random Action
    action = env.action_space.sample()

    # Get the 4 observation values discussed
    observation, reward, done, info = env.step(action)

    print("Performed One Random Action")
    print('\n')
    print('observation')
    print(observation)
    print('\n')

    print('reward')
    print(reward)
    print('\n')

    print('done')
    print(done)
    print('\n')

    print('info')
    print(info)
    print('\n')
