import gym

# Make the environment, replace this string with any
# from the docs. (Some environments have dependencies)
env = gym.make('CartPole-v0')

# Reset the environment to default beginning (place the vehicle atthe start position)
env.reset()

# The simulation goes 200 steps chosing random actions
total_steps = 200
for i in range(200):
    if i%20 == 0:
        print(f'step {i} out of {total_steps}')
    
    # Render the environment (visualize it)
    env.render()

    # The car makes random actions 
    env.step(env.action_space.sample())

