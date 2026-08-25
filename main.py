#implementing a reinforcement learning for a super mario game 
#  import neccessary libraries
# import the game
import gym_super_mario_bros

#import the joy pad wrapper
from nes_py.wrappers import JoypadSpace # this is used to limit the action space of the agent

#import the simplified controls 
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT # this is used to limit the action space of the agent

env = gym_super_mario_bros.make('super-mario-bros-v0') # this is used to create the environment for the game
env = JoypadSpace(env, SIMPLE_MOVEMENT) # this is used to limit the action space of the agent

env.observation_space.shape # this is used to get the shape of the observation space
env.action_space

#create a flag f - restart the game or not

done = True
#loop thhrough each frame of the game
for step in range(100000):
    if done:
        #start a new game
        env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        env.render() # this is used to render the game
    env.close() # this is used to close the game    