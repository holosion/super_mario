#=================================================================
#train a ppo agent to play super mario bros (goal beat the bowser,save peach)
#==================================================================

#--------------------------- 1. IMPORTS ----------------------------
import os
import time
import cv2
import numpy as np


import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor  #records rewards and lenght of each episode
from stable_baselines3.common.vec_env import DummyVecEnv  #runs N environments one after the other
from stable_baselines3.common.vec_env import SubprocVecEnv #runs N environments in parallel
from stable_baselines3.common.vec_env import VecFrameStack #stacks the last 4 frames so ai sees motion 
from stable_baselines3.common.callbacks import BaseCallback # parent class for writing our own callbacks
from stable_baselines3.common.callbacks import EvalCallback # tests the model ofen and keeps the Best one
from stable_baselines3.common.callbacks import CheckpointCallback #saves the model every x steps (safety copies)
from stable_baselines3.common.callbacks import CallbackList # lets us use several callbacks at once


#----------------------------2. SETTINGS ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #get the path of this file
TRAIN_DIR = os.path.join(BASE_DIR, "training")  #main folder for  training data
LOG_DIR = os.path.join(TRAIN_DIR, "logs") #subfolder for tensorboard  + evaluation logs
SAVE_DIR = os.path.join(TRAIN_DIR, "saved_models") #subfolder for saved models(best/final/ checkpoints)

# Curriculum: teach Mario step by step, then the whole game. Change PHASE to move on.
# Phase 1 = only level 1-1 (quick test that everything works, ~1-2 million steps)
# Phase 2 = random levels 1-1..8-4 (learns to cope with every kind of level)
# Phase 3 = the FULL game 1-1 -> 8-4 with Bowser and the princess (needs a LOT of steps)

PHASES = {
    1: {"env_id": "SuperMarioBros-1-1-v0", "steps": 2_000_000},
    2: {"env_id": "SuperMarioBros-1-1-v0", "steps": 10_000_000},
    3: {"env_id": "SuperMarioBros-1-1-v0", "steps": 30_000_000}
    
}

PHASE = 1 # Change this to switch between phases
LOAD_MODEL_PATH = None  # e.g. "training/saved_models/<run>/best_model.zip" to continue
N_ENVS = 8 # Number of parallel environments for training
USE_SUBPROC = True #  True = one process per game (faster,) False - single process
FRAME_SKIP = 4 # THE AGENT PICKS AN ACTION EVERY ONCE EVERY 4 GAME FRAMES
IMG_SIZE = 84 # THE SCREEN IS SHRUNK TO 84 BY 84 PIXELS 
N_STACK = 4 # THE NUMBER OF PAST FRAMES THE AGENT SEES AT ONCE (SO IT CAN SEE MOTION)
EVAL_FREQ = 50_000 # TEST THE MODEL EVERY 50K TRAINING STEPS
CHECKPOINT_FREQ = 100_000 # SAVE A CHECKPOINT EVERY 100K 
N_EVAL_EPISODES = 5 # HOW MANY GAMES EACH TEST USES TO SCORE THE MODEL
SEED = 42  # FIXED RANDOM SEED SO RUNS ARE REPEATABLE

#----------------------------3. ENVIRONMENT WRAPPERS ----------------------------
class OldGymToGymnasium(gym.Env):
    """gym_super_mario_bros speaks  the old gym api . stable-baselines3 2.x 
    needs GYMNASIUM. THIS Class is a translator sitting between the two."""
    
    def __init__(self, old_env):
        super().__init__()
        self.env = old_env
        old_obs = old_env.observation_space
        self.observation_space =  spaces.Box(low= old_obs.low, high=old_obs.high,shape=old_obs.shape, dtype=old_obs.dtype)
        self.action_space = spaces.Discrete(old_env.action_space.n)
        
        
    def reset(self, *, seed=None, option=None):
        super () .reset(seed=seed) #let gymnasium handle seed book keeping
        result = self._env.reset() # old api returns only the first picture, not the info dict
        obs = result[0] if isinstance(result, tuple) else result # be safe if a version returns(obs, info)
        
        
        
        



