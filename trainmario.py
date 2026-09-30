# =====================================================================================
# trainmario.py  -  Train a PPO agent to play Super Mario Bros (goal: beat Bowser, save Peach)
# Run it with:   python trainmario.py
# =====================================================================================

# ------------------------------- 1. IMPORTS ------------------------------------------
import os                                       # 'os' = operating-system tools (make folders, join paths)
import time                                     # 'time' = lets us read the clock (used to name each run)

import cv2                                      # OpenCV = image library (we use it to shrink/grey the screen)
import numpy as np                              # NumPy = fast arrays (images are NumPy arrays)
import gymnasium as gym                         # Gymnasium = the maintained version of OpenAI Gym (RL env API)
from gymnasium import spaces                    # 'spaces' describe the shape/type of observations and actions

import gym_super_mario_bros                     # the Super Mario Bros NES game packaged as an RL environment
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT   # a short list of useful button combos (7 actions)
from nes_py.wrappers import JoypadSpace         # restricts the NES controller to a chosen list of button combos

from stable_baselines3 import PPO               # PPO = Proximal Policy Optimisation, our learning algorithm
from stable_baselines3.common.monitor import Monitor              # records reward + length of each episode
from stable_baselines3.common.vec_env import DummyVecEnv          # runs N envs one after another (1 process)
from stable_baselines3.common.vec_env import SubprocVecEnv        # runs N envs in parallel (N processes)
from stable_baselines3.common.vec_env import VecFrameStack        # stacks the last 4 frames so AI sees motion
from stable_baselines3.common.callbacks import BaseCallback       # parent class for writing our own callback
from stable_baselines3.common.callbacks import EvalCallback       # tests the model often and keeps the BEST one
from stable_baselines3.common.callbacks import CheckpointCallback # saves the model every X steps (safety copies)
from stable_baselines3.common.callbacks import CallbackList       # lets us use several callbacks together

# ------------------------------- 2. SETTINGS -----------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # folder that contains this script
TRAIN_DIR = os.environ.get("MARIO_TRAIN_DIR", os.path.join(BASE_DIR, "training"))
LOG_DIR = os.path.join(TRAIN_DIR, "logs")               # subfolder for TensorBoard + evaluation logs
SAVE_DIR = os.path.join(TRAIN_DIR, "saved_models")      # subfolder for saved models (best/final/checkpoints)

# Curriculum: teach Mario step by step, then the whole game. Change PHASE to move on.
#   Phase 1 = only level 1-1      (quick test that everything works, ~1-2 million steps)
#   Phase 2 = random levels 1-1..8-4 (learns to cope with every kind of level)
#   Phase 3 = the FULL game 1-1 -> 8-4 with Bowser and the princess (needs a LOT of steps)
PHASES = {
    1: {"env_id": "SuperMarioBros-1-1-v0",          "steps": 2_000_000},    # env name + training length
    2: {"env_id": "SuperMarioBrosRandomStages-v0",  "steps": 10_000_000},
    3: {"env_id": "SuperMarioBros-v0",              "steps": 30_000_000},
}
PHASE = 1                                       # <-- which phase to train right now (1, 2 or 3)
LOAD_MODEL_PATH = os.environ.get("MARIO_LOAD_MODEL_PATH") or None

N_ENVS = 8                                      # how many Mario games run at the same time (use 4 if low RAM)
USE_SUBPROC = True                              # True = one process per game (faster), False = single process
FRAME_SKIP = 4                                  # the agent picks an action once every 4 game frames
IMG_SIZE = 84                                   # screen is shrunk to 84 x 84 pixels
N_STACK = 4                                     # number of past frames the agent sees at once
EVAL_FREQ = 50_000                              # test the model every 50k training steps (all envs combined)
CHECKPOINT_FREQ = 100_000                       # save a safety copy every 100k steps
N_EVAL_EPISODES = 5                             # how many games each test uses to score the model
SEED = 42                                       # fixed random seed so runs are more repeatable

# NumPy 2 no longer silently casts the intermediate value 256 back to uint8.
# The upstream Mario environment multiplies a uint8 RAM byte by 256 while
# calculating Mario's world x-position, which raises during environment reset.
# Convert the bytes to Python ints first, preserving the original calculation.
SuperMarioBrosEnv._x_position = property(
    lambda self: int(self.ram[0x6D]) * 0x100 + int(self.ram[0x86])
)
SuperMarioBrosEnv._x_position_screen = property(
    lambda self: (int(self.ram[0x86]) - int(self.ram[0x071C])) % 256
)
SuperMarioBrosEnv._y_position = property(
    lambda self: (255 + (255 - int(self.ram[0x03B8]))
                  if int(self.ram[0x00B5]) < 1
                  else 255 - int(self.ram[0x03B8]))
)


# ------------------------------- 3. ENVIRONMENT WRAPPERS -----------------------------
class OldGymToGymnasium(gym.Env):
    """gym_super_mario_bros speaks the OLD 'gym' API. Stable-Baselines3 2.x needs GYMNASIUM.
    This class is a translator sitting between the two."""

    def __init__(self, old_env):                                  # 'old_env' = the env made by gym_super_mario_bros
        super().__init__()                                        # initialise the Gymnasium parent class
        self._env = old_env                                       # keep the old environment inside this object
        old_obs = old_env.observation_space                       # old-style description of the screen
        self.observation_space = spaces.Box(                      # rebuild it as a Gymnasium 'Box' space
            low=old_obs.low, high=old_obs.high,                   # smallest / largest pixel values (0 and 255)
            shape=old_obs.shape, dtype=old_obs.dtype)             # (240, 256, 3) uint8 RGB picture
        self.action_space = spaces.Discrete(old_env.action_space.n)   # 'Discrete(7)' = choose 1 of 7 actions

    def reset(self, *, seed=None, options=None):                  # start a new game
        super().reset(seed=seed)                                  # let Gymnasium handle seeding bookkeeping
        result = self._env.reset()                                # old API returns only the first picture
        obs = result[0] if isinstance(result, tuple) else result  # be safe if a version returns (obs, info)
        return obs, {}                                            # new API returns (observation, info dict)

    def step(self, action):                                       # play one step with the chosen action
        result = self._env.step(int(action))                      # old API returns 4 values (or 5 in new builds)
        if len(result) == 5:                                      # already new style?
            obs, reward, terminated, truncated, info = result     # then just unpack it
        else:                                                     # old style: obs, reward, done, info
            obs, reward, done, info = result                      # unpack the 4 values
            terminated, truncated = bool(done), False             # 'done' becomes 'terminated'; nothing truncated
        return obs, reward, terminated, truncated, info           # new API always returns 5 values

    def render_human(self):                                       # small helper used by evaluatemario.py
        self._env.render_mode = "human"                           # current nes_py chooses render mode on the env
        self._env.render()                                         # opens/updates the game window

    def close(self):                                              # free the emulator when finished
        self._env.close()                                         # close the underlying NES emulator


class SkipFrame(gym.Wrapper):
    """Repeat the same action for 'skip' frames and add up the rewards (faster + easier learning)."""

    def __init__(self, env, skip):                                # 'env' = the env we wrap, 'skip' = frames
        super().__init__(env)                                     # remember the wrapped env
        self._skip = skip                                         # store how many frames to repeat

    def step(self, action):                                       # one agent step = several game frames
        total_reward = 0.0                                        # running sum of rewards
        terminated = truncated = False                            # flags that tell if the episode ended
        for _ in range(self._skip):                               # repeat the action 'skip' times
            obs, reward, terminated, truncated, info = self.env.step(action)   # play one game frame
            total_reward += reward                                # add up this frame's reward
            if terminated or truncated:                           # Mario died / level ended -> stop repeating
                break                                             # leave the loop early
        return obs, total_reward, terminated, truncated, info     # return the last picture and the summed reward


class MarioRewardShaper(gym.Wrapper):
    """Scales rewards and adds bonuses for the flag and for beating the final castle (8-4)."""

    def __init__(self, env, reward_scale=0.1, flag_bonus=5.0, game_bonus=50.0):
        super().__init__(env)                                     # remember the wrapped env
        self._scale = reward_scale                                # multiply raw reward by this (keeps numbers small)
        self._flag_bonus = flag_bonus                             # extra reward for finishing a level
        self._game_bonus = game_bonus                             # huge reward for beating the whole game
        self._prev_flag = False                                   # was the flag already touched last step?

    def reset(self, **kwargs):                                    # start of every new episode
        self._prev_flag = False                                   # forget the flag state
        return self.env.reset(**kwargs)                           # reset the wrapped env

    def step(self, action):                                       # one step of the game
        obs, reward, terminated, truncated, info = self.env.step(action)   # ask the wrapped env to play
        reward = reward * self._scale                             # shrink the reward (raw values are large)
        flag_now = bool(info.get("flag_get", False))              # True when the flag/axe was reached
        if flag_now and not self._prev_flag:                      # only pay the bonus the FIRST time it turns True
            reward += self._flag_bonus                            # level-clear bonus
        self._prev_flag = flag_now                                # remember for the next step
        info["game_beaten"] = False                               # by default the game is not beaten yet
        if flag_now and info.get("world") == 8 and info.get("stage") == 4:   # reached the end of 8-4
            reward += self._game_bonus                            # Bowser defeated -> Princess saved -> BIG bonus
            terminated = True                                     # end the episode: mission accomplished
            info["game_beaten"] = True                            # tell the evaluation script we won
        return obs, reward, terminated, truncated, info           # give everything back to the caller


class GrayResize(gym.ObservationWrapper):
    """Turn the colour 240x256 screen into a small 84x84 grey picture (much cheaper for the network)."""

    def __init__(self, env, size):                                # 'size' = width and height of the new image
        super().__init__(env)                                     # remember the wrapped env
        self._size = size                                         # store the target size
        self.observation_space = spaces.Box(                      # declare the NEW observation shape
            low=0, high=255, shape=(size, size, 1), dtype=np.uint8)   # 84 x 84 x 1 channel, values 0..255

    def observation(self, obs):                                   # called on every picture the game returns
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)              # colour -> grey (3 channels -> 1 channel)
        small = cv2.resize(gray, (self._size, self._size),        # shrink to size x size pixels
                           interpolation=cv2.INTER_AREA)          # INTER_AREA = best method for shrinking
        return small[:, :, None]                                  # add the channel axis: (84, 84) -> (84, 84, 1)


def make_mario_env(env_id, rank=0):
    """Returns a function that builds ONE fully-wrapped Mario environment (needed by the VecEnv classes)."""

    def _init():                                                  # the builder function
        # Its bundled nes_py currently exposes Gymnasium spaces while this
        # registration still uses legacy Gym's checker; our adapter below
        # handles the API boundary, so skip the incompatible legacy checker.
        raw = gym_super_mario_bros.make(
            env_id, disable_env_checker=True
        ).unwrapped  # remove legacy Gym wrappers; nes_py itself uses Gymnasium
        raw = JoypadSpace(raw, SIMPLE_MOVEMENT)                   # allow only 7 button combos (right, jump, ...)
        env = OldGymToGymnasium(raw)                              # translate old gym -> gymnasium
        env = SkipFrame(env, skip=FRAME_SKIP)                     # repeat each action for 4 frames
        env = MarioRewardShaper(env)                              # scale rewards, add flag / final bonuses
        env = GrayResize(env, size=IMG_SIZE)                      # grey + 84x84 pictures
        env = Monitor(env)                                        # record episode reward and length for logging
        return env                                                # hand the finished environment back

    return _init                                                  # return the builder (not the env itself)


def build_vec_env(env_id, n_envs, use_subproc):
    """Creates n_envs games, runs them together and stacks 4 frames so the AI can see movement."""
    env_fns = [make_mario_env(env_id, rank=i) for i in range(n_envs)]   # list of n_envs builder functions
    if use_subproc and n_envs > 1:                                # parallel processes only make sense for >1 env
        vec_env = SubprocVecEnv(env_fns)                          # each game runs in its own process
    else:
        vec_env = DummyVecEnv(env_fns)                            # all games run in this one process
    vec_env = VecFrameStack(vec_env, n_stack=N_STACK, channels_order="last")   # stack 4 frames on last axis
    return vec_env                                                # observation shape is now (84, 84, 4)


# ------------------------------- 4. CUSTOM CALLBACK ----------------------------------
class MarioStatsCallback(BaseCallback):
    """Sends Mario-specific numbers to TensorBoard: how far he gets and which level he reaches."""

    def __init__(self, verbose=0):
        super().__init__(verbose)                                 # 'verbose' controls how much text is printed

    def _on_step(self) -> bool:                                   # SB3 calls this after EVERY training step
        infos = self.locals["infos"]                              # list of 'info' dicts, one per environment
        dones = self.locals["dones"]                              # list of booleans: did each episode just end?
        for info, done in zip(infos, dones):                      # look at every environment
            if done:                                              # only when an episode has finished
                level = (info.get("world", 1) - 1) * 4 + info.get("stage", 1)   # world 2-3 -> level number 7
                self.logger.record_mean("mario/final_x_pos", info.get("x_pos", 0))          # how far right he got
                self.logger.record_mean("mario/level_reached", level)                       # 1..32
                self.logger.record_mean("mario/flag_rate", float(info.get("flag_get", False)))  # % reaching flag
                self.logger.record_mean("mario/game_beaten_rate", float(info.get("game_beaten", False)))
        return True                                               # returning False would stop training


# ------------------------------- 5. MAIN TRAINING FUNCTION ---------------------------
def main():
    phase = PHASES[PHASE]                                         # pick the settings for the chosen phase
    env_id = phase["env_id"]                                      # e.g. "SuperMarioBros-1-1-v0"
    total_steps = phase["steps"]                                  # how long to train in this run
    run_name = f"phase{PHASE}_{time.strftime('%Y%m%d_%H%M%S')}"   # unique name, e.g. phase1_20260929_101500
    run_dir = os.path.join(SAVE_DIR, run_name)                    # this run's folder inside saved_models/
    os.makedirs(LOG_DIR, exist_ok=True)                           # create training/logs (ignore if it exists)
    os.makedirs(run_dir, exist_ok=True)                           # create training/saved_models/<run_name>

    print(f"Training on {env_id} for {total_steps:,} steps  |  run name: {run_name}")   # tell the user

    train_env = build_vec_env(env_id, N_ENVS, USE_SUBPROC)        # the games the agent LEARNS from
    eval_env = build_vec_env(env_id, 1, False)                    # a separate single game used only for TESTING

    eval_callback = EvalCallback(                                 # test the model regularly, keep the best one
        eval_env,                                                 # the test environment
        best_model_save_path=run_dir,                             # writes  <run_dir>/best_model.zip when it improves
        log_path=os.path.join(LOG_DIR, run_name + "_eval"),       # saves evaluations.npz with all test scores
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),                    # counted per env, so divide by N_ENVS
        n_eval_episodes=N_EVAL_EPISODES,                          # play this many games per test
        deterministic=False,                                      # keep some randomness (Mario gets stuck otherwise)
        render=False,                                             # no game window while training
    )
    checkpoint_callback = CheckpointCallback(                     # safety copies in case the PC crashes
        save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),              # counted per env, so divide by N_ENVS
        save_path=os.path.join(run_dir, "checkpoints"),           # training/saved_models/<run>/checkpoints/
        name_prefix="mario",                                      # files become mario_250000_steps.zip, ...
    )
    callbacks = CallbackList([eval_callback, checkpoint_callback, MarioStatsCallback()])   # use all three

    if LOAD_MODEL_PATH:                                           # continue training an older model?
        model = PPO.load(LOAD_MODEL_PATH, env=train_env,          # load its brain and attach the new env
                         tensorboard_log=LOG_DIR, device="auto")  # keep logging to training/logs
        reset_steps = False                                       # keep counting steps from where it stopped
    else:                                                         # otherwise build a brand-new agent
        model = PPO(
            policy="CnnPolicy",                                   # CNN network because the input is a picture
            env=train_env,                                        # the games to learn from
            learning_rate=1e-4,                                   # size of each learning update (small = stable)
            n_steps=512,                                          # steps collected per env before each update
            batch_size=256,                                       # mini-batch size used in each update
            n_epochs=4,                                           # times PPO re-uses the same collected data
            gamma=0.99,                                           # discount: how much future reward matters
            gae_lambda=0.95,                                      # smoothing of the advantage estimate
            clip_range=0.2,                                       # PPO's limit on how big one update may be
            ent_coef=0.01,                                        # entropy bonus = encourages exploring
            vf_coef=0.5,                                          # weight of the value-function loss
            max_grad_norm=0.5,                                    # gradient clipping (prevents exploding updates)
            tensorboard_log=LOG_DIR,                              # where TensorBoard files are written
            seed=SEED,                                            # reproducibility
            device="auto",                                        # use the GPU (CUDA) if available, else the CPU
            verbose=1,                                            # print progress in the terminal
        )
        reset_steps = True                                        # new model -> step counter starts from 0

    interrupted = False
    try:
        model.learn(
            total_timesteps=total_steps,                          # how many game steps to train for
            callback=callbacks,                                   # attach best-model / checkpoint / stats callbacks
            tb_log_name=run_name,                                 # name of this run inside TensorBoard
            reset_num_timesteps=reset_steps,                      # start at 0 or keep the old count
            progress_bar=True,                                    # show a progress bar (needs tqdm + rich)
        )
    except KeyboardInterrupt:
        interrupted = True
        interrupted_path = os.path.join(run_dir, "interrupted_model")
        print("\nTraining interrupted. Saving the current model...", flush=True)
        model.save(interrupted_path)
    finally:
        train_env.close()                                         # close worker games on completion or interruption
        eval_env.close()

    if interrupted:
        print(f"Saved resume point: {interrupted_path}.zip")
        print("Run the notebook's resume cell, then start training again.")
        return

    final_path = os.path.join(run_dir, "final_model")             # where the last model will be stored
    model.save(final_path)                                        # writes final_model.zip
    print(f"Finished. Final model : {final_path}.zip")            # report locations
    print(f"          Best model  : {os.path.join(run_dir, 'best_model.zip')}")
    print(f"          TensorBoard : tensorboard --logdir {LOG_DIR}")



# ------------------------------- 6. ENTRY POINT --------------------------------------
if __name__ == "__main__":                                        # True only when you run THIS file directly
    main()                                                        # needed on Windows so subprocesses don't re-run it
