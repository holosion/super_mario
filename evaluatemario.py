# =====================================================================================
# evaluatemario.py  -  Watch a trained Mario agent play and record the results in TensorBoard
# Run it with:   python evaluatemario.py
# Then view:     tensorboard --logdir training/logs
# =====================================================================================

# ------------------------------- 1. IMPORTS ------------------------------------------
import os                                       # 'os' = folder / path tools
import glob                                     # 'glob' = find files that match a pattern (e.g. */best_model.zip)
import time                                     # 'time' = clock (sleep to slow the game to watchable speed)

import numpy as np                              # NumPy = averages / standard deviations of our scores
from torch.utils.tensorboard import SummaryWriter   # writes numbers to TensorBoard files

from stable_baselines3 import PPO               # we load a saved PPO model
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack   # same env setup as in training

# Re-use everything from the training file so training and evaluation always match exactly.
from trainmario import make_mario_env, LOG_DIR, SAVE_DIR, PHASES, N_STACK

# ------------------------------- 2. SETTINGS -----------------------------------------
PHASE_TO_EVAL = 1                               # which phase's game to test on (1 = level 1-1, 3 = full game)
MODEL_PATH = None                               # None = auto-pick newest best_model.zip; or give an exact path
N_EPISODES = 1                                  # one visible demo episode; increase for repeated evaluation
DETERMINISTIC = False                           # False = sample actions (usually plays better in Mario)
RENDER = os.environ.get("MARIO_RENDER", "1").lower() not in {"0", "false", "no"}
FPS = 30                                        # slows the window down so a human can follow (only if RENDER)
MAX_STEPS = 2_000                               # keep the first visible demo to about a minute at 30 FPS


# ------------------------------- 3. HELPERS ------------------------------------------
def find_newest_best_model():
    """Looks inside training/saved_models/*/best_model.zip and returns the most recently saved one."""
    pattern = os.path.join(SAVE_DIR, "*", "best_model.zip")       # match every run folder's best model
    candidates = glob.glob(pattern)                               # list of all matching files
    if not candidates:                                            # nothing found -> training has not run yet
        raise FileNotFoundError(f"No best_model.zip found under {SAVE_DIR}. Run trainmario.py first.")
    return max(candidates, key=os.path.getmtime)                  # newest file = largest modification time


def play_episodes(model, env_id, writer):
    """Plays N_EPISODES games with the model, prints results and logs them to TensorBoard."""
    base_env = DummyVecEnv([make_mario_env(env_id)])              # one Mario game (same wrappers as training)
    env = VecFrameStack(base_env, n_stack=N_STACK, channels_order="last")   # stack frames like in training
    game = base_env.envs[0].unwrapped                             # the raw game object (needed for the window)

    rewards, x_positions, levels = [], [], []                     # lists that collect scores of every game
    flags, beaten = 0, 0                                          # counters: levels cleared / whole game beaten

    for ep in range(N_EPISODES):                                  # play one game per loop turn
        obs = env.reset()                                         # start a new game, get the first observation
        done, ep_reward, steps, info = False, 0.0, 0, {}          # per-game bookkeeping
        while not done and steps < MAX_STEPS:                     # keep playing until game over (or the cap)
            action, _ = model.predict(obs, deterministic=DETERMINISTIC)   # the brain chooses a button combo
            obs, reward, dones, infos = env.step(action)          # press it; receive next picture and reward
            ep_reward += float(reward[0])                         # add up reward (env 0 is the only env)
            done = bool(dones[0])                                 # did this game just end?
            info = infos[0]                                       # extra facts: x_pos, world, stage, flag_get...
            steps += 1                                            # count agent steps
            if RENDER:                                            # show the game window?
                game.render_human()                               # draw the current frame on screen
                time.sleep(1.0 / FPS)                             # wait a little so it is watchable

        level = (info.get("world", 1) - 1) * 4 + info.get("stage", 1)   # convert world-stage into 1..32
        got_flag = bool(info.get("flag_get", False))              # did Mario reach the flag / axe?
        won = bool(info.get("game_beaten", False))                # did he beat 8-4 (Bowser + Princess)?
        rewards.append(ep_reward)                                 # store this game's score
        x_positions.append(info.get("x_pos", 0))                  # store how far right he got
        levels.append(level)                                      # store the level he ended on
        flags += int(got_flag)                                    # count flags reached
        beaten += int(won)                                        # count full-game wins

        writer.add_scalar("eval/episode_reward", ep_reward, ep)   # TensorBoard curve: reward per game
        writer.add_scalar("eval/final_x_pos", info.get("x_pos", 0), ep)   # TensorBoard curve: distance per game
        writer.add_scalar("eval/level_reached", level, ep)        # TensorBoard curve: level per game
        writer.add_scalar("eval/flag_reached", int(got_flag), ep) # 1 if the flag was reached, else 0
        writer.add_scalar("eval/game_beaten", int(won), ep)       # 1 if the whole game was beaten, else 0

        print(f"Game {ep + 1:>2}/{N_EPISODES} | reward {ep_reward:8.1f} | x_pos {int(info.get('x_pos', 0)):5d} | "
              f"world {info.get('world', '?')}-{info.get('stage', '?')} | flag {got_flag} | BEATEN {won}")

    env.close()                                                   # close the emulator window / process

    summary = {                                                   # final numbers for the whole test
        "mean_reward": float(np.mean(rewards)),                   # average score
        "std_reward": float(np.std(rewards)),                     # how much the score varies
        "mean_x_pos": float(np.mean(x_positions)),                # average distance travelled
        "max_level": int(np.max(levels)),                         # furthest level ever reached
        "flag_rate": flags / N_EPISODES,                          # fraction of games that reached a flag
        "win_rate": beaten / N_EPISODES,                          # fraction of games that beat the whole game
    }
    for name, value in summary.items():                           # write each summary number to TensorBoard
        writer.add_scalar(f"summary/{name}", value, 0)            # one point per summary value
    return summary                                                # give the numbers back to main()


# ------------------------------- 4. MAIN ---------------------------------------------
def main():
    env_id = PHASES[PHASE_TO_EVAL]["env_id"]                      # which Mario environment to test on
    model_path = MODEL_PATH or find_newest_best_model()           # use given path, else the newest best model
    print(f"Evaluating {model_path}\non {env_id}")                # tell the user what is being tested

    model = PPO.load(model_path, device="auto")                   # load the trained brain from the .zip file

    run_name = "eval_" + time.strftime("%Y%m%d_%H%M%S")           # unique name for this evaluation
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, "evaluation", run_name))   # TensorBoard writer

    summary = play_episodes(model, env_id, writer)                # play the games and collect the results

    writer.add_text("model", model_path)                          # store which model was tested
    writer.flush()                                                # force everything to disk
    writer.close()                                                # close the writer

    print("\n========== SUMMARY ==========")                      # pretty header
    for name, value in summary.items():                           # print every summary number
        print(f"{name:>12}: {value:.3f}")
    print("View graphs with:  tensorboard --logdir", LOG_DIR)     # reminder of the TensorBoard command


if __name__ == "__main__":                                        # run only when executed directly
    main()                                                        # start the evaluation
