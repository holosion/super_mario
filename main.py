"""Run Super Mario Bros. with random controller inputs.

Install dependencies once, if needed:
    python -m pip install gym-super-mario-bros
"""

import sys

import gymnasium

# gym-super-mario-bros still imports the old ``gym`` module, while current
# nes-py uses Gymnasium.  Alias it before importing Mario so both use the
# same environment API.
sys.modules["gym"] = gymnasium

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from nes_py.wrappers import JoypadSpace


def create_environment() -> JoypadSpace:
    """Create a Mario environment compatible with current dependencies."""
    # gym-super-mario-bros reads NES RAM as NumPy uint8 values.  Converting
    # before arithmetic keeps it compatible with NumPy 2.x.
    SuperMarioBrosEnv._x_position = property(
        lambda game: int(game.ram[0x6D]) * 0x100 + int(game.ram[0x86])
    )
    SuperMarioBrosEnv._x_position_screen = property(
        lambda game: (int(game.ram[0x86]) - int(game.ram[0x071C])) % 256
    )
    SuperMarioBrosEnv._y_position = property(
        lambda game: 255 + (255 - int(game.ram[0x03B8]))
        if int(game.ram[0x00B5]) < 1
        else 255 - int(game.ram[0x03B8])
    )
    return JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-v0"), SIMPLE_MOVEMENT)


def main() -> None:
    """Open Mario and repeatedly play using random actions."""
    env = create_environment()

    try:
        observation, info = env.reset()
        for _ in range(100_000):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()

            # Start another level attempt after Mario dies or the episode ends.
            if terminated or truncated:
                observation, info = env.reset()
    except KeyboardInterrupt:
        print("\nMario stopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
