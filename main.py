"""Run Super Mario Bros. with random controller inputs.

Install dependencies once, if needed:
    python -m pip install gym-super-mario-bros
"""

import sys

import cv2
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
    # v0 uses the original NES "vanilla" view instead of rectangle crop.
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    # Avoid pyglet viewer issues on Python 3.13 by using RGB frame rendering.
    env.unwrapped.render_mode = "rgb_array"
    return JoypadSpace(env, SIMPLE_MOVEMENT)


def main() -> None:
    """Open Mario and play one episode with random actions."""
    env = create_environment()

    try:
        observation, info = env.reset()
        for _ in range(100_000):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            if frame is not None:
                cv2.imshow("Super Mario AI", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                # Allow immediate quit from keyboard or window close button.
                if key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty("Super Mario AI", cv2.WND_PROP_VISIBLE) < 1:
                    break

            # Stop after this episode; do not auto-restart forever.
            if terminated or truncated:
                break
    except KeyboardInterrupt:
        print("\nMario stopped.")
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
