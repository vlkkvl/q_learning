import numpy as np
from simulation.signs import SignType

# Give penalties for bad decisions and rewards for good ones.
# E.g. going into the dead end results in reward -= 1
def assign_reward(moved, goal_reached, sign):
    reward = -0.2 # step cost
    done = False

    if goal_reached:
        reward += 100.0
        done = True

    if not moved:
        reward -= 1.0  # wall

    if isinstance(sign, np.ndarray):
        sign = sign.tolist()

    # determine sign type
    sign_type = SignType.from_vector(sign) if sign is not None else None

    # warning signs
    if sign_type == SignType.DEAD_END:
        reward -= 5.0
        print("DEAD END penalty applied.")
    elif sign_type in {
        SignType.AVOID_LEFT,
        SignType.AVOID_RIGHT,
        SignType.AVOID_UP,
        SignType.AVOID_DOWN,
    }:
        reward -= 2.0
        print("AVOID penalty applied.")

    return reward, done