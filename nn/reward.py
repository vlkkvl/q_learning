import numpy as np
from simulation.signs import SignType

# Give penalties for bad decisions and rewards for good ones.
# E.g. going into the dead end results in reward -= 1

AVOID_ACTIONS = {
    SignType.AVOID_LEFT: 0,
    SignType.AVOID_RIGHT: 1,
    SignType.AVOID_UP: 2,
    SignType.AVOID_DOWN: 3,
}

def assign_reward(
    moved,
    goal_reached,
    sign,
    next_position,
    previous_position,
    visited_positions,
    goal_position,
    previous_distance
):
    reward = -0.05 # step cost
    done = False

    if goal_reached:
        reward += 50.0
        done = True

    if not moved:
        reward -= 1.5  # wall

    if isinstance(sign, np.ndarray):
        sign = sign.tolist()

    # determine sign type
    sign_type = SignType.from_vector(sign) if sign is not None else None

    # warning signs
    if sign_type == SignType.DEAD_END:
        reward -= 2.0
        print("DEAD END penalty applied.")

    updated_distance = previous_distance

    if moved:
        if next_position not in visited_positions:
            reward += 0.1  # progress_bonus
            visited_positions.add(next_position)
        else:
            reward -= 0.5  # revisit_penalty

        if (
                previous_position is not None
                and next_position == previous_position
                and SignType.from_vector(sign) != SignType.DEAD_END
        ):
            reward -= 0.5  # backtrack_penalty

        if goal_position is not None:
            dx = abs(next_position[0] - goal_position[0])
            dy = abs(next_position[1] - goal_position[1])
            updated_distance = dx + dy
            if previous_distance is not None and updated_distance < previous_distance:
                reward += 1  # distance_bonus

    return reward, done, updated_distance