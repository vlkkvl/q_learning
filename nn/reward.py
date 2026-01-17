import numpy as np
from simulation.signs import SignType

PREFERRED_ACTIONS = {
    SignType.PREFER_LEFT: 0,
    SignType.PREFER_RIGHT: 1,
    SignType.PREFER_UP: 2,
    SignType.PREFER_DOWN: 3
}

DEAD_ENDS = {
    SignType.DEAD_END_LEFT: 0,
    SignType.DEAD_END_RIGHT: 1,
    SignType.DEAD_END_UP: 2,
    SignType.DEAD_END_DOWN: 3
}

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def assign_reward(
        moved,
        goal_reached,
        next_sign,
        current_sign,
        action,
        next_available_actions,
        current_position,
        next_position,
        visited_counts,
        goal_position,
):
    reward = -0.01  # step cost
    done = False
    details = {"step": -0.01}

    if goal_reached:
        reward += 100.0
        details["goal"] = +100.0
        done = True
        return reward, done, details

    # invalid action taken
    if not moved:
        reward -= 10 # wall
        details["wall"] = -10
        done = True
        return reward, done, details

    # === DISTANCE-BASED SHAPING ===
    prev_distance = manhattan(current_position, goal_position)
    next_distance = manhattan(next_position, goal_position)

    # reward for getting closer, penalty for moving away
    distance_delta = prev_distance - next_distance
    distance_reward = distance_delta * 0.5
    reward += distance_reward
    details["distance"] = distance_reward

    # === SIGN PROCESSING ===
    if isinstance(next_sign, np.ndarray):
        next_sign = next_sign.tolist()

    if isinstance(current_sign, np.ndarray):
        current_sign = current_sign.tolist()

    # determine sign type
    next_sign_type = SignType.from_vector(next_sign) if next_sign is not None else None
    current_sign_type = SignType.from_vector(current_sign) if current_sign is not None else None

    # === PREFER SIGN REWARD ===
    if current_sign_type in PREFERRED_ACTIONS:
        preferred_action = PREFERRED_ACTIONS[current_sign_type]
        if action == preferred_action:
            reward += 2.0
            details["followed_sign"] = +2.0

    # === DEAD ENDS ===
    is_dead_end_cell_ahead = (
            next_available_actions is not None
            and np.sum(np.asarray(next_available_actions) > 0.5) == 1
    )

    if is_dead_end_cell_ahead:
        reward -= 1.0
        details["dead_end"] = -1.0
    elif next_sign_type in DEAD_ENDS:
        dead_end_direction = DEAD_ENDS[next_sign_type]
        if action == dead_end_direction:
            reward -= 1.0
            details["dead_end"] = -1.0

    # === EXPLORATION/REVISIT PENALTIES ===
    k = visited_counts.get(next_position, 0)

    if k == 0:
        details["new_cell"] = +0.3
        reward += 0.3  # bonus for new cells
    else:
        revisit_pen = min(1.5, 0.1 * k)  # k=1 -> -0.2, k=2 -> -0.4, ...
        reward -= revisit_pen
        details["revisit"] = -revisit_pen

    return reward, done, details