"""
DIAGNOSTIC SCRIPT
"""

import numpy as np


def diagnose_during_training(agent, episode_num, state, action, reward, details, next_state, current_sign, next_sign, epsilon, done):
    # Get Q-values for current state
    Q_local, _ = agent.q_local.forward(state[None, :])
    Q_target, _ = agent.q_target.forward(state[None, :])

    print(f"\n{'=' * 60}")
    print(f"DIAGNOSTIC - Episode {episode_num}")
    print(f"{'=' * 60}")
    print(
        f"State: {state[:4]} (walls) | pos=({state[12]:.2f}, {state[13]:.2f}) | goal_delta=({state[14]:.2f}, {state[15]:.2f})")
    print(f"Action taken: {action} | Reward: {reward:.3f} | Done: {done}")
    print(f"Reward details: {details}")
    print(
        f"Next state: {next_state[:4]} (walls) | pos=({next_state[12]:.2f}, {next_state[13]:.2f}) | goal_delta=({next_state[14]:.2f}, {next_state[15]:.2f})")
    print(f"Current sign: {current_sign}")
    print(f"Next sign: {next_sign}")
    print(f"Q_local:  {Q_local[0]}")
    print(f"Q_target: {Q_target[0]}")
    print(f"Max Q_local: {np.max(Q_local[0]):.4f} | Min: {np.min(Q_local[0]):.4f}")
    print(f"Epsilon: {epsilon}")

    # Check for issues
    if np.all(np.abs(Q_local[0]) < 0.01):
        print("WARNING: Q-values are near zero, network may not be learning")
    if np.all(np.isclose(Q_local[0], Q_local[0, 0])):
        print("WARNING: All Q-values are the same, network not differentiating actions")
    if np.any(np.isnan(Q_local[0])) or np.any(np.isinf(Q_local[0])):
        print("ERROR: NaN or Inf in Q-values")

    return Q_local[0]


def diagnose_test_episode(agent, env, max_steps=50):
    print("\n" + "=" * 70)
    print("DIAGNOSTIC TEST EPISODE (epsilon=0, greedy policy)")
    print("=" * 70)

    env.reset()
    state = env.get_state()

    positions_visited = [env.get_coordinates()]
    actions_taken = []
    q_values_history = []

    for t in range(max_steps):
        pos = env.get_coordinates()

        # Get Q-values
        Q, _ = agent.q_local.forward(state[None, :])
        q_vals = Q[0].copy()

        # Mask invalid actions
        valid_actions = agent.valid_actions(state)
        for a in range(4):
            if a not in valid_actions:
                q_vals[a] = -np.inf

        action = int(np.argmax(q_vals))

        print(f"\nStep {t}: pos={pos}")
        print(f"  Valid actions: {valid_actions}")
        print(f"  Q-values: L={Q[0, 0]:.3f} R={Q[0, 1]:.3f} U={Q[0, 2]:.3f} D={Q[0, 3]:.3f}")
        print(f"  Chosen (greedy): {['LEFT', 'RIGHT', 'UP', 'DOWN'][action]} (Q={q_vals[action]:.3f})")

        next_state, moved, goal_reached = env.execute_action(action)

        if not moved:
            print(f"Hit wall")

        actions_taken.append(action)
        q_values_history.append(Q[0].copy())

        if goal_reached:
            print(f"\nGOAL REACHED in {t + 1} steps!")
            return True, positions_visited, actions_taken

        new_pos = env.get_coordinates()
        positions_visited.append(new_pos)

        # Detect loops
        if len(positions_visited) >= 4:
            last_4 = positions_visited[-4:]
            if last_4[0] == last_4[2] and last_4[1] == last_4[3]:
                print(f"\nSTUCK IN LOOP: {last_4}")
                # why it's looping
                print(f"   Q-values suggest: always choosing {['L', 'R', 'U', 'D'][action]}")
                break

        state = next_state

    print(f"\n --- Failed to reach goal in {max_steps} steps --- ")
    print(f"Positions visited: {positions_visited[:20]}{'...' if len(positions_visited) > 20 else ''}")

    # Analyze Q-values
    q_arr = np.array(q_values_history)
    print(f"\nQ-value statistics across episode:")
    print(f"  Mean: {np.mean(q_arr):.4f}")
    print(f"  Std:  {np.std(q_arr):.4f}")
    print(f"  Range: [{np.min(q_arr):.4f}, {np.max(q_arr):.4f}]")

    return False, positions_visited, actions_taken


def check_network_weights(agent):
    """Check if network weights are reasonable."""
    print("\n" + "=" * 60)
    print("NETWORK WEIGHT ANALYSIS")
    print("=" * 60)

    w1_mag = np.abs(agent.q_local.W1).mean()
    w2_mag = np.abs(agent.q_local.W2).mean()
    b1_mag = np.abs(agent.q_local.b1).mean()
    b2_mag = np.abs(agent.q_local.b2).mean()

    print(f"Q_local weights:")
    print(f"  W1: mean_abs={w1_mag:.6f}, shape={agent.q_local.W1.shape}")
    print(f"  W2: mean_abs={w2_mag:.6f}, shape={agent.q_local.W2.shape}")
    print(f"  b1: mean_abs={b1_mag:.6f}")
    print(f"  b2: mean_abs={b2_mag:.6f}")

    # Compare local vs target
    w1_diff = np.abs(agent.q_local.W1 - agent.q_target.W1).mean()
    w2_diff = np.abs(agent.q_local.W2 - agent.q_target.W2).mean()

    print(f"\nLocal vs Target difference:")
    print(f"  W1 diff: {w1_diff:.6f}")
    print(f"  W2 diff: {w2_diff:.6f}")

    if w1_diff < 1e-10 and w2_diff < 1e-10:
        print("WARNING: Local and Target are identical - soft update may not be working")

    if w1_mag < 1e-6 or w2_mag < 1e-6:
        print("ERROR: Weights are essentially zero")

    if w1_mag > 100 or w2_mag > 100:
        print("ERROR: Weights are exploding")


def test_gradient_flow(agent):
    """Verify gradients are flowing correctly."""
    print("\n" + "=" * 60)
    print("GRADIENT FLOW TEST")
    print("=" * 60)

    # Create a fake batch
    states = np.random.randn(32, agent.q_local.n_features)
    actions = np.random.randint(0, 4, size=32)
    targets = np.random.randn(32) * 10  # Random targets

    # Get initial weights
    W1_before = agent.q_local.W1.copy()
    W2_before = agent.q_local.W2.copy()

    # Do one update
    Q, cache = agent.q_local.forward(states)
    N = Q.shape[0]
    q_sa = Q[np.arange(N), actions]

    dQ = np.zeros_like(Q)
    dQ[np.arange(N), actions] = (q_sa - targets) / N

    grads = agent.q_local.backward(cache, dQ)

    print(f"Gradient magnitudes:")
    print(f"  dW1: {np.abs(grads['W1']).mean():.6f}")
    print(f"  dW2: {np.abs(grads['W2']).mean():.6f}")
    print(f"  db1: {np.abs(grads['b1']).mean():.6f}")
    print(f"  db2: {np.abs(grads['b2']).mean():.6f}")

    agent.q_local.update(grads)

    W1_after = agent.q_local.W1
    W2_after = agent.q_local.W2

    w1_change = np.abs(W1_after - W1_before).mean()
    w2_change = np.abs(W2_after - W2_before).mean()

    print(f"\nWeight changes after update:")
    print(f"  W1 change: {w1_change:.6f}")
    print(f"  W2 change: {w2_change:.6f}")

    if w1_change < 1e-10:
        print("ERROR: W1 not changing")
    if w2_change < 1e-10:
        print("ERROR: W2 not changing")

    # Restore original weights
    agent.q_local.W1 = W1_before
    agent.q_local.W2 = W2_before


def full_diagnosis(agent, env):
    """Run complete diagnosis."""
    print("\n" + "#" * 70)
    print("# FULL DIAGNOSTIC REPORT")
    print("#" * 70)

    check_network_weights(agent)
    test_gradient_flow(agent)

    print("\n" + "=" * 60)
    print("RUNNING 3 TEST EPISODES")
    print("=" * 60)

    successes = 0
    for i in range(3):
        print(f"\n--- Test Episode {i + 1} ---")
        success, positions, actions = diagnose_test_episode(agent, env, max_steps=100)
        if success:
            successes += 1

    print(f"\n\nSUMMARY: {successes}/3 test episodes succeeded")

    return successes