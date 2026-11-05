"""Numerical MPC for a differential-drive robot."""

import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def predict_states(state, controls, dt):
    """Predict x, y and yaw after each velocity command."""
    state = np.asarray(state, dtype=float).copy()
    states = []

    for forward, angular in controls:
        # Integrate using the heading halfway through the time step.
        heading = state[2] + 0.5 * angular * dt
        state[0] += forward * math.cos(heading) * dt
        state[1] += forward * math.sin(heading) * dt
        state[2] += angular * dt
        states.append(state.copy())

    return np.asarray(states)


def solve_mpc(state, reference, previous_command, initial_guess=None,
              dt=0.2, max_speed=0.08, max_turn=0.35,
              max_acceleration=0.15, max_angular_acceleration=0.6,
              clearance_function=None):
    """Optimize a sequence of forward and angular velocities.

    reference: N x 3 array containing target x, y and yaw.
    clearance_function: optional function returning non-negative
    clearance margins for predicted states.
    """
    reference = np.asarray(reference, dtype=float)
    state = np.asarray(state, dtype=float)
    previous = np.asarray(previous_command, dtype=float)

    if (
        reference.ndim != 2
        or reference.shape[1] != 3
        or len(reference) == 0
        or state.shape != (3,)
        or previous.shape != (2,)
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(state))
        or not np.all(np.isfinite(previous))
    ):
        raise ValueError("Invalid MPC inputs.")

    horizon = len(reference)

    lower = np.tile([0.0, -max_turn], horizon)
    upper = np.tile([max_speed, max_turn], horizon)

    if initial_guess is None:
        guess = np.tile(previous, (horizon, 1))
    else:
        guess = np.asarray(initial_guess, dtype=float)
        if guess.shape != (horizon, 2):
            raise ValueError("Invalid initial guess.")

    guess = np.clip(guess.ravel(), lower, upper)

    # Constrain changes between consecutive commands.
    difference = np.eye(2 * horizon)
    for index in range(2, 2 * horizon):
        difference[index, index - 2] = -1.0

    limits = np.tile(
        [max_acceleration * dt, max_angular_acceleration * dt],
        horizon,
    )
    rate_lower = -limits.copy()
    rate_upper = limits.copy()
    rate_lower[:2] += previous
    rate_upper[:2] += previous

    constraints = [
        LinearConstraint(difference, rate_lower, rate_upper)
    ]

    def objective(flat_controls):
        controls = flat_controls.reshape(horizon, 2)
        predicted = predict_states(state, controls, dt)

        position_error = predicted[:, :2] - reference[:, :2]
        heading_error = wrap_angle(
            predicted[:, 2] - reference[:, 2]
        )
        changes = np.diff(
            np.vstack([previous, controls]), axis=0
        )

        return float(
            30.0 * np.sum(position_error ** 2)
            + 2.0 * np.sum(heading_error ** 2)
            + 0.1 * np.sum(controls[:, 0] ** 2)
            + 0.2 * np.sum(controls[:, 1] ** 2)
            + 2.0 * np.sum(changes ** 2)
            + 50.0 * np.sum(position_error[-1] ** 2)
        )

    if clearance_function is not None:
        def clearance_constraint(flat_controls):
            predicted = predict_states(
                state, flat_controls.reshape(horizon, 2), dt
            )
            return np.asarray(
                clearance_function(predicted), dtype=float
            )

        constraints.append({
            'type': 'ineq',
            'fun': clearance_constraint,
        })

    result = minimize(
        objective,
        guess,
        method='SLSQP',
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={'maxiter': 60, 'ftol': 1e-5},
    )

    if not result.success or not np.all(np.isfinite(result.x)):
        return None, None, str(result.message)

    controls = result.x.reshape(horizon, 2)
    predicted = predict_states(state, controls, dt)

    rates = difference @ result.x
    if (
        np.any(result.x < lower - 1e-6)
        or np.any(result.x > upper + 1e-6)
        or np.any(rates < rate_lower - 1e-6)
        or np.any(rates > rate_upper + 1e-6)
    ):
        return None, None, "Command constraint validation failed."

    if clearance_function is not None:
        margins = np.asarray(clearance_function(predicted))
        if not np.all(np.isfinite(margins)) or np.any(margins < -1e-6):
            return None, None, "Clearance validation failed."

    return controls, predicted, "OK"
