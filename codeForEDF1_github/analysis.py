import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from functools import partial
from config import ensure_directory_exists


def system_dynamics(x, W, inputVector):
    x = np.array(x)
    dx = - x + np.dot(W, np.maximum(x, 0)) + inputVector
    return dx


def nullclines_and_fixed_points(W, inputVector, fixed_indices, fixed_values, x_range=np.linspace(-10, 40, 500)):
    # x_range=np.linspace(0, 25, 500)
    x_range=np.linspace(0, 15, 500)
    x1, x2 = np.meshgrid(x_range, x_range)
    free_indices = [i for i in range(4) if i not in fixed_indices]
    dynamics_func = partial(system_dynamics, W=W, inputVector=inputVector)
    U = np.zeros_like(x1)
    V = np.zeros_like(x2)

    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            state = np.zeros(4)
            state[fixed_indices] = fixed_values
            state[free_indices] = [x1[i, j], x2[i, j]]
            dx = dynamics_func(state)
            U[i, j] = dx[free_indices[0]]
            V[i, j] = dx[free_indices[1]]

    null_x1 = np.abs(U) < 0.05
    null_x2 = np.abs(V) < 0.05
    magnitude = np.sqrt(U ** 2 + V ** 2)
    U /= magnitude
    V /= magnitude

    def eqs(varls):
        full_state = np.zeros(4)
        full_state[fixed_indices] = fixed_values
        full_state[free_indices] = varls
        return dynamics_func(full_state)[free_indices]

    fixed_point = fsolve(eqs, np.zeros(len(free_indices)))
    return x1, x2, null_x1, null_x2, U, V, fixed_point


def plot_results(x1, x2, null_x1, null_x2, U, V, fixed_point, trajectories, network_type, title_suffix, magnitude, fixed_indices):
   # total_indices = set(range(4))
    free_indices = [3,2] #list(total_indices - set(fixed_indices))
    print(fixed_indices)
    print(free_indices)
    plt.figure(figsize=(10, 8))
    plt.streamplot(x1, x2, U, V, color=magnitude, linewidth=1, cmap='gray', density=1, arrowstyle='-|>', arrowsize=1.0)
    plt.plot(x1[null_x1], x2[null_x1], 'r.', markersize=2, label='dx1/dt=0 Nullcline')
    plt.plot(x1[null_x2], x2[null_x2], 'b.', markersize=2, label='dx2/dt=0 Nullcline')
    plt.plot(fixed_point[0], fixed_point[1], 'ko', label='Fixed Point')
    trajectory = trajectories[4]
    plt.plot(trajectory[free_indices[0], :], trajectory[free_indices[1], :], label='Trajectory')
    end_x = trajectory[free_indices[0], -1]
    end_y = trajectory[free_indices[1], -1]
    plt.plot(end_x, end_y, marker='^', markersize=10, color='green')
    start_x = trajectory[free_indices[0], 0]
    start_y = trajectory[free_indices[1], 0]
    plt.plot(start_x, start_y, marker='o', markersize=10, color='blue')
    plt.xlabel('x{} (free variable)'.format(free_indices[0] + 1))
    plt.ylabel('x{} (free variable)'.format(free_indices[1] + 1))
    plt.legend()
    plt.title('Phase Portrait with Nullclines, Fixed Point, and Trajectories')
    network_folder = ensure_directory_exists(network_type)
    filename = f"{title_suffix.replace(' ', '_')}_Phase_Plane.png"
    filename1 = f"{title_suffix.replace(' ', '_')}_Phase_Plane.eps"
    plt.savefig(os.path.join(network_folder, filename))
    plt.savefig(os.path.join(network_folder, filename1))
    plt.close()


def compute_correlation_matrix(trajectories):
    trajectories = np.array(trajectories)
    norm_trajectories = trajectories / np.linalg.norm(trajectories, axis=0)
    correlation_matrix = norm_trajectories.T @ norm_trajectories
    return correlation_matrix


def plot_correlation_matrix(correlation_matrix, title, network_type):
    plt.figure(figsize=(8, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest', vmin=0.5, vmax=1)
    plt.colorbar()
    plt.title(f'Correlation Matrix - {title}')
    plt.xlabel('Time Points')
    plt.ylabel('Time Points')
    network_folder = ensure_directory_exists(network_type)
    filename = f"{title.replace(' ', '_')}_correlation_matrix.png"
    plt.savefig(os.path.join(network_folder, filename))
    print(f'Saved correlation matrix as {filename}')
