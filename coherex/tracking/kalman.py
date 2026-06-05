import numpy as np
from filterpy.kalman import KalmanFilter


def create_kalman_filter(init_x, init_y):
    """
    State: [x, y, vx, vy]
    Measurement: [x, y]
    """

    kf = KalmanFilter(dim_x=4, dim_z=2)

    # Constant velocity model
    kf.F = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)

    # Measurement model
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=float)

    # Initial state
    kf.x = np.array([[init_x], [init_y], [0.0], [0.0]])

    # State covariance
    kf.P *= 1000.0

    # Measurement noise (trust detections more to capture spikes)
    kf.R = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    # Process noise
    kf.Q *= 0.05

    return kf
