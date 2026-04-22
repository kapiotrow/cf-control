import numpy as np


def quat_to_rot(q):
    qw, qx, qy, qz = q

    R = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )
    return R


def dynamics(x, u, p):

    v = x[3:6]
    q = x[6:10]
    w = x[10:13]

    m = p['mass']
    J = p['J']
    g = np.array([0, 0, -p['gravity']])

    T = u[0]
    tau = u[1:4]

    R = quat_to_rot(q)

    # translational
    r_dot = v
    v_dot = g + (1 / m) * R @ np.array([0, 0, T])

    # quaternion kinematics
    wx, wy, wz = w
    Omega = np.array([[0, -wx, -wy, -wz], [wx, 0, wz, -wy], [wy, -wz, 0, wx], [wz, wy, -wx, 0]])
    q_dot = 0.5 * Omega @ q

    # rotational
    w_dot = np.linalg.inv(J) @ (tau - np.cross(w, J @ w))

    return np.concatenate([r_dot, v_dot, q_dot, w_dot])
