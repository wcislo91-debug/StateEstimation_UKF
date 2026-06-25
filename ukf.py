# Starter code for the Coursera SDC Course 2 final project.
#
# Author: Trevor Ablett and Jonathan Kelly
# University of Toronto Institute for Aerospace Studies
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion

#### 1. Data ###################################################################################

with open('data/pt1_data.pkl', 'rb') as file:
    data = pickle.load(file)

gt    = data['gt']
imu_f = data['imu_f']
imu_w = data['imu_w']
gnss  = data['gnss']
lidar = data['lidar']

gt_fig = plt.figure()
ax = gt_fig.add_subplot(111, projection='3d')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Ground Truth trajectory')
ax.set_zlim(-1, 5)
plt.show()

# Correct calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.1).
C_li = np.array([
   [ 0.99376, -0.09722,  0.05466],
   [ 0.09971,  0.99401, -0.04475],
   [-0.04998,  0.04992,  0.9975 ]
])

t_i_li = np.array([0.5, 0.1, 0.5])

# Transform from the LIDAR frame to the vehicle (IMU) frame.
lidar.data = (C_li @ lidar.data.T).T + t_i_li

#### 2. Constants ##############################################################################

var_imu_f = 0.10
var_imu_w = 0.25
var_gnss  = 0.01
var_lidar = 0.01

g = np.array([0, 0, -9.81])  # gravity
l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)

#### 3. State definition and initial values (UKF) ###############################################

# UKF state: x = [p(3), v(3), theta(3)], theta is axis-angle orientation
n_x = 9

p_est  = np.zeros([imu_f.data.shape[0], 3])
v_est  = np.zeros([imu_f.data.shape[0], 3])
q_est  = np.zeros([imu_f.data.shape[0], 4])
p_cov  = np.zeros([imu_f.data.shape[0], n_x, n_x])

p0 = gt.p[0]
v0 = gt.v[0]
q0 = Quaternion(euler=gt.r[0])
theta0 = q0.to_axis_angle()

x0 = np.zeros(n_x)
x0[0:3] = p0
x0[3:6] = v0
x0[6:9] = theta0

P0 = np.eye(n_x) * 0.1

p_est[0] = p0
v_est[0] = v0
q_est[0] = q0.to_numpy()
p_cov[0] = P0

x_prev = x0.copy()
P_prev = P0.copy()

gnss_i  = 0
lidar_i = 0

#### 4. UKF helper functions ###################################################################

def state_to_vec(p, v, theta):
    x = np.zeros(n_x)
    x[0:3] = p
    x[3:6] = v
    x[6:9] = theta
    return x

def vec_to_state(x):
    p = x[0:3]
    v = x[3:6]
    theta = x[6:9]
    return p, v, theta

def process_model(x, f_imu, w_imu, dt):
    """
    Nonlinear motion model for UKF.
    State: [p, v, theta_axis_angle]
    """
    p, v, theta = vec_to_state(x)

    # Build quaternion from axis-angle
    q = Quaternion(axis_angle=theta)
    C_I_v = q.to_mat()

    # acceleration in inertial frame
    f_inertial = C_I_v @ f_imu + g

    # propagate position and velocity
    p_new = p + v * dt + 0.5 * f_inertial * (dt ** 2)
    v_new = v + f_inertial * dt

    # propagate orientation: compose with incremental rotation from angular velocity
    q_delta = Quaternion(axis_angle=w_imu * dt)
    q_new_np = q.quat_mult_left(q_delta.to_numpy(), out='np')
    # normalize to avoid drift
    q_new_np = q_new_np / np.linalg.norm(q_new_np)
    q_new = Quaternion(*q_new_np)
    theta_new = q_new.to_axis_angle()

    return state_to_vec(p_new, v_new, theta_new)

def measurement_model(x):
    """
    Measurement model: position only.
    """
    p, _, _ = vec_to_state(x)
    return p

def generate_sigma_points(x, P, alpha=0.2, beta=2.0, kappa=0.0):
    """
    Stable sigma-point generation with jitter and symmetry enforcement.
    """
    n = x.size
    lam = alpha**2 * (n + kappa) - n

    P = 0.5 * (P + P.T)

    jitter = 1e-9
    while True:
        try:
            S = np.linalg.cholesky((n + lam) * P + jitter * np.eye(n))
            break
        except np.linalg.LinAlgError:
            jitter *= 10

    sigma_points = np.zeros((2 * n + 1, n))
    sigma_points[0] = x
    for i in range(n):
        sigma_points[i + 1]     = x + S[i]
        sigma_points[i + 1 + n] = x - S[i]

    Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
    Wc = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
    Wm[0] = lam / (n + lam)
    Wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)

    return sigma_points, Wm, Wc

def ukf_predict(x, P, f_imu, w_imu, dt, Q):
    """
    UKF prediction step.
    """
    n = x.size
    sigma_points, Wm, Wc = generate_sigma_points(x, P)

    X_pred = np.zeros_like(sigma_points)
    for i in range(2 * n + 1):
        X_pred[i] = process_model(sigma_points[i], f_imu, w_imu, dt)

    x_pred = np.sum(Wm[:, None] * X_pred, axis=0)

    P_pred = np.zeros((n, n))
    for i in range(2 * n + 1):
        dx = X_pred[i] - x_pred
        P_pred += Wc[i] * np.outer(dx, dx)
    P_pred += Q

    P_pred = 0.5 * (P_pred + P_pred.T)

    return x_pred, P_pred

def ukf_update(x_pred, P_pred, y, R):
    """
    UKF measurement update (position).
    """
    n = x_pred.size
    m = 3

    sigma_points, Wm, Wc = generate_sigma_points(x_pred, P_pred)

    Z = np.zeros((2 * n + 1, m))
    for i in range(2 * n + 1):
        Z[i] = measurement_model(sigma_points[i])

    z_pred = np.sum(Wm[:, None] * Z, axis=0)

    S = np.zeros((m, m))
    Cxz = np.zeros((n, m))
    for i in range(2 * n + 1):
        dz = Z[i] - z_pred
        dx = sigma_points[i] - x_pred
        S   += Wc[i] * np.outer(dz, dz)
        Cxz += Wc[i] * np.outer(dx, dz)
    S += R

    S = 0.5 * (S + S.T)
    jitter = 1e-9
    while True:
        try:
            np.linalg.cholesky(S + jitter * np.eye(m))
            break
        except np.linalg.LinAlgError:
            jitter *= 10
    S = S + jitter * np.eye(m)

    K = Cxz @ np.linalg.inv(S)

    innovation = y - z_pred
    x_upd = x_pred + K @ innovation
    P_upd = P_pred - K @ S @ K.T

    P_upd = 0.5 * (P_upd + P_upd.T)

    return x_upd, P_upd

#### 5. Main Filter Loop (UKF) #################################################################

Q = np.zeros((n_x, n_x))
Q[3:6, 3:6] = np.eye(3) * (var_imu_f * 1.0)
Q[6:9, 6:9] = np.eye(3) * (var_imu_w * 0.1)

for k in range(1, imu_f.data.shape[0]):
    # timestamps are in ms → convert to seconds
    dt = (imu_f.t[k] - imu_f.t[k - 1]) * 1e-3

    f_imu = imu_f.data[k]
    w_imu = imu_w.data[k]

    x_pred, P_pred = ukf_predict(x_prev, P_prev, f_imu, w_imu, dt, Q)

    # GNSS
    while gnss_i < len(gnss.t) and gnss.t[gnss_i] <= imu_f.t[k]:
        if abs(gnss.t[gnss_i] - imu_f.t[k]) < 1e-2:
            # anisotropic GNSS noise: vertical less accurate
            R_gnss = np.diag([var_gnss, var_gnss, 0.1])
            x_pred, P_pred = ukf_update(x_pred, P_pred, gnss.data[gnss_i], R_gnss)
        gnss_i += 1

    # LIDAR
    while lidar_i < len(lidar.t) and lidar.t[lidar_i] <= imu_f.t[k]:
        if lidar.t[lidar_i] == imu_f.t[k]:
            R_lidar = np.diag([var_lidar, var_lidar, 0.1])
            x_pred, P_pred = ukf_update(x_pred, P_pred, lidar.data[lidar_i], R_lidar)
        lidar_i += 1

    p_k, v_k, theta_k = vec_to_state(x_pred)
    q_k = Quaternion(axis_angle=theta_k).to_numpy()

    p_est[k] = p_k
    v_est[k] = v_k
    q_est[k] = q_k
    p_cov[k] = P_pred

    x_prev = x_pred
    P_prev = P_pred

#### 6. Results and Analysis ###################################################################

est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111, projection='3d')
ax.plot(p_est[:,0], p_est[:,1], p_est[:,2], label='Estimated')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2], label='Ground Truth')
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_zlabel('Up [m]')
ax.set_title('Ground Truth and Estimated Trajectory')
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
ax.set_zlim(-2, 2)
ax.set_xticks([0, 50, 100, 150, 200])
ax.set_yticks([0, 50, 100, 150, 200])
ax.set_zticks([-2, -1, 0, 1, 2])
ax.legend(loc=(0.62,0.77))
ax.view_init(elev=45, azim=-50)
plt.show()

error_fig, ax = plt.subplots(2, 3)
error_fig.suptitle('Error Plots')
num_gt = gt.p.shape[0]
p_est_euler = []
p_cov_euler_std = []

for i in range(len(q_est)):
    qc = Quaternion(*q_est[i, :])
    p_est_euler.append(qc.to_euler())

    J = rpy_jacobian_axis_angle(qc.to_axis_angle())
    p_cov_euler_std.append(np.sqrt(np.diagonal(J @ p_cov[i, 6:, 6:] @ J.T)))

p_est_euler = np.array(p_est_euler)
p_cov_euler_std = np.array(p_cov_euler_std)

p_cov_std = np.sqrt(np.maximum(
    0.0,
    np.diagonal(p_cov[:, :6, :6], axis1=1, axis2=2)
))

titles = ['Easting', 'Northing', 'Up', 'Roll', 'Pitch', 'Yaw']
pos_clip = 10.0
angle_clip = 0.5
for i in range(3):
    position_error = np.clip(gt.p[:, i] - p_est[:num_gt, i], -pos_clip, pos_clip)
    pos_bound = np.clip(3 * p_cov_std[:num_gt, i], -pos_clip, pos_clip)
    ax[0, i].plot(range(num_gt), position_error)
    ax[0, i].plot(range(num_gt),  pos_bound, 'r--')
    ax[0, i].plot(range(num_gt), -pos_bound, 'r--')
    ax[0, i].set_title(titles[i])
ax[0,0].set_ylabel('Meters')

for i in range(3):
    angle_error = np.clip(angle_normalize(gt.r[:, i] - p_est_euler[:num_gt, i]), -angle_clip, angle_clip)
    ang_bound = np.clip(3 * p_cov_euler_std[:num_gt, i], -angle_clip, angle_clip)
    ax[1, i].plot(range(num_gt), angle_error)
    ax[1, i].plot(range(num_gt),  ang_bound, 'r--')
    ax[1, i].plot(range(num_gt), -ang_bound, 'r--')
    ax[1, i].set_title(titles[i+3])
ax[1,0].set_ylabel('Radians')
plt.show()

plt.figure(figsize=(12,4))
plt.title("Sensor Availability Timeline")
plt.plot(imu_f.t, np.zeros_like(imu_f.t), 'k.', markersize=2, label='IMU')
plt.plot(gnss.t, np.ones_like(gnss.t), 'g.', markersize=4, label='GNSS')
plt.plot(lidar.t, 2*np.ones_like(lidar.t), 'r.', markersize=4, label='LIDAR')
plt.yticks([0,1,2], ['IMU','GNSS','LIDAR'])
plt.xlabel("Time [ms]")
plt.grid(True)
plt.legend()
plt.show()

#### 7. Submission #############################################################################

p1_indices = [9000, 9400, 9800, 10200, 10600]
p1_str = ''
for val in p1_indices:
    for i in range(3):
        p1_str += '%.3f ' % (p_est[val, i])
with open('pt1_submission.txt', 'w') as file:
    file.write(p1_str)

# Uncomment for parts 2 and 3 as needed.
