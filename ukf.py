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

################################################################################################
# This is where you will load the data from the pickle files. For parts 1 and 2, you will use
# p1_data.pkl. For Part 3, you will use pt3_data.pkl.
################################################################################################
with open('data/pt1_data.pkl', 'rb') as file:
    data = pickle.load(file)

################################################################################################
# Each element of the data dictionary is stored as an item from the data dictionary, which we
# will store in local variables, described by the following:
#   gt: Data object containing ground truth. with the following fields:
#     a: Acceleration of the vehicle, in the inertial frame
#     v: Velocity of the vehicle, in the inertial frame
#     p: Position of the vehicle, in the inertial frame
#     alpha: Rotational acceleration of the vehicle, in the inertial frame
#     w: Rotational velocity of the vehicle, in the inertial frame
#     r: Rotational position of the vehicle, in Euler (XYZ) angles in the inertial frame
#     _t: Timestamp in ms.
#   imu_f: StampedData object with the imu specific force data (given in vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   imu_w: StampedData object with the imu rotational velocity (given in the vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   gnss: StampedData object with the GNSS data.
#     data: The actual data
#     t: Timestamps in ms.
#   lidar: StampedData object with the LIDAR data (positions only).
#     data: The actual data
#     t: Timestamps in ms.
################################################################################################
gt = data['gt']
imu_f = data['imu_f']
imu_w = data['imu_w']
gnss = data['gnss']
lidar = data['lidar']

################################################################################################
# Let's plot the ground truth trajectory to see what it looks like. When you're testing your
# code later, feel free to comment this out.
################################################################################################
gt_fig = plt.figure()
ax = gt_fig.add_subplot(111, projection='3d')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Ground Truth trajectory')
ax.set_zlim(-1, 5)
plt.show()

################################################################################################
# Remember that our LIDAR data is actually just a set of positions estimated from a separate
# scan-matching system, so we can insert it into our solver as another position measurement,
# just as we do for GNSS. However, the LIDAR frame is not the same as the frame shared by the
# IMU and the GNSS. To remedy this, we transform the LIDAR data to the IMU frame using our 
# known extrinsic calibration rotation matrix C_li and translation vector t_i_li.
#
# THIS IS THE CODE YOU WILL MODIFY FOR PART 2 OF THE ASSIGNMENT.
################################################################################################
# Correct calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.1).
C_li = np.array([
   [ 0.99376, -0.09722,  0.05466],
   [ 0.09971,  0.99401, -0.04475],
   [-0.04998,  0.04992,  0.9975 ]
])

# Incorrect calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.05).
#C_li = np.array([
#     [ 0.9975 , -0.04742,  0.05235],
#     [ 0.04992,  0.99763, -0.04742],
#     [-0.04998,  0.04992,  0.9975 ]
#])

t_i_li = np.array([0.5, 0.1, 0.5])

# Transform from the LIDAR frame to the vehicle (IMU) frame.
lidar.data = (C_li @ lidar.data.T).T + t_i_li

#### 2. Constants ##############################################################################

################################################################################################
# Now that our data is set up, we can start getting things ready for our solver. One of the
# most important aspects of a filter is setting the estimated sensor variances correctly.
# We set the values here.
################################################################################################
var_imu_f = 0.10
var_imu_w = 0.25
var_gnss  = 0.05
var_lidar = 0.05
#var_lidar = 1e6  # Effectively ignores LIDAR measurements

################################################################################################
# We can also set up some constants that won't change for any iteration of our solver.
################################################################################################
g = np.array([0, 0, -9.81])  # gravity
# EKF Jacobians kept for reference; UKF does not use them directly.
l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)

#### 3. State definition and initial values (UKF) ###############################################

# UKF state: x = [p(3), v(3), theta(3)], theta is axis-angle orientation
n_x = 9

p_est  = np.zeros([imu_f.data.shape[0], 3])        # position estimates
v_est  = np.zeros([imu_f.data.shape[0], 3])        # velocity estimates
q_est  = np.zeros([imu_f.data.shape[0], 4])        # orientation estimates as quaternions
p_cov  = np.zeros([imu_f.data.shape[0], n_x, n_x]) # covariance matrices

# Initial state from ground truth
p0 = gt.p[0]
v0 = gt.v[0]
q0 = Quaternion(euler=gt.r[0])          # from Euler to quaternion
theta0 = q0.to_axis_angle()             # axis-angle (3)

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
    q_new = Quaternion(*q_new_np)
    theta_new = q_new.to_axis_angle()

    return state_to_vec(p_new, v_new, theta_new)

def measurement_model(x):
    """
    Measurement model: position only.
    """
    p, _, _ = vec_to_state(x)
    return p  # 3D position

def generate_sigma_points(x, P, alpha=0.05, beta=2.0, kappa=0.0):
    """
    Stable sigma-point generation with jitter and symmetry enforcement.
    """
    n = x.size
    lam = alpha**2 * (n + kappa) - n

    # enforce symmetry
    P = 0.5 * (P + P.T)

    # add jitter until PD
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

    # propagate sigma points through process model
    X_pred = np.zeros_like(sigma_points)
    for i in range(2 * n + 1):
        X_pred[i] = process_model(sigma_points[i], f_imu, w_imu, dt)

    # mean state (linear in axis-angle representation)
    x_pred = np.sum(Wm[:, None] * X_pred, axis=0)

    # covariance
    P_pred = np.zeros((n, n))
    for i in range(2 * n + 1):
        dx = X_pred[i] - x_pred
        P_pred += Wc[i] * np.outer(dx, dx)
    P_pred += Q

    # enforce symmetry
    P_pred = 0.5 * (P_pred + P_pred.T)

    return x_pred, P_pred

def ukf_update(x_pred, P_pred, y, R):
    """
    UKF measurement update (position).
    """
    n = x_pred.size
    m = 3  # measurement dimension (position)

    sigma_points, Wm, Wc = generate_sigma_points(x_pred, P_pred)

    # propagate sigma points through measurement model
    Z = np.zeros((2 * n + 1, m))
    for i in range(2 * n + 1):
        Z[i] = measurement_model(sigma_points[i])

    # predicted measurement mean
    z_pred = np.sum(Wm[:, None] * Z, axis=0)

    # innovation covariance and cross-covariance
    S = np.zeros((m, m))
    Cxz = np.zeros((n, m))
    for i in range(2 * n + 1):
        dz = Z[i] - z_pred
        dx = sigma_points[i] - x_pred
        S   += Wc[i] * np.outer(dz, dz)
        Cxz += Wc[i] * np.outer(dx, dz)
    S += R

    # enforce symmetry and jitter on S
    S = 0.5 * (S + S.T)
    jitter = 1e-9
    while True:
        try:
            np.linalg.cholesky(S + jitter * np.eye(m))
            break
        except np.linalg.LinAlgError:
            jitter *= 10
    S = S + jitter * np.eye(m)

    # Kalman gain
    K = Cxz @ np.linalg.inv(S)

    # update state and covariance
    innovation = y - z_pred
    x_upd = x_pred + K @ innovation
    P_upd = P_pred - K @ S @ K.T

    # enforce symmetry
    P_upd = 0.5 * (P_upd + P_upd.T)

    return x_upd, P_upd

#### 5. Main Filter Loop (UKF) #################################################################

# Process noise covariance Q for state [p, v, theta]
Q = np.zeros((n_x, n_x))
Q[3:6, 3:6] = np.eye(3) * (var_imu_f)
Q[6:9, 6:9] = np.eye(3) * (var_imu_w * 0.1)

for k in range(1, imu_f.data.shape[0]):  # start at 1 b/c we have initial prediction from gt
    dt = imu_f.t[k] - imu_f.t[k - 1]

    f_imu = imu_f.data[k]
    w_imu = imu_w.data[k]

    # 1. UKF prediction
    x_pred, P_pred = ukf_predict(x_prev, P_prev, f_imu, w_imu, dt, Q)

    # 2. Measurement updates (GNSS and LIDAR)

    # GNSS
    while gnss_i < len(gnss.t) and gnss.t[gnss_i] <= imu_f.t[k]:
        if abs(gnss.t[gnss_i] - imu_f.t[k]) < 1e-2:
            R_gnss = np.eye(3) * var_gnss
            x_pred, P_pred = ukf_update(x_pred, P_pred, gnss.data[gnss_i], R_gnss)
        gnss_i += 1

    # LIDAR
    while lidar_i < len(lidar.t) and lidar.t[lidar_i] <= imu_f.t[k]:
        if lidar.t[lidar_i] == imu_f.t[k]:
            R_lidar = np.eye(3) * var_lidar
            x_pred, P_pred = ukf_update(x_pred, P_pred, lidar.data[lidar_i], R_lidar)
        lidar_i += 1

    # save estimates
    p_k, v_k, theta_k = vec_to_state(x_pred)
    q_k = Quaternion(axis_angle=theta_k).to_numpy()

    p_est[k] = p_k
    v_est[k] = v_k
    q_est[k] = q_k
    p_cov[k] = P_pred

    # prepare for next iteration
    x_prev = x_pred
    P_prev = P_pred

#### 6. Results and Analysis ###################################################################

################################################################################################
# Now that we have state estimates for all of our sensor data, let's plot the results. This plot
# will show the ground truth and the estimated trajectories on the same plot. Notice that the
# estimated trajectory continues past the ground truth. This is because we will be evaluating
# your estimated poses from the part of the trajectory where you don't have ground truth!
################################################################################################
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

################################################################################################
# We can also plot the error for each of the 6 DOF, with estimates for our uncertainty
# included. The error estimates are in blue, and the uncertainty bounds are red and dashed.
# The uncertainty bounds are +/- 3 standard deviations based on our uncertainty (covariance).
################################################################################################
error_fig, ax = plt.subplots(2, 3)
error_fig.suptitle('Error Plots')
num_gt = gt.p.shape[0]
p_est_euler = []
p_cov_euler_std = []

# Convert estimated quaternions to euler angles
for i in range(len(q_est)):
    qc = Quaternion(*q_est[i, :])
    p_est_euler.append(qc.to_euler())

    # First-order approximation of RPY covariance
    J = rpy_jacobian_axis_angle(qc.to_axis_angle())
    p_cov_euler_std.append(np.sqrt(np.diagonal(J @ p_cov[i, 6:, 6:] @ J.T)))

p_est_euler = np.array(p_est_euler)
p_cov_euler_std = np.array(p_cov_euler_std)

# Get uncertainty estimates from P matrix
p_cov_std = np.sqrt(np.diagonal(p_cov[:, :6, :6], axis1=1, axis2=2))

titles = ['Easting', 'Northing', 'Up', 'Roll', 'Pitch', 'Yaw']
# Clip error values for plotting so extreme outliers do not distort the visualization
pos_clip = 10.0  # meters
angle_clip = 0.5  # radians
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

# IMU timestamps (always present)
plt.plot(imu_f.t, np.zeros_like(imu_f.t), 'k.', markersize=2, label='IMU')

# GNSS timestamps
plt.plot(gnss.t, np.ones_like(gnss.t), 'g.', markersize=4, label='GNSS')

# LIDAR timestamps
plt.plot(lidar.t, 2*np.ones_like(lidar.t), 'r.', markersize=4, label='LIDAR')

plt.yticks([0,1,2], ['IMU','GNSS','LIDAR'])
plt.xlabel("Time [ms]")
plt.grid(True)
plt.legend()
plt.show()


#### 7. Submission #############################################################################

################################################################################################
# Now we can prepare your results for submission to the Coursera platform. Uncomment the
# corresponding lines to prepare a file that will save your position estimates in a format
# that corresponds to what we're expecting on Coursera.
################################################################################################

# Pt. 1 submission
p1_indices = [9000, 9400, 9800, 10200, 10600]
p1_str = ''
for val in p1_indices:
    for i in range(3):
        p1_str += '%.3f ' % (p_est[val, i])
with open('pt1_submission.txt', 'w') as file:
    file.write(p1_str)

# Pt. 2 submission
##p2_indices = [9000, 9400, 9800, 10200, 10600]
##p2_str = ''
##for val in p2_indices:
##    for i in range(3):
##        p2_str += '%.3f ' % (p_est[val, i])
##with open('pt2_submission.txt', 'w') as file:
##    file.write(p2_str)

# Pt. 3 submission
#p3_indices = [6800, 7600, 8400, 9200, 10000]
#p3_str = ''
#for val in p3_indices:
#    for i in range(3):
#        p3_str += '%.3f ' % (p_est[val, i])
#with open('pt3_submission.txt', 'w') as file:
#    file.write(p3_str)
