import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise

# Function to apply a low-pass filter to the accelerometer data
def low_pass_filter(data, alpha=0.5):
    filtered_data = np.zeros_like(data)
    filtered_data[0] = data[0]
    for i in range(1, len(data)):
        filtered_data[i] = alpha * data[i] + (1 - alpha) * filtered_data[i-1]
    return filtered_data

# Function to scale IMU data
def scale_imu_data(df):
    # Convert acceleration to m/s^2
    accel_scale = 9.81 / 16384.0  # Adjust based on your IMU's sensitivity
    df[['Acc_x', 'Acc_y', 'Acc_z']] *= accel_scale
    
    # Convert angular velocity to rad/s
    gyro_scale = np.pi / (180 * 16.4)  # Adjust based on your IMU's sensitivity
    df[['Gyro_x', 'Gyro_y', 'Gyro_z']] *= gyro_scale
    
    return df

# Function to multiply two quaternions
def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

# Function to convert quaternion to a measurement model (accelerometer and magnetometer)
def quat_to_acc_mag(q):
    r = R.from_quat(q)  # Create rotation object from quaternion
    acc = r.apply([0, 0, -1])  # Gravity direction
    mag = r.apply([1, 0, 0])   # Magnetic north direction
    return np.hstack([acc, mag])

# Function to compute the Jacobian matrix of the measurement model
def jacobian(q):
    q0, q1, q2, q3 = q
    H = np.array([
        [2*q2, -2*q3,  2*q0, -2*q1],
        [2*q3,  2*q2,  2*q1,  2*q0],
        [-4*q1, -4*q0,  0,  0],
        [-2*q2,  2*q3, -2*q0,  2*q1],
        [-2*q3, -2*q2,  2*q1,  2*q0],
        [0,  0,  -4*q2, -4*q3]
    ])
    return H

# Function to convert angular velocity to a skew-symmetric matrix
def quat_to_mat(omega):
    return np.array([
        [0, -omega[0], -omega[1], -omega[2]],
        [omega[0], 0, omega[2], -omega[1]],
        [omega[1], -omega[2], 0, omega[0]],
        [omega[2], omega[1], -omega[0], 0]
    ])

# Function to implement the Extended Kalman Filter (EKF)
def extended_kalman_filter(gyro, acc, mag, dt):
    # Initialize state and covariance
    ekf = ExtendedKalmanFilter(dim_x=10, dim_z=6)
    
    ekf.F = np.eye(10)
    ekf.F[0:3, 3:6] = np.eye(3) * dt
    
    ekf.H = np.zeros((6, 10))
    ekf.H[0:3, 0:3] = np.eye(3)
    ekf.H[3:6, 6:10] = np.eye(4)[:3, :]  # Corrected to match dimensions
    
    ekf.R = np.eye(6) * 0.1
    ekf.Q = np.eye(10) * 0.1  # Corrected to match dimensions of P
    ekf.x = np.zeros(10)
    ekf.P = np.eye(10)
    
    positions = []
    velocity = np.array([0.0, 0.0, 0.0])
    position = np.array([0.0, 0.0, 0.0])
    q = np.array([1, 0, 0, 0])
    
    for i in range(1, len(gyro)):
        # Predict step
        omega = gyro[i]
        omega_quat = np.array([0, omega[0], omega[1], omega[2]])
        q_dot = 0.5 * quat_multiply(q, omega_quat)
        q = q + q_dot * dt
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            q = np.array([1, 0, 0, 0])  # Reset to identity quaternion if norm is zero
        else:
            q = q / q_norm  # Normalize quaternion
        
        F = np.eye(4) + 0.5 * dt * quat_to_mat(omega)
        ekf.F[6:10, 6:10] = F
        ekf.predict()
        
        # Update step (using accelerometer and magnetometer)
        z = np.hstack([acc[i], mag[i]])  # 6D measurement vector
        h = quat_to_acc_mag(q)  # 6D predicted measurement
        y = z - h
        H = np.zeros((6, 10))
        H[:, 6:10] = jacobian(q)  # Corrected to match dimensions
        S = H @ ekf.P @ H.T + ekf.R
        K = ekf.P @ H.T @ np.linalg.inv(S)
        
        ekf.x = ekf.x + K @ y
        ekf.P = (np.eye(10) - K @ H) @ ekf.P
        q = ekf.x[6:10]
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            q = np.array([1, 0, 0, 0])  # Reset to identity quaternion if norm is zero
        else:
            q = q / q_norm  # Normalize quaternion
        
        # Update velocity and position
        try:
            r = R.from_quat(q)
        except ValueError as e:
            st.error(f"Error creating rotation from quaternion: {q}, norm: {q_norm}")
            q = np.array([1, 0, 0, 0])  # Reset to identity quaternion if error occurs
            r = R.from_quat(q)
        
        acc_world = r.apply(acc[i]) - np.array([0, 0, 9.81])  # Remove gravity
        velocity += acc_world * dt
        position += velocity * dt
        positions.append(position.copy())
    
    return np.array(positions)

# Streamlit App
def main():
    st.title("Device Trajectory from IMU Data")
    uploaded_file = st.file_uploader("Upload your IMU data CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("IMU Data:")
        st.write(df.head())

        # Always scale IMU data
        df = scale_imu_data(df)

        # Extract gyroscope, accelerometer, and magnetometer data
        gyro = df[['Gyro_x', 'Gyro_y', 'Gyro_z']].values
        acc = df[['Acc_x', 'Acc_y', 'Acc_z']].values
        mag = df[['Mag_x', 'Mag_y', 'Mag_z']].values
        dt = 1 / 100  # Assuming 100Hz sampling rate

        # Apply Low-Pass Filter
        acc[:, 0] = low_pass_filter(acc[:, 0])
        acc[:, 1] = low_pass_filter(acc[:, 1])
        acc[:, 2] = low_pass_filter(acc[:, 2])

        # Estimate Trajectory using EKF
        trajectory = extended_kalman_filter(gyro, acc, mag, dt)

        # Plot the Trajectory
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='r', label='Position Path')
        ax.set_xlabel('X Position', labelpad=10)
        ax.set_ylabel('Y Position', labelpad=10)
        ax.set_zlabel('Z Position', labelpad=8)
        ax.set_title('Device Trajectory')
        ax.legend()

        st.pyplot(fig)

if __name__ == "__main__":
    main()