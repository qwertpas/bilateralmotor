import numpy as np
import matplotlib.pyplot as plt

# System parameters
k_r = 14000      # Robot spring constant
b_r = 500        # Robot damping constant
k_h = 14000      # Human spring constant
b_h = 500        # Human damping constant
C_fb = 1         # Feedback coupling coefficient
m_h = 3.5        # Human mass
m_r = 8.75       # Robot mass
F_robot_saturation = 100
F_human_saturation = 50
delay_r2h = 10   # Delay from robot to human (time steps)
delay_h2r = 10   # Delay from human to robot (time steps)
dt = 0.001       # Time step (seconds)

# Simulation parameters
sim_time = 2
sim_steps = int(sim_time / dt)
time = np.arange(0, sim_time, dt)

# Initialize arrays
x_r = np.zeros(sim_steps)
x_r_dot = np.zeros(sim_steps)
x_h = np.zeros(sim_steps)
x_h_dot = np.zeros(sim_steps)
F_robot = np.zeros(sim_steps)
F_human = np.zeros(sim_steps)

# Step input - robot desired position
reference = np.zeros(sim_steps)
step_start_time = 0.1  # Start step at 0.1 seconds
step_amplitude = 0.01  # 1 cm step
step_start_index = int(step_start_time / dt)
reference[step_start_index:] = step_amplitude

# Main simulation loop
for i in range(max(delay_r2h, delay_h2r), sim_steps):
    
    # Get delayed signals
    x_h_delayed = x_h[i-delay_r2h] if i >= delay_r2h else 0
    x_h_dot_delayed = x_h_dot[i-delay_r2h] if i >= delay_r2h else 0
    F_robot_delayed = F_robot[i-delay_h2r] if i >= delay_h2r else 0
    
    # Calculate forces
    # Robot force: PD controller tracking human position with delay
    F_robot[i] = k_r*(x_h_delayed - x_r[i-1]) + b_r*(x_h_dot_delayed - x_r_dot[i-1])
    
    # Human force: interaction with robot + delayed robot force feedback
    F_human[i] = k_h*(reference[i] - x_h[i-1]) + b_h*(0 - x_h_dot[i-1]) - C_fb * F_robot_delayed
    
    # Add step input force to robot (alternative: could modify desired position)
    F_robot[i] += 0

    F_robot[i] = np.clip(F_robot[i], -F_robot_saturation, F_robot_saturation)
    F_human[i] = np.clip(F_human[i], -F_human_saturation, F_human_saturation)
    
    # Calculate accelerations
    x_h_ddot = F_human[i] / m_h
    x_r_ddot = F_robot[i] / m_r
    
    # Semi-implicit Euler integration
    # Update velocities first using current accelerations
    x_h_dot[i] = x_h_dot[i-1] + x_h_ddot * dt
    x_r_dot[i] = x_r_dot[i-1] + x_r_ddot * dt
    
    # Update positions using the NEW velocities (semi-implicit)
    x_h[i] = x_h[i-1] + x_h_dot[i] * dt
    x_r[i] = x_r[i-1] + x_r_dot[i] * dt

# Plot step response
plt.figure(figsize=(12, 8))

# Position response
plt.subplot(2, 1, 1)
plt.plot(time, x_r*1000, 'b-', label='Robot Position', linewidth=2)
plt.plot(time, x_h*1000, 'r-', label='Human Position', linewidth=2)
plt.plot(time, reference*1000, 'k--', label='Desired Position', linewidth=1)
plt.ylabel('Position (mm)')
plt.title('Bilateral Haptic System - Step Response')
plt.legend()
plt.grid(True, alpha=0.3)

# Force response
plt.subplot(2, 1, 2)
plt.plot(time, F_robot, 'b-', label='Robot Force', linewidth=2)
plt.plot(time, F_human, 'r-', label='Human Force', linewidth=2)
plt.ylabel('Force (N)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()