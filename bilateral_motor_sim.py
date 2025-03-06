#!/usr/bin/env python3
"""
Bilateral Motor Simulation

This script simulates two motors connected by bilateral feedback.
When one motor moves, the other follows using a PD controller.
Forces can be applied to either motor and are displayed as arrows.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

class Motor:
    """
    Simulates a motor with dynamics and control input.
    
    Attributes:
        position: Current angular position (radians)
        velocity: Current angular velocity (radians/s)
        target_position: Desired position for controller
        control_input: Control force/torque applied to the motor
        external_force: External force/torque applied to the motor
    """
    def __init__(self, name, init_position=0.0, inertia=0.05):
        self.name = name
        self.position = init_position
        self.velocity = 0.0
        self.target_position = init_position
        self.target_velocity = 0.0
        self.control_input = 0.0
        self.external_force = 0.0
        self.inertia = inertia  # Motor inertia
        
    def calculate_control(self, kp, kd):
        """Calculate control input using PD controller"""
        position_error = self.target_position - self.position
        velocity_error = self.target_velocity - self.velocity  # Target velocity is zero
        self.control_input = kp * position_error + kd * velocity_error
        return self.control_input
    
    def apply_force(self, force):
        """Apply external force to the motor"""
        self.external_force = force
        
    def update(self, dt, kp, kd):
        """Update motor state for one simulation step"""
        # Calculate control force
        self.calculate_control(kp, kd)
        
        # Total force is control plus external
        total_force = self.control_input + self.external_force
        
        # Update velocity and position based on dynamics
        acceleration = total_force / self.inertia

        if abs(self.position) < np.pi or total_force*self.position < 0:
            self.velocity += acceleration * dt
            self.position += self.velocity * dt
        
        return self.position, self.velocity

class BilateralMotorSimulation:
    """
    Simulation of two motors with bilateral feedback.
    
    Handles the simulation loop, visualization, and user interface.
    """
    def __init__(self):
        # Create motors
        self.motor1 = Motor("Motor 1", inertia=0.20)
        self.motor2 = Motor("Motor 2", inertia=0.05)
        
        # Simulation parameters
        self.dt = 0.01  # Time step
        self.time = 0.0  # Current simulation time
        self.history_length = 500  # Number of time steps to plot
        
        # Controller parameters
        self.kp = 100.0  # Proportional gain
        self.kd = 1.0   # Derivative gain
        
        # Initialize history arrays
        self.time_history = np.zeros(self.history_length)
        self.motor1_position_history = np.zeros(self.history_length)
        self.motor2_position_history = np.zeros(self.history_length)
        self.motor1_force_history = np.zeros(self.history_length)
        self.motor2_force_history = np.zeros(self.history_length)
        
        # Initialize text annotation objects
        self.text_motor1 = None
        self.text_motor2 = None
        
        # Slider control flags
        self.motor1_slider_active = False
        self.motor2_slider_active = False
        
        # Setup plot and UI
        self.setup_plot()
        
    def setup_plot(self):
        """Set up matplotlib figure and UI controls"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plt.subplots_adjust(bottom=0.3)  # Increase bottom margin for sliders
        
        # Position plot
        self.ax1.set_title('Motor Positions')
        self.ax1.set_ylabel('Position (rad)')
        # self.ax1.set_xlabel('Time (s)')
        self.ax1.grid(True)
        
        # Force plot
        self.ax2.set_title('Motor Forces')
        self.ax2.set_ylabel('Force')
        self.ax2.set_xlabel('Time (s)')
        self.ax2.grid(True)
        
        # Position lines
        self.line_motor1_pos, = self.ax1.plot([], [], 'b-', label='Motor 1')
        self.line_motor2_pos, = self.ax1.plot([], [], 'r-', label='Motor 2')
        self.ax1.legend()
        
        # Force lines
        self.line_motor1_force, = self.ax2.plot([], [], 'b-', label='Motor 1')
        self.line_motor2_force, = self.ax2.plot([], [], 'r-', label='Motor 2')
        self.ax2.legend()
        
        # Add sliders
        slider_color = 'lightgoldenrodyellow'
        
        # Motor position sliders
        self.ax_motor1_slider = plt.axes([0.2, 0.22, 0.65, 0.02], facecolor=slider_color)
        self.ax_motor2_slider = plt.axes([0.2, 0.19, 0.65, 0.02], facecolor=slider_color)

        # External force sliders
        self.ax_force1_slider = plt.axes([0.2, 0.16, 0.65, 0.02], facecolor=slider_color)
        self.ax_force2_slider = plt.axes([0.2, 0.13, 0.65, 0.02], facecolor=slider_color)
        
        # Controller gain sliders
        self.ax_kp_slider = plt.axes([0.2, 0.10, 0.65, 0.02], facecolor=slider_color)
        self.ax_kd_slider = plt.axes([0.2, 0.07, 0.65, 0.02], facecolor=slider_color)
        
        # Inertia sliders
        self.ax_inertia1_slider = plt.axes([0.2, 0.04, 0.65, 0.02], facecolor=slider_color)
        self.ax_inertia2_slider = plt.axes([0.2, 0.01, 0.65, 0.02], facecolor=slider_color)
        
        # Create sliders
        self.motor1_slider = Slider(self.ax_motor1_slider, 'Motor 1 Position', -np.pi, np.pi, 
                                    valinit=self.motor1.position)
        self.motor2_slider = Slider(self.ax_motor2_slider, 'Motor 2 Position', -np.pi, np.pi, 
                                    valinit=self.motor2.position)
        self.kp_slider = Slider(self.ax_kp_slider, 'Kp', 1, 1000.0, valinit=self.kp)
        self.kd_slider = Slider(self.ax_kd_slider, 'Kd', -10, 10.0, valinit=self.kd)
        self.force1_slider = Slider(self.ax_force1_slider, 'Force on Motor 1', -100.0, 100.0, valinit=0.0)
        self.force2_slider = Slider(self.ax_force2_slider, 'Force on Motor 2', -100.0, 100.0, valinit=0.0)
        self.inertia1_slider = Slider(self.ax_inertia1_slider, 'Motor 1 Inertia', 0.01, 1.0, 
                                    valinit=self.motor1.inertia)
        self.inertia2_slider = Slider(self.ax_inertia2_slider, 'Motor 2 Inertia', 0.01, 1.0, 
                                    valinit=self.motor2.inertia)
        
        # Connect callbacks
        self.motor1_slider.on_changed(self.update_motor1_slider)
        self.motor2_slider.on_changed(self.update_motor2_slider)
        self.kp_slider.on_changed(self.update_gains)
        self.kd_slider.on_changed(self.update_gains)
        self.force1_slider.on_changed(self.update_forces)
        self.force2_slider.on_changed(self.update_forces)
        self.inertia1_slider.on_changed(self.update_inertias)
        self.inertia2_slider.on_changed(self.update_inertias)
        
        # Connect mouse press/release events for sliders
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        
        # Add a legend for forces
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='Motor 1'),
            Line2D([0], [0], color='red', lw=2, label='Motor 2')
        ]
        self.ax1.legend(handles=legend_elements, loc='upper right')
        self.ax2.legend(handles=legend_elements, loc='upper right')
    
    def on_mouse_press(self, event):
        """Handle mouse press events to detect slider interaction"""
        if event.inaxes == self.ax_motor1_slider:
            self.motor1_slider_active = True
            print("Motor 1 slider active")
        elif event.inaxes == self.ax_motor2_slider:
            self.motor2_slider_active = True
            print("Motor 2 slider active")
    
    def on_mouse_release(self, event):
        """Handle mouse release events to detect end of slider interaction"""
        self.motor1_slider_active = False
        self.motor2_slider_active = False
        print("Motor sliders inactive")
        
    def update_motor1_slider(self, val):
        """Callback for Motor 1 slider"""
        self.motor1.position = val
        self.motor1.velocity = 0  # Reset velocity when direct control is used
        
    def update_motor2_slider(self, val):
        """Callback for Motor 2 slider"""
        self.motor2.position = val
        self.motor2.velocity = 0  # Reset velocity when direct control is used
        
    def update_gains(self, val):
        """Callback for gain sliders"""
        self.kp = self.kp_slider.val
        self.kd = self.kd_slider.val
        
    def update_forces(self, val):
        """Callback for force sliders"""
        self.motor1.apply_force(self.force1_slider.val)
        self.motor2.apply_force(self.force2_slider.val)
        
    def update_inertias(self, val):
        """Callback for inertia sliders"""
        self.motor1.inertia = self.inertia1_slider.val
        self.motor2.inertia = self.inertia2_slider.val
        
    def update_simulation(self):
        """Run one simulation step updating both motors"""
        # Connect motors through bilateral feedback
        self.motor1.target_position = self.motor2.position
        self.motor2.target_position = self.motor1.position
        self.motor1.target_velocity = self.motor2.velocity
        self.motor2.target_velocity = self.motor1.velocity

        # Update slider positions to match motor positions if not being controlled
        # Or update motor target positions if sliders are active
        if self.motor1_slider_active:
            self.motor1.target_position = self.motor1_slider.val
        else:
            self.motor1_slider.set_val(self.motor1.position)
            
        if self.motor2_slider_active:
            self.motor2.target_position = self.motor2_slider.val
        else:
            self.motor2_slider.set_val(self.motor2.position)
        
        # Update motor states
        self.motor1.update(self.dt, self.kp, self.kd)
        self.motor2.update(self.dt, self.kp, self.kd)
        
        # Update time
        self.time += self.dt
        
        # Shift history arrays
        self.time_history = np.roll(self.time_history, -1)
        self.motor1_position_history = np.roll(self.motor1_position_history, -1)
        self.motor2_position_history = np.roll(self.motor2_position_history, -1)
        self.motor1_force_history = np.roll(self.motor1_force_history, -1)
        self.motor2_force_history = np.roll(self.motor2_force_history, -1)
        
        # Add new data points
        self.time_history[-1] = self.time
        self.motor1_position_history[-1] = self.motor1.position
        self.motor2_position_history[-1] = self.motor2.position
        self.motor1_force_history[-1] = self.motor1.control_input + self.motor1.external_force
        self.motor2_force_history[-1] = self.motor2.control_input + self.motor2.external_force
        
    def update_plot(self):
        """Update plot lines and text annotations"""
        # Update position lines
        self.line_motor1_pos.set_data(self.time_history, self.motor1_position_history)
        self.line_motor2_pos.set_data(self.time_history, self.motor2_position_history)
        
        # Update force lines
        self.line_motor1_force.set_data(self.time_history, self.motor1_force_history)
        self.line_motor2_force.set_data(self.time_history, self.motor2_force_history)
        
        # Current time for text placement
        current_time = self.time_history[-1]
        
        # Update control and external force values
        control_force1 = self.motor1.control_input
        control_force2 = self.motor2.control_input
        external_force1 = self.motor1.external_force
        external_force2 = self.motor2.external_force
        
        # Show force values as text annotations
        textstr1 = f"Control: {control_force1:.2f}\nExternal: {external_force1:.2f}"
        textstr2 = f"Control: {control_force2:.2f}\nExternal: {external_force2:.2f}"
        
        # Remove old text annotations if they exist
        if hasattr(self, 'text_motor1') and self.text_motor1 is not None:
            self.text_motor1.remove()
        if hasattr(self, 'text_motor2') and self.text_motor2 is not None:
            self.text_motor2.remove()
            
        # Add new text annotations
        self.text_motor1 = self.ax1.text(current_time - 0.5, self.motor1.position + 0.2, 
                                         textstr1, bbox=dict(facecolor='white', alpha=0.5))
        self.text_motor2 = self.ax1.text(current_time - 0.5, self.motor2.position - 0.5, 
                                         textstr2, bbox=dict(facecolor='white', alpha=0.5))
        
        # Adjust plot limits
        self.ax1.set_xlim(self.time_history[0], self.time_history[-1] + self.dt)
        self.ax1.set_ylim(min(min(self.motor1_position_history), min(self.motor2_position_history)) - 1.0,
                          max(max(self.motor1_position_history), max(self.motor2_position_history)) + 1.0)
        
        self.ax2.set_xlim(self.time_history[0], self.time_history[-1] + self.dt)
        self.ax2.set_ylim(min(min(self.motor1_force_history), min(self.motor2_force_history)) - 1.0,
                          max(max(self.motor1_force_history), max(self.motor2_force_history)) + 1.0)
        
        self.fig.canvas.draw_idle()
        
    def run(self):
        """Run the simulation"""
        # Initialize the animation
        timer = self.fig.canvas.new_timer(interval=self.dt * 1000)  # interval in milliseconds
        timer.add_callback(self.animation_step)
        timer.start()
        
        plt.show()
        
    def animation_step(self):
        """Callback for animation - one step of simulation and update plot"""
        self.update_simulation()
        self.update_plot()

def main():
    """Main function to run the simulation"""
    simulation = BilateralMotorSimulation()
    simulation.run()

if __name__ == "__main__":
    main() 