#!/usr/bin/env python3
"""
Test Scenarios for Bilateral Motor Simulation

This script provides functions to run different test scenarios with the
bilateral motor simulation. It helps demonstrate key concepts of bilateral
control by setting up specific initial conditions and parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from bilateral_motor_sim import BilateralMotorSimulation, Motor

class ScenarioTester:
    """
    Utility to setup and run test scenarios for the bilateral motor simulation.
    """
    
    def __init__(self):
        """Initialize the scenario tester"""
        self.simulation = None
    
    def run_scenario(self, scenario_name):
        """Run a specific test scenario"""
        scenario_functions = {
            'basic': self.scenario_basic,
            'position_tracking': self.scenario_position_tracking,
            'force_reflection': self.scenario_force_reflection,
            'high_stiffness': self.scenario_high_stiffness,
            'low_stiffness': self.scenario_low_stiffness,
            'underdamped': self.scenario_underdamped,
            'overdamped': self.scenario_overdamped
        }
        
        if scenario_name not in scenario_functions:
            print(f"Unknown scenario: {scenario_name}")
            print(f"Available scenarios: {list(scenario_functions.keys())}")
            return
            
        # Create a new simulation
        self.simulation = BilateralMotorSimulation()
        
        # Apply the scenario setup
        scenario_functions[scenario_name]()
        
        # Run the simulation
        print(f"Running scenario: {scenario_name}")
        self.simulation.run()
    
    def scenario_basic(self):
        """Basic scenario with default parameters"""
        # No changes needed - just use default parameters
        print("Basic scenario with default parameters")
        print("- Both motors will try to track each other")
        print("- Try moving the motor position sliders to see tracking behavior")
        print("- Try applying external forces to see force reflection")
    
    def scenario_position_tracking(self):
        """Demonstrate position tracking with one motor at non-zero position"""
        self.simulation.motor1.position = np.pi/2  # Start Motor 1 at 90 degrees
        self.simulation.motor1_slider.set_val(np.pi/2)
        
        print("Position Tracking Scenario")
        print("- Motor 1 starts at 90 degrees")
        print("- Motor 2 will try to track Motor 1")
        print("- Then try moving Motor 2 and observe how Motor 1 follows")
    
    def scenario_force_reflection(self):
        """Demonstrate force reflection by applying external force to one motor"""
        # Set positive external force on Motor 1
        self.simulation.motor1.apply_force(5.0)
        self.simulation.force1_slider.set_val(5.0)
        
        print("Force Reflection Scenario")
        print("- External force applied to Motor 1")
        print("- Observe how Motor 2 responds to maintain synchronization")
        print("- The controllers will generate forces to maintain position tracking")
    
    def scenario_high_stiffness(self):
        """Demonstrate behavior with high stiffness (high Kp)"""
        self.simulation.kp = 30.0
        self.simulation.kp_slider.set_val(30.0)
        
        print("High Stiffness Scenario")
        print("- High proportional gain (Kp = 30)")
        print("- The system will respond quickly to position errors")
        print("- Try applying forces and observe the strong resistance")
    
    def scenario_low_stiffness(self):
        """Demonstrate behavior with low stiffness (low Kp)"""
        self.simulation.kp = 2.0
        self.simulation.kp_slider.set_val(2.0)
        
        print("Low Stiffness Scenario")
        print("- Low proportional gain (Kp = 2)")
        print("- The system will respond slowly to position errors")
        print("- Try applying forces and observe the weak resistance")
    
    def scenario_underdamped(self):
        """Demonstrate underdamped behavior (low Kd)"""
        self.simulation.kp = 15.0
        self.simulation.kd = 0.2
        self.simulation.kp_slider.set_val(15.0)
        self.simulation.kd_slider.set_val(0.2)
        
        # Start with motor positions different to see oscillations
        self.simulation.motor1.position = np.pi/2
        self.simulation.motor1_slider.set_val(np.pi/2)
        
        print("Underdamped Scenario")
        print("- High Kp (15) and low Kd (0.2)")
        print("- The system will oscillate when responding to position changes")
        print("- Try moving the sliders and observe the oscillatory response")
    
    def scenario_overdamped(self):
        """Demonstrate overdamped behavior (high Kd)"""
        self.simulation.kp = 10.0
        self.simulation.kd = 8.0
        self.simulation.kp_slider.set_val(10.0)
        self.simulation.kd_slider.set_val(8.0)
        
        # Start with motor positions different
        self.simulation.motor1.position = np.pi/2
        self.simulation.motor1_slider.set_val(np.pi/2)
        
        print("Overdamped Scenario")
        print("- Moderate Kp (10) and high Kd (8)")
        print("- The system will respond slowly without oscillations")
        print("- Try moving the sliders and observe the sluggish response")

def main():
    """Main function to run test scenarios"""
    import sys
    
    # List of available scenarios
    available_scenarios = [
        'basic', 'position_tracking', 'force_reflection',
        'high_stiffness', 'low_stiffness', 'underdamped', 'overdamped'
    ]
    
    # Get scenario name from command line argument
    if len(sys.argv) > 1:
        scenario_name = sys.argv[1]
    else:
        # If no argument provided, list available scenarios and use 'basic'
        print("Available scenarios:")
        for scenario in available_scenarios:
            print(f"  - {scenario}")
        print("\nUsing 'basic' scenario. To use a different one, run:")
        print("python test_scenarios.py <scenario_name>")
        scenario_name = 'basic'
    
    # Run the selected scenario
    tester = ScenarioTester()
    tester.run_scenario(scenario_name)

if __name__ == "__main__":
    main() 