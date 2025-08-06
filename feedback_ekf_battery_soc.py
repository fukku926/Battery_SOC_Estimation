"""
Feedback-based Extended Kalman Filter for Battery State of Charge Estimation
=======================================================================

This implementation is based on the research paper and provides a comprehensive
Python version that can be easily tested on Google Colab.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class BatteryModel:
    """
    Thevenin equivalent circuit model for lithium-ion battery
    """
    
    def __init__(self, capacity: float = 1.5):
        """
        Initialize battery model parameters
        
        Args:
            capacity: Battery capacity in Ah
        """
        self.capacity = capacity  # Ah
        self.eta = 1.0  # Coulombic efficiency
        
    def get_parameters(self, soc: float) -> Tuple[float, float, float, float]:
        """
        Get battery parameters based on SoC
        
        Args:
            soc: State of charge (0-1)
            
        Returns:
            Tuple of (UOC, Ro, Rp, Cp)
        """
        # Open circuit voltage (V)
        UOC = (3.44003 + 1.71448 * soc - 3.51247 * soc**2 + 
               5.70868 * soc**3 - 5.06869 * soc**4 + 1.86699 * soc**5)
        
        # Internal resistance (Ohm)
        Ro = (0.04916 + 1.19552 * soc - 6.25333 * soc**2 + 
              14.24181 * soc**3 - 13.93388 * soc**4 + 2.553 * soc**5 + 
              4.16285 * soc**6 - 1.8713 * soc**7)
        
        # Polarization resistance (Ohm)
        Rp = (0.02346 - 0.10537 * soc + 1.1371 * soc**2 - 
              4.55188 * soc**3 + 8.26827 * soc**4 - 6.93032 * soc**5 + 
              2.1787 * soc**6)
        
        # Polarization capacitance (F)
        Cp = (203.1404 + 3522.78847 * soc - 31392.66753 * soc**2 + 
              122406.91269 * soc**3 - 227590.94382 * soc**4 + 
              198281.56406 * soc**5 - 65171.90395 * soc**6)
        
        return UOC, Ro, Rp, Cp

class FeedbackEKF:
    """
    Feedback-based Extended Kalman Filter for battery SoC estimation
    """
    
    def __init__(self, capacity: float = 1.5, dt: float = 1.0):
        """
        Initialize the Feedback EKF
        
        Args:
            capacity: Battery capacity in Ah
            dt: Sampling time in seconds
        """
        self.capacity = capacity
        self.dt = dt
        self.battery = BatteryModel(capacity)
        
        # State vector: [SoC, Up] where Up is polarization voltage
        self.n_states = 2
        
        # Initialize state and covariance
        self.x = np.array([1.0, 0.0])  # Initial state [SoC, Up]
        self.P = np.diag([1e-8, 1e-6])  # Initial covariance
        
        # Process noise covariance
        self.Q = np.diag([4e-9, 1e-8])  # [SoC noise, Up noise]
        
        # Measurement noise covariance
        self.R = 1e-6
        
        # Feedback parameters
        self.alpha = 0.1  # Feedback gain
        self.beta = 0.05  # Adaptive parameter
        
        # History for plotting
        self.soc_history = []
        self.up_history = []
        self.voltage_history = []
        self.gain_history = []
        
    def predict(self, current: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step of the EKF
        
        Args:
            current: Battery current (A)
            
        Returns:
            Tuple of (predicted_state, predicted_covariance)
        """
        # Get current SoC
        soc = self.x[0]
        
        # Get battery parameters
        _, _, Rp, Cp = self.battery.get_parameters(soc)
        tao = Rp * Cp
        
        # State transition matrix
        A = np.array([[1, 0],
                     [0, np.exp(-self.dt / tao)]])
        
        # Input control matrix
        B = np.array([[-self.dt / (self.capacity * 3600)],
                     [Rp * (1 - np.exp(-self.dt / tao))]])
        
        # Predict state
        x_pred = A @ self.x + B.flatten() * current
        
        # Predict covariance
        P_pred = A @ self.P @ A.T + self.Q
        
        return x_pred, P_pred
    
    def update(self, voltage_measured: float, current: float, 
               x_pred: np.ndarray, P_pred: np.ndarray) -> None:
        """
        Update step of the EKF with feedback mechanism
        
        Args:
            voltage_measured: Measured battery voltage (V)
            current: Battery current (A)
            x_pred: Predicted state
            P_pred: Predicted covariance
        """
        # Get battery parameters for predicted state
        soc_pred = x_pred[0]
        UOC_pred, Ro_pred, Rp_pred, Cp_pred = self.battery.get_parameters(soc_pred)
        
        # Predicted voltage
        voltage_pred = UOC_pred - x_pred[1] - current * Ro_pred
        
        # Innovation (measurement residual)
        innovation = voltage_measured - voltage_pred
        
        # Jacobian matrix (linearization)
        # dV/dSoC = dUOC/dSoC
        dUOC_dSoC = (1.71448 - 2 * 3.51247 * soc_pred + 
                     3 * 5.70868 * soc_pred**2 - 
                     4 * 5.06869 * soc_pred**3 + 
                     5 * 1.86699 * soc_pred**4)
        
        # dV/dUp = -1
        C = np.array([dUOC_dSoC, -1])
        
        # Kalman gain
        S = C @ P_pred @ C.T + self.R
        K = P_pred @ C.T / S
        
        # Feedback-based adaptive gain
        feedback_gain = self.alpha * np.abs(innovation) + self.beta
        K_adaptive = K * (1 + feedback_gain)
        
        # Update state
        self.x = x_pred + K_adaptive * innovation
        
        # Update covariance
        self.P = P_pred - K_adaptive @ C @ P_pred
        
        # Store history
        self.soc_history.append(self.x[0])
        self.up_history.append(self.x[1])
        self.voltage_history.append(voltage_measured)
        self.gain_history.append(np.linalg.norm(K_adaptive))
    
    def step(self, voltage_measured: float, current: float) -> None:
        """
        Complete EKF step (predict + update)
        
        Args:
            voltage_measured: Measured battery voltage (V)
            current: Battery current (A)
        """
        x_pred, P_pred = self.predict(current)
        self.update(voltage_measured, current, x_pred, P_pred)

class BatterySimulator:
    """
    Battery simulator for testing the EKF
    """
    
    def __init__(self, capacity: float = 1.5, dt: float = 1.0):
        """
        Initialize battery simulator
        
        Args:
            capacity: Battery capacity in Ah
            dt: Sampling time in seconds
        """
        self.capacity = capacity
        self.dt = dt
        self.battery = BatteryModel(capacity)
        
        # Real state
        self.soc_real = 1.0
        self.up_real = 0.0
        
        # History
        self.soc_real_history = []
        self.voltage_real_history = []
        self.current_history = []
        
    def step(self, current: float, add_noise: bool = True) -> Tuple[float, float]:
        """
        Simulate battery for one step
        
        Args:
            current: Battery current (A)
            add_noise: Whether to add measurement noise
            
        Returns:
            Tuple of (voltage, soc_real)
        """
        # Update real SoC
        self.soc_real -= current * self.dt / (self.capacity * 3600)
        self.soc_real = np.clip(self.soc_real, 0, 1)
        
        # Get battery parameters
        UOC, Ro, Rp, Cp = self.battery.get_parameters(self.soc_real)
        tao = Rp * Cp
        
        # Update polarization voltage
        self.up_real = (self.up_real * np.exp(-self.dt / tao) + 
                       Rp * current * (1 - np.exp(-self.dt / tao)))
        
        # Calculate real voltage
        voltage_real = UOC - self.up_real - current * Ro
        
        # Add measurement noise
        if add_noise:
            voltage_noise = np.random.normal(0, np.sqrt(1e-6))
            current_noise = np.random.normal(0, 0.01 * self.capacity)
            voltage_real += voltage_noise
            current += current_noise
        
        # Store history
        self.soc_real_history.append(self.soc_real)
        self.voltage_real_history.append(voltage_real)
        self.current_history.append(current)
        
        return voltage_real, self.soc_real

def generate_bbst_condition(duration: int = 5000) -> np.ndarray:
    """
    Generate BBDST (Beijing Bus Dynamic Street Test) working condition
    
    Args:
        duration: Duration in seconds
        
    Returns:
        Current profile array
    """
    # Create a realistic driving cycle
    t = np.linspace(0, duration, duration)
    
    # Base frequency components
    base_current = 0.5 * np.sin(2 * np.pi * 0.001 * t) + \
                   0.3 * np.sin(2 * np.pi * 0.002 * t) + \
                   0.2 * np.sin(2 * np.pi * 0.005 * t)
    
    # Add acceleration/deceleration events
    acceleration_events = np.zeros_like(t)
    for i in range(0, duration, 300):  # Every 5 minutes
        if i + 60 < duration:
            acceleration_events[i:i+60] = 2.0 * np.sin(np.pi * np.linspace(0, 1, 60))
    
    # Add regenerative braking
    braking_events = np.zeros_like(t)
    for i in range(150, duration, 300):  # Every 5 minutes, offset by 2.5 minutes
        if i + 30 < duration:
            braking_events[i:i+30] = -1.5 * np.sin(np.pi * np.linspace(0, 1, 30))
    
    # Combine all components
    current_profile = base_current + acceleration_events + braking_events
    
    # Add some random variations
    noise = np.random.normal(0, 0.1, duration)
    current_profile += noise
    
    # Ensure current is within reasonable bounds
    current_profile = np.clip(current_profile, -3.0, 3.0)
    
    return current_profile

def generate_constant_current(duration: int = 5000, current: float = 1.0) -> np.ndarray:
    """
    Generate constant current profile
    
    Args:
        duration: Duration in seconds
        current: Constant current value (A)
        
    Returns:
        Current profile array
    """
    return np.full(duration, current)

def run_simulation(ekf: FeedbackEKF, simulator: BatterySimulator, 
                  current_profile: np.ndarray, 
                  method_name: str = "Feedback EKF") -> dict:
    """
    Run the complete simulation
    
    Args:
        ekf: Feedback EKF instance
        simulator: Battery simulator instance
        current_profile: Current profile array
        method_name: Name of the estimation method
        
    Returns:
        Dictionary with results
    """
    duration = len(current_profile)
    
    # Initialize arrays for storing results
    soc_estimated = []
    soc_real = []
    voltage_measured = []
    errors = []
    
    print(f"Running {method_name} simulation...")
    
    for i, current in enumerate(current_profile):
        # Simulate battery
        voltage, soc_real_val = simulator.step(current)
        
        # Update EKF
        ekf.step(voltage, current)
        
        # Store results
        soc_estimated.append(ekf.x[0])
        soc_real.append(soc_real_val)
        voltage_measured.append(voltage)
        errors.append(soc_real_val - ekf.x[0])
        
        # Progress indicator
        if i % 1000 == 0:
            print(f"Progress: {i}/{duration} ({100*i/duration:.1f}%)")
    
    # Calculate statistics
    errors = np.array(errors)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    rmse = np.sqrt(np.mean(errors**2))
    max_error = np.max(np.abs(errors))
    
    results = {
        'soc_estimated': np.array(soc_estimated),
        'soc_real': np.array(soc_real),
        'voltage_measured': np.array(voltage_measured),
        'current_profile': current_profile,
        'errors': errors,
        'mean_error': mean_error,
        'std_error': std_error,
        'rmse': rmse,
        'max_error': max_error,
        'method_name': method_name
    }
    
    print(f"{method_name} Results:")
    print(f"  Mean Error: {mean_error:.6f}")
    print(f"  Std Error: {std_error:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Max Error: {max_error:.6f}")
    
    return results

def plot_results(results: dict, save_plot: bool = False) -> None:
    """
    Plot the simulation results
    
    Args:
        results: Results dictionary from run_simulation
        save_plot: Whether to save the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    t = np.arange(len(results['soc_estimated']))
    
    # Plot 1: SoC comparison
    axes[0].plot(t, results['soc_real'], 'b-', linewidth=2, label='Real SoC')
    axes[0].plot(t, results['soc_estimated'], 'r--', linewidth=2, label='Estimated SoC')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('State of Charge')
    axes[0].set_title(f'{results["method_name"]} - SoC Estimation')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Voltage
    axes[1].plot(t, results['voltage_measured'], 'g-', linewidth=2)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Voltage (V)')
    axes[1].set_title('Battery Voltage')
    axes[1].grid(True)
    
    # Plot 3: Current and Error
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(t, results['current_profile'], 'k-', linewidth=1, label='Current')
    line2 = ax3_twin.plot(t, results['errors'], 'm-', linewidth=1, label='Error')
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Current (A)', color='k')
    ax3_twin.set_ylabel('Error', color='m')
    ax3.set_title('Current Profile and Estimation Error')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right')
    
    ax3.grid(True)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(f'{results["method_name"].replace(" ", "_")}_results.png', 
                   dpi=300, bbox_inches='tight')
    
    plt.show()

def compare_methods(current_profile: np.ndarray, duration: int = 5000) -> dict:
    """
    Compare different estimation methods
    
    Args:
        current_profile: Current profile array
        duration: Duration in seconds
        
    Returns:
        Dictionary with comparison results
    """
    print("Comparing estimation methods...")
    
    # Method 1: Feedback EKF
    ekf_feedback = FeedbackEKF(capacity=1.5, dt=1.0)
    simulator1 = BatterySimulator(capacity=1.5, dt=1.0)
    results_feedback = run_simulation(ekf_feedback, simulator1, current_profile, "Feedback EKF")
    
    # Method 2: Standard EKF (no feedback)
    class StandardEKF(FeedbackEKF):
        def update(self, voltage_measured: float, current: float, 
                  x_pred: np.ndarray, P_pred: np.ndarray) -> None:
            """Standard EKF update without feedback"""
            soc_pred = x_pred[0]
            UOC_pred, Ro_pred, Rp_pred, Cp_pred = self.battery.get_parameters(soc_pred)
            voltage_pred = UOC_pred - x_pred[1] - current * Ro_pred
            innovation = voltage_measured - voltage_pred
            
            dUOC_dSoC = (1.71448 - 2 * 3.51247 * soc_pred + 
                         3 * 5.70868 * soc_pred**2 - 
                         4 * 5.06869 * soc_pred**3 + 
                         5 * 1.86699 * soc_pred**4)
            C = np.array([dUOC_dSoC, -1])
            
            S = C @ P_pred @ C.T + self.R
            K = P_pred @ C.T / S
            
            self.x = x_pred + K * innovation
            self.P = P_pred - K @ C @ P_pred
            
            self.soc_history.append(self.x[0])
            self.up_history.append(self.x[1])
            self.voltage_history.append(voltage_measured)
            self.gain_history.append(np.linalg.norm(K))
    
    ekf_standard = StandardEKF(capacity=1.5, dt=1.0)
    simulator2 = BatterySimulator(capacity=1.5, dt=1.0)
    results_standard = run_simulation(ekf_standard, simulator2, current_profile, "Standard EKF")
    
    # Method 3: Coulomb Counting (Ampere-hour method)
    class CoulombCounter:
        def __init__(self, capacity: float = 1.5, dt: float = 1.0):
            self.capacity = capacity
            self.dt = dt
            self.soc = 1.0
            self.soc_history = []
            
        def step(self, current: float) -> float:
            self.soc -= current * self.dt / (self.capacity * 3600)
            self.soc = np.clip(self.soc, 0, 1)
            self.soc_history.append(self.soc)
            return self.soc
    
    cc = CoulombCounter(capacity=1.5, dt=1.0)
    simulator3 = BatterySimulator(capacity=1.5, dt=1.0)
    
    soc_cc = []
    soc_real_cc = []
    errors_cc = []
    
    for i, current in enumerate(current_profile):
        voltage, soc_real_val = simulator3.step(current)
        soc_est = cc.step(current)
        
        soc_cc.append(soc_est)
        soc_real_cc.append(soc_real_val)
        errors_cc.append(soc_real_val - soc_est)
    
    errors_cc = np.array(errors_cc)
    results_cc = {
        'soc_estimated': np.array(soc_cc),
        'soc_real': np.array(soc_real_cc),
        'voltage_measured': np.array(simulator3.voltage_real_history),
        'current_profile': current_profile,
        'errors': errors_cc,
        'mean_error': np.mean(errors_cc),
        'std_error': np.std(errors_cc),
        'rmse': np.sqrt(np.mean(errors_cc**2)),
        'max_error': np.max(np.abs(errors_cc)),
        'method_name': 'Coulomb Counting'
    }
    
    print("Coulomb Counting Results:")
    print(f"  Mean Error: {results_cc['mean_error']:.6f}")
    print(f"  Std Error: {results_cc['std_error']:.6f}")
    print(f"  RMSE: {results_cc['rmse']:.6f}")
    print(f"  Max Error: {results_cc['max_error']:.6f}")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    t = np.arange(len(current_profile))
    
    # Plot 1: SoC comparison
    axes[0].plot(t, results_feedback['soc_real'], 'b-', linewidth=2, label='Real SoC')
    axes[0].plot(t, results_feedback['soc_estimated'], 'r--', linewidth=2, label='Feedback EKF')
    axes[0].plot(t, results_standard['soc_estimated'], 'g-.', linewidth=2, label='Standard EKF')
    axes[0].plot(t, results_cc['soc_estimated'], 'm:', linewidth=2, label='Coulomb Counting')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('State of Charge')
    axes[0].set_title('SoC Estimation Comparison')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Errors
    axes[1].plot(t, results_feedback['errors'], 'r-', linewidth=1, label='Feedback EKF')
    axes[1].plot(t, results_standard['errors'], 'g-', linewidth=1, label='Standard EKF')
    axes[1].plot(t, results_cc['errors'], 'm-', linewidth=1, label='Coulomb Counting')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Estimation Error')
    axes[1].set_title('Estimation Errors')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Summary table
    comparison_data = {
        'Method': ['Feedback EKF', 'Standard EKF', 'Coulomb Counting'],
        'Mean Error': [results_feedback['mean_error'], results_standard['mean_error'], results_cc['mean_error']],
        'Std Error': [results_feedback['std_error'], results_standard['std_error'], results_cc['std_error']],
        'RMSE': [results_feedback['rmse'], results_standard['rmse'], results_cc['rmse']],
        'Max Error': [results_feedback['max_error'], results_standard['max_error'], results_cc['max_error']]
    }
    
    df = pd.DataFrame(comparison_data)
    print("\nComparison Summary:")
    print(df.to_string(index=False, float_format='%.6f'))
    
    return {
        'feedback_ekf': results_feedback,
        'standard_ekf': results_standard,
        'coulomb_counting': results_cc,
        'comparison_table': df
    }

def main():
    """
    Main function to run the complete simulation
    """
    print("Feedback-based Extended Kalman Filter for Battery SoC Estimation")
    print("=" * 60)
    
    # Generate current profiles
    print("Generating current profiles...")
    bbst_current = generate_bbst_condition(duration=5000)
    constant_current = generate_constant_current(duration=5000, current=1.0)
    
    # Run simulations
    print("\nRunning BBDST condition simulation...")
    ekf_bbst = FeedbackEKF(capacity=1.5, dt=1.0)
    simulator_bbst = BatterySimulator(capacity=1.5, dt=1.0)
    results_bbst = run_simulation(ekf_bbst, simulator_bbst, bbst_current, "Feedback EKF (BBDST)")
    
    print("\nRunning constant current simulation...")
    ekf_constant = FeedbackEKF(capacity=1.5, dt=1.0)
    simulator_constant = BatterySimulator(capacity=1.5, dt=1.0)
    results_constant = run_simulation(ekf_constant, simulator_constant, constant_current, "Feedback EKF (Constant)")
    
    # Plot results
    print("\nPlotting results...")
    plot_results(results_bbst)
    plot_results(results_constant)
    
    # Compare methods
    print("\nComparing different estimation methods...")
    comparison_results = compare_methods(bbst_current)
    
    print("\nSimulation completed successfully!")
    return {
        'bbst_results': results_bbst,
        'constant_results': results_constant,
        'comparison_results': comparison_results
    }

if __name__ == "__main__":
    # Run the main simulation
    results = main()