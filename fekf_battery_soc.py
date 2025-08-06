import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple, List, Optional
import pandas as pd

class BatterySoCEstimator:
    """
    Battery State of Charge (SoC) Estimation using various Kalman Filter methods
    including EKF, UKF, and FEKF (Feedback Extended Kalman Filter)
    """
    
    def __init__(self, capacity: float = 1.5):
        """
        Initialize the battery SoC estimator
        
        Args:
            capacity: Battery capacity in Ah
        """
        self.capacity = capacity
        self.ts = 1.0  # sample interval
        self.tr = 0.1  # smallest time interval for real SOC simulation
        
    def calculate_battery_parameters(self, soc: float) -> Tuple[float, float, float, float]:
        """
        Calculate battery parameters based on SoC using polynomial models
        
        Args:
            soc: State of charge (0-1)
            
        Returns:
            Tuple of (UOC, Rint, Rp, Cp)
        """
        # Open Circuit Voltage (6th order polynomial)
        UOC = (3.44003 + 1.71448 * soc - 3.51247 * soc**2 + 
               5.70868 * soc**3 - 5.06869 * soc**4 + 1.86699 * soc**5)
        
        # Internal Resistance (7th order polynomial)
        Rint = (0.04916 + 1.19552 * soc - 6.25333 * soc**2 + 
                14.24181 * soc**3 - 13.93388 * soc**4 + 2.553 * soc**5 + 
                4.16285 * soc**6 - 1.8713 * soc**7)
        
        # Polarization Resistance (6th order polynomial)
        Rp = (0.02346 - 0.10537 * soc + 1.1371 * soc**2 - 4.55188 * soc**3 + 
              8.26827 * soc**4 - 6.93032 * soc**5 + 2.1787 * soc**6)
        
        # Polarization Capacitance (6th order polynomial)
        Cp = (203.1404 + 3522.78847 * soc - 31392.66753 * soc**2 + 
              122406.91269 * soc**3 - 227590.94382 * soc**4 + 
              198281.56406 * soc**5 - 65171.90395 * soc**6)
        
        return UOC, Rint, Rp, Cp
    
    def generate_current_profile(self, mode: int = 1, N: int = 5000) -> np.ndarray:
        """
        Generate current profile for simulation
        
        Args:
            mode: 1 for BBDST-like profile, 2 for constant current with interruptions
            N: Number of samples
            
        Returns:
            Current profile array
        """
        if mode == 1:
            # BBDST-like profile (simplified)
            t = np.linspace(0, N, N)
            I = 1.5 * np.ones(N)
            # Add some variations to simulate BBDST
            I += 0.3 * np.sin(2 * np.pi * t / 1000)
            I += 0.2 * np.random.randn(N) * 0.1
        else:
            # Constant current with interruptions
            I = 1.5 * np.ones(N)
            I[int(N/5):int(N*3/9)] = 0
            I[int(N*5/9):int(N*4/5)] = 0
            
        return I
    
    def simulate_real_battery(self, current_profile: np.ndarray, soc_init: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate real battery behavior
        
        Args:
            current_profile: Current profile array
            soc_init: Initial SoC
            
        Returns:
            Tuple of (real_soc, real_voltage, real_states)
        """
        N = len(current_profile)
        real_soc = np.zeros(N)
        real_voltage = np.zeros(N)
        real_states = np.zeros((2, N))
        
        # Initialize
        real_soc[0] = soc_init
        real_states[0, 0] = soc_init
        real_states[1, 0] = 0  # Up (polarization voltage)
        
        # Simulation parameters
        Qs = 4e-9
        Qu = 1e-8
        R = 1e-6
        
        for T in range(1, N):
            # Simulate at higher resolution
            for t in range(int(self.ts/self.tr)):
                idx = (T-1) * int(self.ts/self.tr) + t
                if idx >= len(current_profile):
                    break
                    
                # Calculate battery parameters
                UOC, Rint, Rp, Cp = self.calculate_battery_parameters(real_soc[T-1])
                tao = Rp * Cp
                
                # State transition matrix
                A = np.array([[1, 0], [0, np.exp(-self.tr/tao)]])
                B = np.array([[-self.tr/(self.capacity * 3600)], 
                             [Rp * (1 - np.exp(-self.tr/tao))]])
                
                # State update
                noise = np.array([np.sqrt(Qs) * np.random.randn(),
                                np.sqrt(Qu) * np.random.randn()])
                real_states[:, T] = (A @ real_states[:, T-1].reshape(-1, 1) + 
                                    B * current_profile[idx] + noise.reshape(-1, 1)).flatten()
                real_soc[T] = real_states[0, T]
            
            # Calculate voltage
            UOC, Rint, Rp, Cp = self.calculate_battery_parameters(real_soc[T])
            real_voltage[T] = UOC - real_states[1, T] - current_profile[T] * Rint + np.sqrt(R) * np.random.randn()
        
        return real_soc, real_voltage, real_states
    
    def ekf_estimation(self, current_profile: np.ndarray, voltage_obs: np.ndarray, 
                       soc_init: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extended Kalman Filter estimation
        
        Args:
            current_profile: Current profile
            voltage_obs: Observed voltage
            soc_init: Initial SoC estimate
            
        Returns:
            Tuple of (soc_ekf, error_ekf)
        """
        N = len(current_profile)
        soc_ekf = np.zeros(N)
        error_ekf = np.zeros(N)
        
        # Initialize
        soc_ekf[0] = soc_init
        states_ekf = np.array([soc_init, 0])  # [SoC, Up]
        P_cov = np.array([[1e-8, 0], [0, 1e-6]])  # covariance matrix
        
        # Noise parameters
        Qs = 4e-9
        Qu = 1e-8
        R = 1e-6
        
        for T in range(1, N):
            # Calculate battery parameters
            UOC, Rint, Rp, Cp = self.calculate_battery_parameters(soc_ekf[T-1])
            tao = Rp * Cp
            
            # State transition matrix
            A = np.array([[1, 0], [0, np.exp(-self.ts/tao)]])
            B = np.array([[-self.ts/(self.capacity * 3600)], 
                         [Rp * (1 - np.exp(-self.ts/tao))]])
            
            # Prediction
            states_pre = A @ states_ekf.reshape(-1, 1) + B * current_profile[T]
            P_cov = A @ P_cov @ A.T + np.array([[Qs, 0], [0, Qu]])
            
            # Measurement prediction
            UOC_pre, Rint_pre, _, _ = self.calculate_battery_parameters(states_pre[0, 0])
            voltage_pre = UOC_pre - states_pre[1, 0] - current_profile[T] * Rint_pre
            
            # Linearization
            C1 = (1.71448 - 2 * 3.51247 * soc_ekf[T-1] + 3 * 5.70868 * soc_ekf[T-1]**2 - 
                  4 * 5.06869 * soc_ekf[T-1]**3 + 5 * 1.86699 * soc_ekf[T-1]**4)
            C = np.array([C1, -1])
            
            # Update
            K = P_cov @ C.T @ np.linalg.inv(C @ P_cov @ C.T + R)
            states_ekf = states_pre.flatten() + K.flatten() * (voltage_obs[T] - voltage_pre)
            P_cov = P_cov - K @ C @ P_cov
            
            soc_ekf[T] = states_ekf[0]
        
        return soc_ekf, error_ekf
    
    def ukf_estimation(self, current_profile: np.ndarray, voltage_obs: np.ndarray, 
                       soc_init: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unscented Kalman Filter estimation
        
        Args:
            current_profile: Current profile
            voltage_obs: Observed voltage
            soc_init: Initial SoC estimate
            
        Returns:
            Tuple of (soc_ukf, error_ukf)
        """
        N = len(current_profile)
        soc_ukf = np.zeros(N)
        error_ukf = np.zeros(N)
        
        # Initialize
        soc_ukf[0] = soc_init
        P = 1e-6
        
        # UKF parameters
        n = 1
        alpha = 0.04
        beta = 2
        lambda_param = 1.5
        
        # Weights
        Wm = np.array([lambda_param / (n + lambda_param)] + [0.5 / (n + lambda_param)] * (2 * n))
        Wc = Wm.copy()
        Wc[0] = Wc[0] + (1 - alpha**2 + beta)
        
        # Noise parameters
        Qs = 4e-9
        R = 1e-6
        
        for T in range(1, N):
            # Sigma points
            pk = np.sqrt((n + lambda_param) * P)
            sigma = np.array([soc_ukf[T-1], soc_ukf[T-1] + pk, soc_ukf[T-1] - pk])
            
            # Predict
            for i in range(len(sigma)):
                sigma[i] = sigma[i] - current_profile[T] * self.ts / (self.capacity * 3600)
            
            sxk = np.sum(Wm * sigma)
            spk = np.sum(Wc * (sigma - sxk)**2) + Qs
            
            # Update sigma points
            pkr = np.sqrt((n + lambda_param) * spk)
            sigma = np.array([sxk, sxk + pkr, sxk - pkr])
            
            # Measurement prediction
            gamma = np.zeros(len(sigma))
            for i in range(len(sigma)):
                UOC, Rint, Rp, Cp = self.calculate_battery_parameters(sigma[i])
                tao = Rp * Cp
                gamma[i] = UOC - current_profile[T] * Rint - current_profile[T] * Rp * (1 - np.exp(-self.ts/tao))
            
            syk = np.sum(Wm * gamma)
            pyy = np.sum(Wc * (gamma - syk)**2) + R
            pxy = np.sum(Wc * (sigma - sxk) * (gamma - syk))
            
            # Update
            kgs = pxy / pyy
            soc_ukf[T] = sxk + kgs * (voltage_obs[T] - syk)
            P = spk - kgs * pyy * kgs
        
        return soc_ukf, error_ukf
    
    def fekf_basic_estimation(self, current_profile: np.ndarray, voltage_obs: np.ndarray, 
                              soc_init: float = 1.0, feedback_gain: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Basic Feedback Extended Kalman Filter estimation
        
        Args:
            current_profile: Current profile
            voltage_obs: Observed voltage
            soc_init: Initial SoC estimate
            feedback_gain: Feedback gain parameter
            
        Returns:
            Tuple of (soc_fekf, error_fekf)
        """
        N = len(current_profile)
        soc_fekf = np.zeros(N)
        error_fekf = np.zeros(N)
        
        # Initialize
        soc_fekf[0] = soc_init
        states_fekf = np.array([soc_init, 0])  # [SoC, Up]
        P_cov = np.array([[1e-8, 0], [0, 1e-6]])  # covariance matrix
        
        # Noise parameters
        Qs = 4e-9
        Qu = 1e-8
        R = 1e-6
        
        for T in range(1, N):
            # Calculate battery parameters
            UOC, Rint, Rp, Cp = self.calculate_battery_parameters(soc_fekf[T-1])
            tao = Rp * Cp
            
            # State transition matrix
            A = np.array([[1, 0], [0, np.exp(-self.ts/tao)]])
            B = np.array([[-self.ts/(self.capacity * 3600)], 
                         [Rp * (1 - np.exp(-self.ts/tao))]])
            
            # Prediction
            states_pre = A @ states_fekf.reshape(-1, 1) + B * current_profile[T]
            P_cov = A @ P_cov @ A.T + np.array([[Qs, 0], [0, Qu]])
            
            # Measurement prediction
            UOC_pre, Rint_pre, _, _ = self.calculate_battery_parameters(states_pre[0, 0])
            voltage_pre = UOC_pre - states_pre[1, 0] - current_profile[T] * Rint_pre
            
            # Linearization
            C1 = (1.71448 - 2 * 3.51247 * soc_fekf[T-1] + 3 * 5.70868 * soc_fekf[T-1]**2 - 
                  4 * 5.06869 * soc_fekf[T-1]**3 + 5 * 1.86699 * soc_fekf[T-1]**4)
            C = np.array([C1, -1])
            
            # Innovation
            innovation = voltage_obs[T] - voltage_pre
            
            # FEKF gain calculation with feedback
            K = P_cov @ C.T @ np.linalg.inv(C @ P_cov @ C.T + R)
            
            # Apply feedback to gain
            feedback_factor = 1 + feedback_gain * abs(innovation)
            K = K * feedback_factor
            
            # Update
            states_fekf = states_pre.flatten() + K.flatten() * innovation
            P_cov = P_cov - K @ C @ P_cov
            
            soc_fekf[T] = states_fekf[0]
        
        return soc_fekf, error_fekf
    
    def fekf_advanced_estimation(self, current_profile: np.ndarray, voltage_obs: np.ndarray, 
                                 soc_init: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advanced Feedback Extended Kalman Filter estimation with adaptive feedback
        
        Args:
            current_profile: Current profile
            voltage_obs: Observed voltage
            soc_init: Initial SoC estimate
            
        Returns:
            Tuple of (soc_fekf, error_fekf)
        """
        N = len(current_profile)
        soc_fekf = np.zeros(N)
        error_fekf = np.zeros(N)
        
        # Initialize
        soc_fekf[0] = soc_init
        states_fekf = np.array([soc_init, 0])  # [SoC, Up]
        P_cov = np.array([[1e-8, 0], [0, 1e-6]])  # covariance matrix
        
        # Noise parameters
        Qs = 4e-9
        Qu = 1e-8
        R = 1e-6
        
        # Advanced feedback parameters
        feedback_gain_initial = 0.1
        feedback_gain = feedback_gain_initial
        innovation_window = 10
        innovation_history = np.zeros(innovation_window)
        innovation_index = 0
        
        # Adaptive parameters
        adaptive_factor = 1.0
        min_adaptive_factor = 0.1
        max_adaptive_factor = 2.0
        
        for T in range(1, N):
            # Calculate battery parameters
            UOC, Rint, Rp, Cp = self.calculate_battery_parameters(soc_fekf[T-1])
            tao = Rp * Cp
            
            # State transition matrix
            A = np.array([[1, 0], [0, np.exp(-self.ts/tao)]])
            B = np.array([[-self.ts/(self.capacity * 3600)], 
                         [Rp * (1 - np.exp(-self.ts/tao))]])
            
            # Prediction with adaptive factor
            states_pre = A @ states_fekf.reshape(-1, 1) + B * current_profile[T]
            P_cov = A @ P_cov @ A.T + adaptive_factor * np.array([[Qs, 0], [0, Qu]])
            
            # Measurement prediction
            UOC_pre, Rint_pre, _, _ = self.calculate_battery_parameters(states_pre[0, 0])
            voltage_pre = UOC_pre - states_pre[1, 0] - current_profile[T] * Rint_pre
            
            # Linearization
            C1 = (1.71448 - 2 * 3.51247 * soc_fekf[T-1] + 3 * 5.70868 * soc_fekf[T-1]**2 - 
                  4 * 5.06869 * soc_fekf[T-1]**3 + 5 * 1.86699 * soc_fekf[T-1]**4)
            C = np.array([C1, -1])
            
            # Innovation
            innovation = voltage_obs[T] - voltage_pre
            
            # Store innovation in history
            innovation_history[innovation_index] = innovation
            innovation_index = (innovation_index + 1) % innovation_window
            
            # Adaptive feedback mechanism
            if T > innovation_window:
                innovation_variance = np.var(innovation_history)
                innovation_mean = np.mean(innovation_history)
                
                # Adjust feedback gain based on innovation statistics
                if abs(innovation_mean) > 0.01:
                    feedback_gain = feedback_gain_initial * (1 + abs(innovation_mean))
                else:
                    feedback_gain = feedback_gain_initial
                
                # Adjust adaptive factor based on innovation variance
                if innovation_variance > 1e-4:
                    adaptive_factor = min(max_adaptive_factor, adaptive_factor * 1.1)
                else:
                    adaptive_factor = max(min_adaptive_factor, adaptive_factor * 0.95)
            
            # FEKF gain calculation with feedback
            K = P_cov @ C.T @ np.linalg.inv(C @ P_cov @ C.T + R)
            
            # Apply feedback to gain
            feedback_factor = 1 + feedback_gain * abs(innovation)
            K = K * feedback_factor
            
            # Update
            states_fekf = states_pre.flatten() + K.flatten() * innovation
            P_cov = P_cov - K @ C @ P_cov
            
            soc_fekf[T] = states_fekf[0]
        
        return soc_fekf, error_fekf
    
    def ah_estimation(self, current_profile: np.ndarray, soc_init: float = 1.0) -> np.ndarray:
        """
        Ampere-hour counting estimation
        
        Args:
            current_profile: Current profile
            soc_init: Initial SoC estimate
            
        Returns:
            SoC estimated by AH counting
        """
        N = len(current_profile)
        soc_ah = np.zeros(N)
        soc_ah[0] = soc_init
        
        for T in range(1, N):
            soc_ah[T] = soc_ah[T-1] - self.ts / (self.capacity * 3600) * current_profile[T]
        
        return soc_ah
    
    def run_comparison(self, mode: int = 1, soc_init: float = 1.0, N: int = 5000) -> dict:
        """
        Run comprehensive comparison of all methods
        
        Args:
            mode: Working mode (1 for BBDST-like, 2 for constant current)
            soc_init: Initial SoC
            N: Number of samples
            
        Returns:
            Dictionary with results
        """
        print(f"Running battery SoC estimation comparison...")
        print(f"Mode: {mode}, Initial SoC: {soc_init}, Samples: {N}")
        
        # Generate current profile
        current_profile = self.generate_current_profile(mode, N)
        
        # Simulate real battery
        real_soc, real_voltage, real_states = self.simulate_real_battery(current_profile, soc_init)
        
        # Add observation noise to current
        current_obs = current_profile + 0.01 * self.capacity * np.random.randn(N)
        
        results = {}
        
        # AH estimation
        start_time = time.time()
        soc_ah = self.ah_estimation(current_obs, soc_init)
        ah_time = time.time() - start_time
        error_ah = real_soc - soc_ah
        results['AH'] = {
            'soc': soc_ah,
            'error': error_ah,
            'avg_error': np.mean(np.abs(error_ah)),
            'std_error': np.std(error_ah),
            'time': ah_time
        }
        
        # EKF estimation
        start_time = time.time()
        soc_ekf, _ = self.ekf_estimation(current_obs, real_voltage, soc_init)
        ekf_time = time.time() - start_time
        error_ekf = real_soc - soc_ekf
        results['EKF'] = {
            'soc': soc_ekf,
            'error': error_ekf,
            'avg_error': np.mean(np.abs(error_ekf)),
            'std_error': np.std(error_ekf),
            'time': ekf_time
        }
        
        # UKF estimation
        start_time = time.time()
        soc_ukf, _ = self.ukf_estimation(current_obs, real_voltage, soc_init)
        ukf_time = time.time() - start_time
        error_ukf = real_soc - soc_ukf
        results['UKF'] = {
            'soc': soc_ukf,
            'error': error_ukf,
            'avg_error': np.mean(np.abs(error_ukf)),
            'std_error': np.std(error_ukf),
            'time': ukf_time
        }
        
        # Basic FEKF estimation
        start_time = time.time()
        soc_fekf_basic, _ = self.fekf_basic_estimation(current_obs, real_voltage, soc_init)
        fekf_basic_time = time.time() - start_time
        error_fekf_basic = real_soc - soc_fekf_basic
        results['FEKF_Basic'] = {
            'soc': soc_fekf_basic,
            'error': error_fekf_basic,
            'avg_error': np.mean(np.abs(error_fekf_basic)),
            'std_error': np.std(error_fekf_basic),
            'time': fekf_basic_time
        }
        
        # Advanced FEKF estimation
        start_time = time.time()
        soc_fekf_adv, _ = self.fekf_advanced_estimation(current_obs, real_voltage, soc_init)
        fekf_adv_time = time.time() - start_time
        error_fekf_adv = real_soc - soc_fekf_adv
        results['FEKF_Advanced'] = {
            'soc': soc_fekf_adv,
            'error': error_fekf_adv,
            'avg_error': np.mean(np.abs(error_fekf_adv)),
            'std_error': np.std(error_fekf_adv),
            'time': fekf_adv_time
        }
        
        # Store real values
        results['Real'] = {
            'soc': real_soc,
            'voltage': real_voltage,
            'current': current_profile
        }
        
        return results
    
    def plot_results(self, results: dict, save_plot: bool = False):
        """
        Plot comparison results
        
        Args:
            results: Results dictionary from run_comparison
            save_plot: Whether to save the plot
        """
        methods = ['AH', 'EKF', 'UKF', 'FEKF_Basic', 'FEKF_Advanced']
        colors = ['red', 'green', 'orange', 'blue', 'purple']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: SoC comparison
        t = np.arange(len(results['Real']['soc']))
        ax1.plot(t, results['Real']['soc'], 'k-', linewidth=2, label='Real SoC')
        for i, method in enumerate(methods):
            ax1.plot(t, results[method]['soc'], '--', color=colors[i], 
                    linewidth=1.5, label=f'{method} SoC')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('SoC')
        ax1.set_title('SoC Estimation Comparison')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Error comparison
        for i, method in enumerate(methods):
            ax2.plot(t, results[method]['error'], '-', color=colors[i], 
                    linewidth=1.5, label=f'{method} Error')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Error')
        ax2.set_title('Estimation Error Comparison')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Performance metrics
        methods_short = ['AH', 'EKF', 'UKF', 'FEKF-B', 'FEKF-A']
        avg_errors = [results[method]['avg_error'] for method in methods]
        std_errors = [results[method]['std_error'] for method in methods]
        
        x = np.arange(len(methods_short))
        width = 0.35
        
        ax3.bar(x - width/2, avg_errors, width, label='Average Error')
        ax3.bar(x + width/2, std_errors, width, label='Std Error')
        ax3.set_xlabel('Methods')
        ax3.set_ylabel('Error')
        ax3.set_title('Performance Metrics')
        ax3.set_xticks(x)
        ax3.set_xticklabels(methods_short)
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Execution time
        times = [results[method]['time'] for method in methods]
        ax4.bar(methods_short, times, color=colors)
        ax4.set_xlabel('Methods')
        ax4.set_ylabel('Execution Time (s)')
        ax4.set_title('Execution Time Comparison')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('battery_soc_comparison.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_results(self, results: dict):
        """
        Print comparison results
        
        Args:
            results: Results dictionary from run_comparison
        """
        methods = ['AH', 'EKF', 'UKF', 'FEKF_Basic', 'FEKF_Advanced']
        
        print("\n" + "="*60)
        print("BATTERY SoC ESTIMATION COMPARISON RESULTS")
        print("="*60)
        print(f"{'Method':<15} {'Avg Error':<12} {'Std Error':<12} {'Time (s)':<10}")
        print("-"*60)
        
        for method in methods:
            print(f"{method:<15} {results[method]['avg_error']:<12.6f} "
                  f"{results[method]['std_error']:<12.6f} {results[method]['time']:<10.3f}")
        
        # Find best method
        best_method = min(methods, key=lambda x: results[x]['avg_error'])
        baseline_error = results['EKF']['avg_error']
        
        print("\n" + "="*60)
        print("IMPROVEMENT ANALYSIS")
        print("="*60)
        
        for method in methods:
            improvement = (baseline_error - results[method]['avg_error']) / baseline_error * 100
            if method == 'EKF':
                print(f"{method}: Baseline (0% improvement)")
            else:
                if improvement > 0:
                    print(f"{method}: {improvement:.2f}% improvement over EKF")
                else:
                    print(f"{method}: {-improvement:.2f}% degradation compared to EKF")
        
        print(f"\nBest performing method: {best_method}")
        print(f"Average error: {results[best_method]['avg_error']:.6f}")
        print("="*60)


# Example usage and testing
def main():
    """
    Main function to demonstrate the battery SoC estimation
    """
    print("Battery State of Charge Estimation using FEKF")
    print("="*50)
    
    # Initialize estimator
    estimator = BatterySoCEstimator(capacity=1.5)
    
    # Run comparison
    results = estimator.run_comparison(mode=1, soc_init=1.0, N=5000)
    
    # Print results
    estimator.print_results(results)
    
    # Plot results
    estimator.plot_results(results, save_plot=True)
    
    return results


if __name__ == "__main__":
    # Run the main function
    results = main()