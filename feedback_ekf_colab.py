import numpy as np
import matplotlib.pyplot as plt

class BatteryModel:
    def __init__(self, capacity=1.5):
        self.capacity = capacity
    
    def get_parameters(self, soc):
        UOC = (3.44003 + 1.71448*soc - 3.51247*soc**2 + 
               5.70868*soc**3 - 5.06869*soc**4 + 1.86699*soc**5)
        Ro = (0.04916 + 1.19552*soc - 6.25333*soc**2 + 
              14.24181*soc**3 - 13.93388*soc**4 + 2.553*soc**5 + 
              4.16285*soc**6 - 1.8713*soc**7)
        Rp = (0.02346 - 0.10537*soc + 1.1371*soc**2 - 
              4.55188*soc**3 + 8.26827*soc**4 - 6.93032*soc**5 + 
              2.1787*soc**6)
        Cp = (203.1404 + 3522.78847*soc - 31392.66753*soc**2 + 
              122406.91269*soc**3 - 227590.94382*soc**4 + 
              198281.56406*soc**5 - 65171.90395*soc**6)
        return UOC, Ro, Rp, Cp

class FeedbackEKF:
    def __init__(self, capacity=1.5, dt=1.0):
        self.capacity = capacity
        self.dt = dt
        self.battery = BatteryModel(capacity)
        self.x = np.array([1.0, 0.0])
        self.P = np.diag([1e-8, 1e-6])
        self.Q = np.diag([4e-9, 1e-8])
        self.R = 1e-6
        self.alpha = 0.1
        self.beta = 0.05
        self.soc_history = []
    
    def predict(self, current):
        soc = self.x[0]
        _, _, Rp, Cp = self.battery.get_parameters(soc)
        tao = Rp * Cp
        A = np.array([[1, 0], [0, np.exp(-self.dt/tao)]])
        B = np.array([[-self.dt/(self.capacity*3600)],
                     [Rp*(1-np.exp(-self.dt/tao))]])
        x_pred = A @ self.x + B.flatten() * current
        P_pred = A @ self.P @ A.T + self.Q
        return x_pred, P_pred
    
    def update(self, voltage_measured, current, x_pred, P_pred):
        soc_pred = x_pred[0]
        UOC_pred, Ro_pred, _, _ = self.battery.get_parameters(soc_pred)
        voltage_pred = UOC_pred - x_pred[1] - current * Ro_pred
        innovation = voltage_measured - voltage_pred
        
        dUOC_dSoC = (1.71448 - 2*3.51247*soc_pred + 
                     3*5.70868*soc_pred**2 - 4*5.06869*soc_pred**3 + 
                     5*1.86699*soc_pred**4)
        C = np.array([dUOC_dSoC, -1])
        
        S = C @ P_pred @ C.T + self.R
        K = P_pred @ C.T / S
        
        feedback_gain = self.alpha * np.abs(innovation) + self.beta
        K_adaptive = K * (1 + feedback_gain)
        
        self.x = x_pred + K_adaptive * innovation
        self.P = P_pred - K_adaptive @ C @ P_pred
        self.soc_history.append(self.x[0])
    
    def step(self, voltage_measured, current):
        x_pred, P_pred = self.predict(current)
        self.update(voltage_measured, current, x_pred, P_pred)

class BatterySimulator:
    def __init__(self, capacity=1.5, dt=1.0):
        self.capacity = capacity
        self.dt = dt
        self.battery = BatteryModel(capacity)
        self.soc_real = 1.0
        self.up_real = 0.0
        self.soc_real_history = []
        self.voltage_real_history = []
    
    def step(self, current):
        self.soc_real -= current * self.dt / (self.capacity * 3600)
        self.soc_real = np.clip(self.soc_real, 0, 1)
        
        UOC, Ro, Rp, Cp = self.battery.get_parameters(self.soc_real)
        tao = Rp * Cp
        
        self.up_real = (self.up_real * np.exp(-self.dt/tao) + 
                       Rp * current * (1-np.exp(-self.dt/tao)))
        
        voltage_real = UOC - self.up_real - current * Ro
        voltage_noise = np.random.normal(0, np.sqrt(1e-6))
        voltage_real += voltage_noise
        
        self.soc_real_history.append(self.soc_real)
        self.voltage_real_history.append(voltage_real)
        
        return voltage_real, self.soc_real

def generate_bbst_condition(duration=2000):
    t = np.linspace(0, duration, duration)
    base_current = 0.5*np.sin(2*np.pi*0.001*t) + 0.3*np.sin(2*np.pi*0.002*t)
    
    acceleration_events = np.zeros_like(t)
    for i in range(0, duration, 200):
        if i + 40 < duration:
            acceleration_events[i:i+40] = 2.0 * np.sin(np.pi * np.linspace(0, 1, 40))
    
    braking_events = np.zeros_like(t)
    for i in range(100, duration, 200):
        if i + 20 < duration:
            braking_events[i:i+20] = -1.5 * np.sin(np.pi * np.linspace(0, 1, 20))
    
    current_profile = base_current + acceleration_events + braking_events
    current_profile += np.random.normal(0, 0.1, duration)
    current_profile = np.clip(current_profile, -3.0, 3.0)
    
    return current_profile

def run_demo():
    print("Feedback-based Extended Kalman Filter Demo")
    print("=" * 50)
    
    current_profile = generate_bbst_condition(duration=2000)
    ekf = FeedbackEKF(capacity=1.5, dt=1.0)
    simulator = BatterySimulator(capacity=1.5, dt=1.0)
    
    soc_estimated = []
    soc_real = []
    voltage_measured = []
    errors = []
    
    print("Running simulation...")
    
    for i, current in enumerate(current_profile):
        voltage, soc_real_val = simulator.step(current)
        ekf.step(voltage, current)
        
        soc_estimated.append(ekf.x[0])
        soc_real.append(soc_real_val)
        voltage_measured.append(voltage)
        errors.append(soc_real_val - ekf.x[0])
        
        if i % 500 == 0:
            print(f"Progress: {i}/{len(current_profile)} ({100*i/len(current_profile):.1f}%)")
    
    errors = np.array(errors)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    rmse = np.sqrt(np.mean(errors**2))
    
    print(f"\nResults:")
    print(f"Mean Error: {mean_error:.6f}")
    print(f"Std Error: {std_error:.6f}")
    print(f"RMSE: {rmse:.6f}")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    t = np.arange(len(current_profile))
    
    axes[0].plot(t, soc_real, "b-", linewidth=2, label="Real SoC")
    axes[0].plot(t, soc_estimated, "r--", linewidth=2, label="Estimated SoC")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("State of Charge")
    axes[0].set_title("Feedback EKF - SoC Estimation")
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(t, errors, "m-", linewidth=1)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Estimation Error")
    axes[1].set_title("Estimation Errors")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        "soc_estimated": np.array(soc_estimated),
        "soc_real": np.array(soc_real),
        "voltage_measured": np.array(voltage_measured),
        "current_profile": current_profile,
        "errors": errors,
        "mean_error": mean_error,
        "std_error": std_error,
        "rmse": rmse
    }

if __name__ == "__main__":
    results = run_demo()
