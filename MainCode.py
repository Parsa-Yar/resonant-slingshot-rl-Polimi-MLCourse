import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import ExpectedImprovement
from bayes_opt.acquisition import UpperConfidenceBound
from sklearn.gaussian_process.kernels import Matern


# Mission Parameters
TARGET_XD = 0.8
NOMINAL_LIMIT = 0.7 
MIN_FOLD_LIMIT = 0.05   

SYSTEM_PARAMS = {
    'MB': 70,  # Base mass (70 kg)
    'BB': 264, # Base damping (from the paper = 2*hb * sqrt(MB*KB))
    'KB': 4000.0, # Base stiffness(4000)
    'MR': 20.0,  # Robot mass (kg)
}

BO_INIT_POINTS = 15    
BO_ITERATIONS = 500        
SIMULATION_TIME = 12

#  Physics Simulator
def simulate_2dof_system(XC_func, params, t_span, t_eval):
    MB, BB, KB = params['MB'], params['BB'], params['KB']
    MR, KR, DR = params['MR'], params['KR'], params['DR']
    
    actual_DR = 2.0 * DR * np.sqrt(KR * MR)
    
    def dynamics(t, y):
        xb, xb_dot, xr, xr_dot = y
        
        dt_calc = 1e-4
        xc = XC_func(t)
        xc_dot = (XC_func(t + dt_calc) - xc) / dt_calc
        
        F_imp = -KR * (xr - xc) - actual_DR * (xr_dot - xc_dot)
        
        K_wall = 100000.0  
        D_wall = 5000.0    
        MIN_FOLD_LIMIT = 0.05 
        
        F_wall_internal = 0.0
        if xr < MIN_FOLD_LIMIT:
            F_wall_internal = -K_wall * (xr - MIN_FOLD_LIMIT) - D_wall * xr_dot

        F_wall_stretch = 0.0
        if xr > NOMINAL_LIMIT:
            F_wall_stretch = -K_wall * (xr - NOMINAL_LIMIT) - D_wall * xr_dot
        
        F_total_internal = F_imp + F_wall_internal + F_wall_stretch
        
        xb_ddot = (-BB * xb_dot - KB * xb - F_total_internal) / MB
        xr_ddot = (F_total_internal / MR) - xb_ddot
        
        return [xb_dot, xb_ddot, xr_dot, xr_ddot]
    initial_xr = XC_func(0.0)

    sol = solve_ivp(dynamics, t_span, [0.0, 0.0, initial_xr, 0.0], t_eval=t_eval, method='Radau', max_step=0.01)
    return sol
    

def create_chirp_hold_func(f0, f1, t1, amp, offset):
    def XC_func(t):
        # smooth velocity at the start of the simulation to prevent numerical issues with the ODE solver
        safe_t = max(0.0, t)
        fade_in = min(safe_t / 0.5, 1.0)
        
        if safe_t <= t1:
            freq = f0 + ((f1 - f0) / t1) * safe_t
            return offset + (amp * np.sin(2 * np.pi * freq * safe_t)) * fade_in
        else:
            # hold phase
            freq_end = f1 
            val_at_t1 = offset + amp * np.sin(2 * np.pi * freq_end * t1) # constant command after t1
            return val_at_t1
            
    return XC_func


def optimize_target_reach(f0, f1, t1, amp, offset, KR, DR):
    MIN_FOLD_LIMIT = 0.05
    max_command = round(offset + abs(amp), 5)
    min_command = round(offset - abs(amp), 5)
    
    if max_command > NOMINAL_LIMIT:
        return -10.0 - 1000 * (max_command - NOMINAL_LIMIT) 
    if min_command < MIN_FOLD_LIMIT:    
        return -10.0 - 1000 * (MIN_FOLD_LIMIT - min_command)

    current_XC_func = create_chirp_hold_func(f0, f1, t1, amp, offset)

    current_params = SYSTEM_PARAMS.copy()
    current_params.update({'KR': KR, 'DR': DR})
    
    t_eval = np.linspace(0, SIMULATION_TIME, 400) 
    sol = simulate_2dof_system(current_XC_func, current_params, (0, SIMULATION_TIME), t_eval)
    
    penalty = 0
    min_xr = np.min(sol.y[2])
    max_xr = np.max(sol.y[2])

    if min_xr < MIN_FOLD_LIMIT:
        penalty += (MIN_FOLD_LIMIT - min_xr) * 200.0

    if max_xr > NOMINAL_LIMIT:
        penalty += (max_xr - NOMINAL_LIMIT) * 500.0

    absolute_outreach = sol.y[0] + sol.y[2]
    max_outreach = np.max(absolute_outreach)

    diff = max_outreach - TARGET_XD
    error = abs(diff) * (200.0 if diff < 0 else 100.0)

    return -error - penalty



def generate_dynamic_bounds(MB, KB, nominal_limit, min_fold_limit, buffer=0.05):
    # fn = (1 / 2pi) * sqrt(K / M)
    fn = (1.0 / (2.0 * np.pi)) * np.sqrt(KB / MB)
    
    safe_floor = min_fold_limit + buffer
    safe_ceiling = nominal_limit - buffer
    
    max_safe_amp = (safe_ceiling - safe_floor) / 2.0
    perfect_offset = safe_floor + max_safe_amp
    
    print(f"--- Dynamic Bounds Generated ---")
    print(f"Estimated Natural Frequency: {fn:.2f} Hz")
    print(f"Max Safe Amplitude: {max_safe_amp:.3f}m | Offset: {perfect_offset:.3f}m")
    
    return {
        'f0': (max(0.5, fn - 0.5), fn + 0.3),      
        'f1': (fn - 0.1, fn + 0.8),      
        't1': (2.0, 12.0),
        'amp': (0.1, max_safe_amp),    
        'offset': (perfect_offset - 0.01, perfect_offset + 0.01), 
        
        'KR': (3000.0, 5000.0),
        'DR': (0.3, 1.0) 
    }


# Main Execution
if __name__ == "__main__":
    run_idx = 1
    while os.path.exists(f"run_{run_idx}"):
        run_idx += 1
    output_dir = f"run_{run_idx}"
    os.makedirs(output_dir)

    print(f"Target Outreach: {TARGET_XD}m | Workspace Limit: {NOMINAL_LIMIT}m")
    print(f"Base Mass (MB): {SYSTEM_PARAMS['MB']}kg")
    print(f"Output Data Directory: {output_dir}")
    print("Step 1: Running Bayesian Optimization: ")

    pbounds = generate_dynamic_bounds(
        MB=SYSTEM_PARAMS['MB'], 
        KB=SYSTEM_PARAMS['KB'], 
        nominal_limit=NOMINAL_LIMIT, 
        min_fold_limit=MIN_FOLD_LIMIT, 
        buffer=0.02
    )
    
    ucb_acq = UpperConfidenceBound(kappa=2.0)

    optimizer = BayesianOptimization(
        f=optimize_target_reach,
        pbounds=pbounds,
        random_state=42,
        allow_duplicate_points=True,
        verbose=2,
        acquisition_function=ucb_acq
    )

    ard_length_scale = np.ones(7)
    
    ard_kernel = Matern(
        length_scale=ard_length_scale, 
        length_scale_bounds=(1e-5, 1e5),
        nu=2.5
    )

    optimizer.set_gp_params(kernel=ard_kernel, alpha=1e-6, normalize_y=True)

    optimizer.maximize(
        init_points=BO_INIT_POINTS, 
        n_iter=BO_ITERATIONS,
    )
    
    print("\nGenerating Convergence Plot: ")
    targets = [res["target"] for res in optimizer.res]
    errors = [-t for t in targets]
    
    best_errors = np.minimum.accumulate(errors)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(errors) + 1), best_errors, 'b-', linewidth=2)
    plt.plot(range(1, len(errors) + 1), errors, 'k.', alpha=0.3, label='Exploration Points')
    
    plt.yscale('log') 
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Squared Error (m²)', fontsize=12)
    plt.title('Bayesian Optimization Learning Curve', fontsize=14, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(['Best Found Error', 'Evaluated Points'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bo_convergence_plot.png'), dpi=300)
    plt.show()

    best = optimizer.max['params']
    print("\n" + "="*50)
    print("Step 2: Optimization Complete! Best Parameters:")
    for key, value in best.items():
        print(f"  {key}: {value:.3f}")

    print("\nStep 3: Generating Final Plot...")


    def final_optimal_chirp(t):
        safe_t = max(0.0, t)
        fade_in = min(safe_t / 0.5, 1.0) 
        if safe_t <= best['t1']:
            freq = best['f0'] + ((best['f1'] - best['f0']) / best['t1']) * safe_t
            return best['offset'] + (best['amp'] * np.sin(2 * np.pi * freq * safe_t)) * fade_in
        else:
            freq_end = best['f1']
            return best['offset'] + best['amp'] * np.sin(2 * np.pi * freq_end * best['t1'])

    final_params = SYSTEM_PARAMS.copy()
    final_params.update({'KR': best['KR'], 'DR': best['DR']})
    
    t_eval_high_res = np.linspace(0, SIMULATION_TIME, 1500)
    sol = simulate_2dof_system(final_optimal_chirp, final_params, (0, SIMULATION_TIME), t_eval_high_res)

    plt.figure(figsize=(12, 6))
    XC_plot = [final_optimal_chirp(t) for t in sol.t]
    
    absolute_XR = sol.y[0] + sol.y[2]
    absolute_XC = sol.y[0] + np.array(XC_plot) 

    plt.plot(sol.t, sol.y[0], 'r-', alpha=0.6, linewidth=2, label='Base Position ($X_B$)')
    plt.plot(sol.t, absolute_XC, 'k--', alpha=0.8, linewidth=2, label='Absolute Command ($X_B + X_C$)')
    plt.plot(sol.t, absolute_XR, 'b-', linewidth=2.5, label='Absolute Robot Position ($X_B + X_R$)')

    plt.axhline(NOMINAL_LIMIT, color='orange', linestyle=':', linewidth=2, label=f'Nominal Limit ({NOMINAL_LIMIT}m)')
    plt.axhline(TARGET_XD, color='green', linestyle='--', linewidth=2, label=f'Target Outreach ($X_D$ = {TARGET_XD}m)')

    max_outreach = np.max(absolute_XR)
    peak_time = sol.t[np.argmax(absolute_XR)]
    plt.plot(peak_time, max_outreach, 'go', markersize=8)
    plt.annotate(f'Peak: {max_outreach:.3f}m', xy=(peak_time, max_outreach), 
                 xytext=(peak_time + 0.2, max_outreach + 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                 fontsize=10, weight='bold')

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Position (m)', fontsize=12)
    plt.title(f'ML-Optimized Absolute Outreach (Target: {TARGET_XD}m, $M_B$: {SYSTEM_PARAMS["MB"]}kg)', fontsize=14, weight='bold')
    plt.legend(loc='lower right', framealpha=0.9) 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimal_outreach_plot_absolute.png'), dpi=300)
    plt.show()


    print("\nStep 3.5: Generating Relative Extension Plot (Constraint Proof)...")
    plt.figure(figsize=(10, 4))
    
    plt.plot(sol.t, sol.y[2], 'm-', linewidth=2.5, label='Relative Arm Stretch ($X_R$)')
    plt.plot(sol.t, XC_plot, 'k--', alpha=0.6, linewidth=2, label='Motor Command ($X_C$)')
    
    plt.axhline(NOMINAL_LIMIT, color='red', linestyle='-', linewidth=2, label=f'Max Stretch Limit ({NOMINAL_LIMIT}m)')
    plt.axhline(MIN_FOLD_LIMIT, color='red', linestyle='-', linewidth=2, label=f'Min Fold Limit ({MIN_FOLD_LIMIT}m)')
    
    plt.fill_between(sol.t, MIN_FOLD_LIMIT, NOMINAL_LIMIT, color='green', alpha=0.1, label='Safe Operating Zone')
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Relative Extension (m)', fontsize=12)
    plt.title('Proof of Constraint Adherence (Relative Frame)', fontsize=14, weight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'relative_stretch_proof.png'), dpi=300)
    
print(f"Final Actual Error: {abs(max_outreach - TARGET_XD):.5f} m")

RL_TARGET_XD = max_outreach
print(f"RL Target set to BO peak: {RL_TARGET_XD:.4f} m")


# Robustness Analysis  

print("\nStep 4: Running Robustness Analysis (+/- 10% Base Mass)...")

sol_baseline = simulate_2dof_system(final_optimal_chirp, final_params, (0, SIMULATION_TIME), t_eval_high_res)
abs_XR_baseline = sol_baseline.y[0] + sol_baseline.y[2]

#  Heavier Base (+10% Mass in the real world)
params_heavy = final_params.copy()
params_heavy['MB'] = SYSTEM_PARAMS['MB'] * 1.1
sol_heavy = simulate_2dof_system(final_optimal_chirp, params_heavy, (0, SIMULATION_TIME), t_eval_high_res)
abs_XR_heavy = sol_heavy.y[0] + sol_heavy.y[2]

#  Lighter Base (-10% Mass in the real world)
params_light = final_params.copy()
params_light['MB'] = SYSTEM_PARAMS['MB'] * 0.9
sol_light = simulate_2dof_system(final_optimal_chirp, params_light, (0, SIMULATION_TIME), t_eval_high_res)
abs_XR_light = sol_light.y[0] + sol_light.y[2]

plt.figure(figsize=(10, 6))

plt.plot(sol_baseline.t, abs_XR_baseline, 'b-', linewidth=3, label=f'Baseline Mass ({SYSTEM_PARAMS["MB"]}kg)')
plt.plot(sol_heavy.t, abs_XR_heavy, 'r--', linewidth=2, label=f'Heavier Base (+10%: {params_heavy["MB"]}kg)')
plt.plot(sol_light.t, abs_XR_light, 'g:', linewidth=2, label=f'Lighter Base (-10%: {params_light["MB"]}kg)')

plt.axhline(TARGET_XD, color='k', linestyle='--', linewidth=2, label=f'Target Outreach ($X_D$ = {TARGET_XD}m)')

plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Absolute Robot Position (m)', fontsize=12)
plt.title('Robustness Analysis: Open-Loop Vulnerability to Mass Error', fontsize=14, weight='bold')
plt.legend(loc='lower right', framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'robustness_analysis_plot.png'), dpi=300)
plt.show()

print("\nRobustness Results:")
print(f"  Error with Perfect Mass: {abs(np.max(abs_XR_baseline) - TARGET_XD):.5f} m")
print(f"  Error with +10% Mass:    {abs(np.max(abs_XR_heavy) - TARGET_XD):.5f} m")
print(f"  Error with -10% Mass:    {abs(np.max(abs_XR_light) - TARGET_XD):.5f} m")



#  Generating Animation

print("\nStep 5: Generating Slow-Motion Animation (This will take a minute or two to render)...")
import matplotlib.animation as animation

def create_spring(start_x, end_x, nodes=21, width=0.08):
    x = np.linspace(start_x, end_x, nodes)
    y = np.zeros_like(x)
    for i in range(1, nodes - 1):
        y[i] = width if i % 2 == 1 else -width
    return x, y

fig, ax = plt.subplots(figsize=(12, 4))

ax.set_xlim(-0.1, TARGET_XD + 0.15)
ax.set_ylim(-0.5, 0.5)
ax.yaxis.set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xlabel("Absolute Position (m)", fontsize=12)

ax.axvline(0, color='black', lw=4, label='Wall', zorder=1)

ax.axvline(NOMINAL_LIMIT, color='orange', linestyle=':', lw=2.5, zorder=1)
ax.text(NOMINAL_LIMIT, 0.4, f'Nominal Limit ({NOMINAL_LIMIT}m)', color='orange', ha='center', fontsize=10, weight='bold')

ax.axvline(TARGET_XD, color='green', linestyle='--', lw=2.5, zorder=1)
ax.text(TARGET_XD, -0.4, f'Target ({TARGET_XD}m)', color='green', ha='center', fontsize=12, weight='bold')

base_spring_line, = ax.plot([], [], color='tab:orange', lw=2, zorder=2)
robot_spring_line, = ax.plot([], [], color='tab:green', lw=2, zorder=2)

cmd_line = ax.axvline(0, color='blue', linestyle='--', lw=2, alpha=0.5, zorder=3)

peak_line = ax.axvline(0, color='purple', linestyle=':', lw=1.5, alpha=0.8, zorder=3)
peak_text = ax.text(0, 0.15, '', color='purple', ha='left', fontsize=9, weight='bold')

base_mass_marker, = ax.plot([], [], 's', color='tab:red', markersize=22, zorder=4)
robot_mass_marker, = ax.plot([], [], 's', color='tab:purple', markersize=22, zorder=4)

info_text = ax.text(0.01, 0.85, '', transform=ax.transAxes, fontsize=11, fontfamily='monospace', weight='bold')

simulation_fps = 75 
playback_fps = 25
dt_anim = 1.0 / simulation_fps
t_frames = np.arange(0, SIMULATION_TIME, dt_anim)

xb_anim = np.interp(t_frames, sol.t, sol.y[0])
xr_anim = np.interp(t_frames, sol.t, sol.y[2])
xc_anim = [final_optimal_chirp(t) for t in t_frames]

max_reach_tracker = [0.0]

def update_frame(frame):
    t = t_frames[frame]
    xb = xb_anim[frame]
    xr = xr_anim[frame]
    xc = xc_anim[frame]
    
    y_robot = xb + xr
    abs_cmd = xb + xc

    if y_robot > max_reach_tracker[0]:
        max_reach_tracker[0] = y_robot
    
    peak_line.set_xdata([max_reach_tracker[0]])
    peak_text.set_position((max_reach_tracker[0] + 0.01, 0.15))
    peak_text.set_text(f'Max: {max_reach_tracker[0]:.3f}m')

    bx, by = create_spring(0, xb, nodes=15)
    base_spring_line.set_data(bx, by)
    rx, ry = create_spring(xb, y_robot, nodes=25)
    robot_spring_line.set_data(rx, ry)

    base_mass_marker.set_data([xb], [0])
    robot_mass_marker.set_data([y_robot], [0])

    cmd_line.set_xdata([abs_cmd])

    text_color = 'green' if y_robot > NOMINAL_LIMIT else 'black'
    info_text.set_color(text_color)
    info_text.set_text(f"t={t:05.2f}s | Base={xb:05.3f}m | Absolute Robot={y_robot:05.3f}m")

    return (base_spring_line, robot_spring_line, base_mass_marker, 
            robot_mass_marker, cmd_line, peak_line, peak_text, info_text)

ani = animation.FuncAnimation(fig, update_frame, frames=len(t_frames), blit=True)

gif_filename = os.path.join(output_dir, 'robot_outreach_slowmo.gif')
ani.save(gif_filename, writer='pillow', fps=playback_fps)

plt.close(fig)
print(f"Slow-motion animation saved as '{gif_filename}'!")




print("\n" + "-"*50)
run_rl = input("Do you want to run the Residual RL training? (y/n): ").strip().lower()

if run_rl == 'n':
    exit()



print("\nStep 6: Initializing Residual RL Environment...")
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

class ResidualRobustnessEnv(gym.Env):
    def __init__(self, best_bo_params, base_sys_params):
        super(ResidualRobustnessEnv, self).__init__()
        self.best = best_bo_params
        self.base_sys = base_sys_params.copy()
        
        self.action_space = spaces.Box(low=-0.15, high=0.15, shape=(1,), dtype=np.float32)
        
        # Observation Space: [xb, xb_dot, xr, xr_dot, xc_bo, tau, current_MB]
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32) For vanilla PPO
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)  # For LSTM PPO
        self.dt = 0.02 
        self.max_steps = int(SIMULATION_TIME / self.dt)
        self.previous_action = 0.0
        self.smoothed_action = 0.0
        
        self.target_xd = RL_TARGET_XD
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0.0
        self.current_step = 0
        
        initial_cmd = self.get_bo_command(0.0)
        self.state = np.array([0.0, 0.0, initial_cmd, 0.0], dtype=np.float32) 
        
        self.max_reach_achieved = 0.0
        self.previous_action = 0.0
        self.smoothed_action = 0.0
        
        random_mass_variance = np.random.uniform(0.75, 1.25)
        self.current_MB = self.base_sys['MB'] * random_mass_variance
        
        xc_bo = self.get_bo_command(0.0)
        obs = np.array([
            self.state[0], self.state[1], self.state[2], self.state[3],
            xc_bo, 0.0 ], dtype=np.float32)
        
        # for vanilla PPO
        # obs = np.array([
        #     self.state[0], self.state[1], self.state[2], self.state[3],
        #     xc_bo, 0.0, self.current_MB / 100.0
        # ], dtype=np.float32) 
        

        return obs, {}

    def get_bo_command(self, t):
        safe_t = max(0.0, t)
        fade_in = min(safe_t / 0.5, 1.0)
        
        if safe_t <= self.best['t1']:
            freq = self.best['f0'] + ((self.best['f1'] - self.best['f0']) / self.best['t1']) * safe_t
            return self.best['offset'] + (self.best['amp'] * np.sin(2 * np.pi * freq * safe_t)) * fade_in
        else:
            freq_end = self.best['f1']
            val_at_t1 = self.best['offset'] + self.best['amp'] * np.sin(2 * np.pi * freq_end * self.best['t1'])
            return val_at_t1

    def step(self, action):
        residual_cmd = float(action[0])
        alpha = 0.5
        self.smoothed_action = (1 - alpha) * self.smoothed_action + alpha * residual_cmd # Exponential moving average for smoothing

        def step_dynamics(t, y):
            xb, xb_dot, xr, xr_dot = y
            
            actual_DR = 2.0 * self.best['DR'] * np.sqrt(self.best['KR'] * self.base_sys['MR'])
            
            dt_calc = 1e-4
            bo_cmd_t = self.get_bo_command(t)
            bo_cmd_next = self.get_bo_command(t + dt_calc)
            xc_dot = (bo_cmd_next - bo_cmd_t) / dt_calc
            
            offset = self.best['offset']
            #           Dc part   Ac part(scaled by smoothed action)  --->  (X_command = X_static + [X_oscillatory * RL_Gain])
            final_cmd = offset + (bo_cmd_t - offset) * (1.0 + self.smoothed_action)
            final_cmd = np.clip(final_cmd, MIN_FOLD_LIMIT, NOMINAL_LIMIT)
            
            F_imp = -self.best['KR'] * (xr - final_cmd) - actual_DR * (xr_dot - xc_dot)
            
            K_wall, D_wall = 100000.0, 5000.0
            
            F_wall_internal = 0.0
            if xr < MIN_FOLD_LIMIT:
                F_wall_internal = -K_wall * (xr - MIN_FOLD_LIMIT) - D_wall * xr_dot

            F_wall_stretch = 0.0
            if xr > NOMINAL_LIMIT:
                F_wall_stretch = -K_wall * (xr - NOMINAL_LIMIT) - D_wall * xr_dot
            
            F_total_internal = F_imp + F_wall_internal + F_wall_stretch
            
            xb_ddot = (-self.base_sys['BB'] * xb_dot - self.base_sys['KB'] * xb - F_total_internal) / self.current_MB
            xr_ddot = (F_total_internal / self.base_sys['MR']) - xb_ddot
        
            return [xb_dot, xb_ddot, xr_dot, xr_ddot]

        sol = solve_ivp(step_dynamics, [self.t, self.t +  self.dt], self.state, method='Radau', max_step=0.01)
        
        self.state = sol.y[:, -1]
        self.t += self.dt
        self.current_step += 1
        
        absolute_outreach = self.state[0] + self.state[2]
        
        reward = 0.0
        done = self.current_step >= self.max_steps

        if absolute_outreach > self.max_reach_achieved:
            self.max_reach_achieved = absolute_outreach

        # 1. Peak-tracking shaping: symmetric cost on distance from target.
        if absolute_outreach > (self.target_xd - 0.10):
            err_now = absolute_outreach - self.target_xd    
            reward += 50.0 * np.exp(-(err_now ** 2) / (0.02 ** 2)) # Gaussian shaped reward around the target
            reward -= (err_now ** 2) * 5000.0  # Quadratic penalty for being far from the target

        # 2. Action smoothing (kept tiny so it can't dominate the shaping)
        reward += -0.001 * (action[0] ** 2)   # L2 regularization on the action magnitude

        # 3. Kinematic wall penalties
        if self.state[2] > NOMINAL_LIMIT:
            reward -= 500.0 * ((self.state[2] - NOMINAL_LIMIT) ** 2)
        if self.state[2] < MIN_FOLD_LIMIT:
            reward -= 10000.0 * (MIN_FOLD_LIMIT - self.state[2])

        # 4. Terminal precision payout
        if done:
            err = self.max_reach_achieved - self.target_xd
            reward += 2000.0 * np.exp(-(err ** 2) / (0.015 ** 2))   # reward at the end of the episode

        xc_bo = self.get_bo_command(self.t)
        tau = self.t / SIMULATION_TIME
        
        obs = np.array([
            self.state[0], 
            self.state[1], 
            self.state[2], 
            self.state[3], 
            xc_bo, 
            tau
        ], dtype=np.float32) # for LSTM PPO (without mass info)

        # obs = np.array([
        #     self.state[0], 
        #     self.state[1], 
        #     self.state[2], 
        #     self.state[3], 
        #     xc_bo, 
        #     tau,
        #     self.current_MB/100
        # ], dtype=np.float32) # for vanilla PPO (with mass info)
        
        return obs, reward, done, False, {}


# vanilla PPO Training (without LSTM)
# print("Initializing PPO Agent...")
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# env = DummyVecEnv([lambda: ResidualRobustnessEnv(best, SYSTEM_PARAMS)])
# env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0)

# model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, gamma=0.999)

# print("Training Residual RL Agent to compensate for random mass variations...")
# model.learn(total_timesteps=750000)
# print("Training Complete!")


print("Initializing Recurrent PPO Agent...")
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = DummyVecEnv([lambda: ResidualRobustnessEnv(best, SYSTEM_PARAMS)])
env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0)

model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, gamma=0.999)

print("Training Recurrent RL Agent to compensate for random mass variations...")
model.learn(total_timesteps=1500000)  
print("Training Complete!")



# Tiered Evaluation

print("\nStep 7: Running Tiered Validation Strategy...")

validation_tiers = [0.90, 0.95, 1.0, 1.05, 1.10]

class EvalResidualEnv(ResidualRobustnessEnv):
    def __init__(self, best_bo_params, base_sys_params, test_mass):
        super().__init__(best_bo_params, base_sys_params)
        self.test_mass = test_mass
        
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.current_MB = self.test_mass
        return obs, info

print(f"{'Tier':<10} | {'Mass Error':<12} | {'Open-Loop (BO)':<18} | {'Closed-Loop (RL)':<18}")
print("-" * 65)

results = {}

for multiplier in validation_tiers:
    test_mass = SYSTEM_PARAMS['MB'] * multiplier
    
    params_test = final_params.copy()
    params_test['MB'] = test_mass
    sol_ol = simulate_2dof_system(final_optimal_chirp, params_test, (0, SIMULATION_TIME), t_eval_high_res)
    peak_ol = np.max(sol_ol.y[0] + sol_ol.y[2])
    
    error_ol = abs(peak_ol - TARGET_XD) 
    
    eval_env = EvalResidualEnv(best, SYSTEM_PARAMS, test_mass=test_mass)
    obs, _ = eval_env.reset()
    
    rl_t, rl_xb, rl_xr, rl_cmd = [], [], [], []


    lstm_states = None
    episode_start = np.array([True])
    done = False

    while not done:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_start, deterministic=True
        )
        episode_start = np.array([False])
        
        rl_t.append(eval_env.t)
        rl_xb.append(eval_env.state[0])
        rl_xr.append(eval_env.state[2])
        
        expected_smoothed = (1 - 0.5) * eval_env.smoothed_action + 0.5 * float(action[0])
        bo_cmd = eval_env.get_bo_command(eval_env.t)
        _off = eval_env.best['offset']
        final_cmd = np.clip(_off + (bo_cmd - _off) * (1.0 + expected_smoothed), MIN_FOLD_LIMIT, NOMINAL_LIMIT)
        rl_cmd.append(final_cmd)
        
        obs, reward, done, _, _ = eval_env.step(action)

    # for vanilla PPO (without LSTM)
    # while not done:
    #     action, _ = model.predict(obs, deterministic=True)
        
    #     rl_t.append(eval_env.t)
    #     rl_xb.append(eval_env.state[0])
    #     rl_xr.append(eval_env.state[2])
        
    #     expected_smoothed = (1 - 0.5) * eval_env.smoothed_action + 0.5 * float(action[0])
    #     bo_cmd = eval_env.get_bo_command(eval_env.t)
    #     _off = eval_env.best['offset']
    #     final_cmd = np.clip(_off + (bo_cmd - _off) * (1.0 + expected_smoothed), MIN_FOLD_LIMIT, NOMINAL_LIMIT)
    #     rl_cmd.append(final_cmd)
        
    #     obs, reward, done, _, _ = eval_env.step(action)
        
        
    peak_rl = np.max(np.array(rl_xb) + np.array(rl_xr))
    
    error_rl = abs(peak_rl - TARGET_XD) 
    
    results[multiplier] = {
        't': np.array(rl_t),
        'ol_traj': sol_ol.y[0] + sol_ol.y[2],
        'rl_traj': np.array(rl_xb) + np.array(rl_xr),
        'rl_xr': np.array(rl_xr),  
        'rl_cmd': np.array(rl_cmd)
    }
    
    tier_name = f"Tier {validation_tiers.index(multiplier) + 1}"
    error_str = f"{int(round((multiplier-1)*100)):+d}%"
    print(f"{tier_name:<10} | {error_str:<12} | {peak_ol:.4f}m (err: {error_ol:.4f}) | {peak_rl:.4f}m (err: {error_rl:.4f})")


# Static Comparative Plot
print("\nStep 8: Generating Comparative Trajectory Plots for all Tiers...")

for i, multiplier in enumerate(validation_tiers):
    plt.figure(figsize=(10, 5))

    t_plot_rl = results[multiplier]['t']
    ol_traj = results[multiplier]['ol_traj']
    rl_traj = results[multiplier]['rl_traj']

    mass_error_pct = int(round((multiplier - 1.0) * 100))
    current_mass = SYSTEM_PARAMS['MB'] * multiplier

    plt.plot(t_eval_high_res, ol_traj, color='black', alpha=0.5, linestyle='--', linewidth=2, label='Open-Loop BO (Baseline)')
    plt.plot(t_plot_rl, rl_traj, color='blue', linewidth=2.5, label='Closed-Loop RL (Corrected)')
    
    plt.axhline(TARGET_XD, color='green', linestyle=':', linewidth=2.5, label=f'Target ({TARGET_XD}m)')
    plt.axhline(NOMINAL_LIMIT, color='orange', linestyle=':', linewidth=2, label=f'Nominal Limit ({NOMINAL_LIMIT}m)')

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Absolute Robot Position (m)', fontsize=12)
    plt.title(f'RL vs BO Trajectory: +{mass_error_pct}% Mass Error ({current_mass:.1f}kg)', fontsize=14, weight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    filename = f'rl_vs_bo_trajectory_tier{i+1}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close() 
    print(f"Comparative Plot saved as '{filename}'!") 


print("-" * 65)

print("\nStep 9: Generating Relative Extension Proof for RL...")

for i, multiplier in enumerate(validation_tiers):
    plt.figure(figsize=(10, 4))
    
    t_plot = results[multiplier]['t']
    xr_plot = results[multiplier]['rl_xr']
    cmd_plot = results[multiplier]['rl_cmd']
    
    mass_error_pct = int(round((multiplier - 1.0) * 100))
    current_mass = SYSTEM_PARAMS['MB'] * multiplier

    plt.plot(t_plot, xr_plot, 'm-', linewidth=2.5, label='RL Relative Arm Stretch ($X_R$)')
    plt.plot(t_plot, cmd_plot, 'k--', alpha=0.6, linewidth=2, label='RL Final Motor Command ($X_C$)')
    
    plt.axhline(NOMINAL_LIMIT, color='red', linestyle='-', linewidth=2, label=f'Max Stretch Limit ({NOMINAL_LIMIT}m)')
    plt.axhline(MIN_FOLD_LIMIT, color='red', linestyle='-', linewidth=2, label=f'Min Fold Limit ({MIN_FOLD_LIMIT}m)')
    
    plt.fill_between(t_plot, MIN_FOLD_LIMIT, NOMINAL_LIMIT, color='green', alpha=0.1, label='Safe Operating Zone')
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Relative Extension (m)', fontsize=12)
    plt.title(f'RL Constraint Proof: {mass_error_pct:+d}% Mass Error ({current_mass:.1f}kg)', fontsize=14, weight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    filename = f'rl_relative_stretch_tier{i+1}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close() 
    print(f"RL Constraint Plot saved as '{filename}'!")

#  Generate RL Animation


print(f"\nStep 10: Generating Clean RL Slow-Motion Animation (Disturbance: {eval_env.current_MB:.1f}kg)...")

import matplotlib.animation as animation

def create_spring(start_x, end_x, nodes=21, width=0.08):
    x = np.linspace(start_x, end_x, nodes)
    y = np.zeros_like(x)
    for i in range(1, nodes - 1):
        y[i] = width if i % 2 == 1 else -width
    return x, y

fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(-0.1, eval_env.target_xd + 0.15)
ax.set_ylim(-0.5, 0.5)
ax.yaxis.set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xlabel("Absolute Position (m)", fontsize=12)

ax.axvline(0, color='black', lw=4, label='Wall', zorder=1)

ax.axvline(NOMINAL_LIMIT, color='orange', linestyle=':', lw=2.5, zorder=1)
ax.text(NOMINAL_LIMIT, 0.4, f'Nominal Limit ({NOMINAL_LIMIT}m)', color='orange', ha='center', fontsize=10, weight='bold')

target_achieved = eval_env.target_xd
ax.axvline(target_achieved, color='green', linestyle='--', lw=2.5, zorder=1)
ax.text(target_achieved, -0.4, f'Target ({target_achieved}m)', color='green', ha='center', fontsize=12, weight='bold')

base_spring_line, = ax.plot([], [], color='tab:orange', lw=2, zorder=2)
robot_spring_line, = ax.plot([], [], color='tab:green', lw=2, zorder=2)
cmd_line = ax.axvline(0, color='blue', linestyle='--', lw=2, alpha=0.5, zorder=3)

peak_line = ax.axvline(0, color='purple', linestyle=':', lw=1.5, alpha=0.8, zorder=3)
peak_text = ax.text(0, 0.15, '', color='purple', ha='left', fontsize=9, weight='bold')

base_mass_marker, = ax.plot([], [], 's', color='tab:red', markersize=22, zorder=4)
robot_mass_marker, = ax.plot([], [], 's', color='tab:purple', markersize=22, zorder=4)

info_text = ax.text(0.01, 0.85, '', transform=ax.transAxes, fontsize=11, fontfamily='monospace', weight='bold')

simulation_fps = 75 
dt_anim = 1.0 / simulation_fps
t_frames = np.arange(0, SIMULATION_TIME, dt_anim)

xb_anim_rl = np.interp(t_frames, rl_t, rl_xb)
xr_anim_rl = np.interp(t_frames, rl_t, rl_xr)

xc_anim_rl = []
for t_val in t_frames:
    idx = (np.abs(np.array(rl_t) - t_val)).argmin()
    applied_cmd = rl_cmd[idx] 
    xc_anim_rl.append(applied_cmd)

rl_max_tracker = [0.0]

def update_frame_rl(frame):
    t = t_frames[frame]
    
    xb = xb_anim_rl[frame]
    xr = xr_anim_rl[frame]
    xc = xc_anim_rl[frame]
    
    y_robot = xb + xr
    abs_cmd = xb + xc

    if y_robot > rl_max_tracker[0]:
        rl_max_tracker[0] = y_robot
    
    peak_line.set_xdata([rl_max_tracker[0]])
    peak_text.set_position((rl_max_tracker[0] + 0.01, 0.15))
    peak_text.set_text(f'Max: {rl_max_tracker[0]:.3f}m')

    bx, by = create_spring(0, xb, nodes=15)
    base_spring_line.set_data(bx, by)
    rx, ry = create_spring(xb, y_robot, nodes=25)
    robot_spring_line.set_data(rx, ry)

    base_mass_marker.set_data([xb], [0])
    robot_mass_marker.set_data([y_robot], [0])
    cmd_line.set_xdata([abs_cmd])
    
    text_color = 'green' if y_robot > NOMINAL_LIMIT else 'black'
    info_text.set_color(text_color)
    info_text.set_text(f"RL AGENT | M_B={eval_env.current_MB:.1f}kg | t={t:05.2f}s | Robot={y_robot:05.3f}m")
    
    return (base_spring_line, robot_spring_line, base_mass_marker, robot_mass_marker, 
            cmd_line, peak_line, peak_text, info_text)

ani_rl = animation.FuncAnimation(fig, update_frame_rl, frames=len(t_frames), blit=True)
gif_rl_filename = os.path.join(output_dir, f'robot_outreach_rl_clean_{eval_env.current_MB:.1f}kg.gif')
ani_rl.save(gif_rl_filename, writer='pillow', fps=25)
plt.close(fig)
print(f"Clean RL Animation saved as '{gif_rl_filename}'!")