# Resonant Slingshot Control — 2DOF Underactuated System

A two-layer ML pipeline for controlling a 2DOF underactuated arm to reach 
targets **beyond its 0.70 m kinematic limit** by exciting the compliant base 
at resonance and releasing stored momentum.

<img width="1349" height="446" alt="systemSchematic" src="https://github.com/user-attachments/assets/aab1ce0f-b2a9-45a8-96d2-fc7c4c13a94b" />


## Method

**Layer A — Bayesian Optimization:** Optimizes a 7-parameter "Chirp-and-Hold" 
open-loop command to drive the system into resonance for a nominal 70 kg base.

**Layer B — Residual RL:** A parameter-aware MLP and sensorless recurrent LSTM 
policy (PPO) for disturbance rejection on top of the BO command, validated 
across ±10% base-mass variation.

**Target Generalization:** A forward surrogate model learns the command→peak map 
from random simulations, then inverts it via CEM search to reach arbitrary targets.

## Results

- ~3.2 mm worst-case error across 0.72–0.82 m target range
- Robust to ±10% base-mass variation without retraining

## Run Order

1. `MainCode.py` — reproduces BO + RL results
2. `forward_surrogate.py` — trains/saves the surrogate (pre-trained `.pkl` provided)
3. `Invert_targeted.py` — hybrid inversion for arbitrary targets

## Requirements
pip install numpy scipy scikit-learn matplotlib gymnasium stable-baselines3 sb3-contrib

*Course project — M.Sc. Mechanical Engineering (Robotics & Mechatronics), 
Politecnico di Milano, 2026*
