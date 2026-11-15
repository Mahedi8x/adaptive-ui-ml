import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths
BASE_DIR = r"C:\Users\Mehedi\adavtive_model"
OUT_DIR = os.path.join(BASE_DIR, "member1_deliverables")
os.makedirs(OUT_DIR, exist_ok=True)

print("="*60)
print("  Member 1: Online RL Policy Agent (LinUCB & Usability Reward)")
print("="*60)

# Action Space:
# 0: Keep current size (Stability)
# 1: Increase button size (+8dp)
# 2: Decrease button size (-8dp)
ACTIONS = [0, 1, 2]
ACTION_NAMES = {0: "No Change", 1: "Increase Target Size", 2: "Decrease Target Size"}

class LinUCBAdaptiveAgent:
    """
    Contextual Bandit Agent with Linear Upper Confidence Bound (LinUCB)
    for real-time on-device UI adaptation.
    """
    def __init__(self, n_actions=3, d_features=4, alpha=0.5):
        self.n_actions = n_actions
        self.d = d_features
        self.alpha = alpha
        # A matrix: d x d for each action
        self.A = [np.identity(self.d) for _ in range(n_actions)]
        # b vector: d x 1 for each action
        self.b = [np.zeros((self.d, 1)) for _ in range(n_actions)]

    def select_action(self, context_vector):
        x = context_vector.reshape(-1, 1)
        p = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv @ self.b[a]
            # UCB score = expected reward + uncertainty exploration
            confidence = self.alpha * np.sqrt((x.T @ A_inv @ x).item())
            p[a] = (theta_a.T @ x).item() + confidence
            
        return int(np.argmax(p))

    def update(self, action, context_vector, reward):
        x = context_vector.reshape(-1, 1)
        self.A[action] += x @ x.T
        self.b[action] += reward * x

    def save_policy(self, path):
        policy_dict = {
            "alpha": self.alpha,
            "d": self.d,
            "A": [a.tolist() for a in self.A],
            "b": [b.tolist() for b in self.b],
            "actions": ACTION_NAMES
        }
        with open(path, "w") as f:
            json.dump(policy_dict, f, indent=4)
        print(f"[Export] Saved RL Policy JSON: {path}")

# Usability Reward Function:
# R = 0.40 * Accuracy - 0.30 * MissRate - 0.15 * ReactionTimeNorm - 0.05 * ChangeCost
def calculate_reward(tap_acc, miss_rate, reaction_norm, action_taken):
    change_cost = 0.05 if action_taken != 0 else 0.0
    r = (0.40 * tap_acc) - (0.30 * miss_rate) - (0.15 * reaction_norm) - change_cost
    return float(r)

# Simulation: Static UI vs Cold-Start Only vs Cold-Start + RL
np.random.seed(42)
n_steps = 500

# Synthetic stream of user interaction contexts (simulating noisy touch & walking fatigue)
# Context: [miss_rate, reaction_time_norm, current_button_size_norm, is_walking]
true_miss_rates = np.clip(np.sin(np.linspace(0, 8, n_steps)) * 0.15 + 0.12, 0.02, 0.35)
reaction_times = np.clip(np.cos(np.linspace(0, 6, n_steps)) * 0.2 + 0.5, 0.1, 0.9)
is_walking = (np.random.rand(n_steps) > 0.6).astype(int)

# 1. Baseline: Static UI (Always Size 1 = Default 48dp)
static_rewards = []
for i in range(n_steps):
    m_rate = true_miss_rates[i]
    acc = 1.0 - m_rate
    r = calculate_reward(acc, m_rate, reaction_times[i], action_taken=0)
    static_rewards.append(r)

# 2. Cold-Start Only: Predicts initial Size 2 (Large 56dp), but never adapts online
cold_only_rewards = []
for i in range(n_steps):
    # slightly better baseline miss rate due to larger buttons, but still static
    m_rate = max(0.02, true_miss_rates[i] - 0.04)
    acc = 1.0 - m_rate
    r = calculate_reward(acc, m_rate, reaction_times[i], action_taken=0)
    cold_only_rewards.append(r)

# 3. Hybrid: Cold-Start + Online RL (LinUCB)
agent = LinUCBAdaptiveAgent(n_actions=3, d_features=4, alpha=0.3)
rl_rewards = []
cur_size = 2 # Start from Cold-Start decision (Large)

for i in range(n_steps):
    ctx = np.array([true_miss_rates[i], reaction_times[i], cur_size / 3.0, is_walking[i]])
    action = agent.select_action(ctx)
    
    # Execute Action
    if action == 1 and cur_size < 3: # Increase
        cur_size += 1
    elif action == 2 and cur_size > 0: # Decrease
        cur_size -= 1
        
    # Adapting size reduces miss rate according to Fitts's law
    effective_miss = max(0.01, true_miss_rates[i] - (cur_size * 0.035))
    effective_acc = 1.0 - effective_miss
    
    reward = calculate_reward(effective_acc, effective_miss, reaction_times[i], action_taken=action)
    agent.update(action, ctx, reward)
    rl_rewards.append(reward)

# Save RL Policy
policy_out = os.path.join(OUT_DIR, "online_rl_linucb_policy.json")
agent.save_policy(policy_out)

# 4. Generate Paper Graph: Cumulative Reward Comparison
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(static_rewards), label='Static Default UI', color='#d62728', linestyle='--', linewidth=2)
plt.plot(np.cumsum(cold_only_rewards), label='Cold-Start ML Only', color='#ff7f0e', linestyle='-.', linewidth=2)
plt.plot(np.cumsum(rl_rewards), label='Proposed Hybrid (Cold-Start + Online RL)', color='#2ca02c', linewidth=2.5)

plt.title('Cumulative Usability Reward: Baseline vs Proposed Framework', fontsize=14, pad=15)
plt.xlabel('Interaction Steps (Touch Events)', fontsize=12)
plt.ylabel('Cumulative Reward', fontsize=12)
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

chart_path = os.path.join(OUT_DIR, "rl_cumulative_reward_comparison.png")
plt.savefig(chart_path, dpi=300)
plt.close()
print(f"[Paper Graph] Saved Cumulative Reward Graph: {chart_path}")

# Summary Metrics Table
summary_df = pd.DataFrame({
    'Method': ['Static Default UI', 'Cold-Start ML Only', 'Hybrid (Cold-Start + Online RL)'],
    'Average_Reward_Per_Step': [np.mean(static_rewards), np.mean(cold_only_rewards), np.mean(rl_rewards)],
    'Total_Cumulative_Reward': [np.sum(static_rewards), np.sum(cold_only_rewards), np.sum(rl_rewards)],
    'Improvement_Over_Static (%)': [0.0, 
                                     ((np.sum(cold_only_rewards) - np.sum(static_rewards)) / np.sum(static_rewards)) * 100,
                                     ((np.sum(rl_rewards) - np.sum(static_rewards)) / np.sum(static_rewards)) * 100]
})
summary_csv = os.path.join(OUT_DIR, "framework_comparison_results.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"[Paper Table] Saved Comparison Table: {summary_csv}")
print("\n" + summary_df.to_string(index=False))
