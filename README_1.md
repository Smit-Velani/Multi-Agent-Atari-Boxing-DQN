# 🥊 Multi-Agent Atari Boxing with Deep Q-Learning

Two AI agents learn to play Atari Boxing against each other using Deep Q-Networks (DQN). No human input — both agents learn purely from experience through reinforcement learning.

![Training Results](training_results.png)

---

## 🎯 Project Overview

This project implements a **multi-agent reinforcement learning** system where two independent DQN agents compete in the Atari Boxing environment. Each agent learns to maximize its own reward (landing punches) while minimizing the opponent's score.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Total Episodes | 200 |
| Agent 2 Wins | 39 (62% of decisive games) |
| Agent 1 Wins | 23 (37% of decisive games) |
| Training Time | ~12 minutes (T4 GPU) |
| Max Reward Seen | +6 / -6 |

---

## 🧠 How It Works

### Environment
- **Framework**: PettingZoo Atari Boxing v2 (multi-agent)
- **Observation**: 210×160×3 RGB pixel frame per agent
- **Actions**: 18 discrete actions (move, punch, combinations)
- **Reward**: +1 for landing a punch, -1 for receiving one

### DQN Architecture

```
Input: 4 stacked grayscale frames (4 × 84 × 84)
       ↓
Conv2d(4, 32, kernel=8, stride=4)  → ReLU
       ↓
Conv2d(32, 64, kernel=4, stride=2) → ReLU
       ↓
Conv2d(64, 64, kernel=3, stride=1) → ReLU
       ↓
Linear(3136, 512) → ReLU
       ↓
Linear(512, 18)  ← Q-value for each action
```

### Key Techniques
- **Experience Replay** — stores 10,000 past transitions, samples random batches to break correlation
- **Target Network** — separate frozen network updated every 500 steps for stable training
- **Frame Stacking** — stacks 4 consecutive frames so agent perceives motion
- **Epsilon-Greedy** — starts at ε=1.0 (random), decays to ε=0.1 (mostly learned policy)
- **Bellman Equation** — Q(s,a) = r + γ × max(Q(s',a')) with γ=0.99

---

## 🗂️ Project Structure

```
Multi-Agent-Atari-Boxing-DQN/
│
├── Multi_Agent_Atari_Boxing.ipynb  ← Full training notebook
├── training_results.png            ← Reward curves + win rate plots
├── agent1_dqn.pth                  ← Trained Agent 1 weights
├── agent2_dqn.pth                  ← Trained Agent 2 weights
├── requirements.txt                ← Dependencies
└── README.md
```

---

## ⚙️ Tech Stack

- **Python 3.10**
- **PyTorch** — neural network + training
- **PettingZoo** — multi-agent Atari environment
- **OpenCV** — frame preprocessing
- **Matplotlib** — training visualization
- **Google Colab T4 GPU** — training hardware

---

## 🚀 How to Run

### On Google Colab (recommended)
1. Open `Multi_Agent_Atari_Boxing.ipynb` in Google Colab
2. Set runtime to T4 GPU
3. Run all cells in order

### Local Setup
```bash
pip install -r requirements.txt
AutoROM --accept-license
jupyter notebook Multi_Agent_Atari_Boxing.ipynb
```

---

## 📈 Training Progress

| Episode | Agent 1 Wins | Agent 2 Wins | ε |
|---------|-------------|-------------|-------|
| 20      | 0           | 1           | 0.289 |
| 50      | 6           | 8           | 0.100 |
| 100     | 7           | 17          | 0.100 |
| 150     | 16          | 26          | 0.100 |
| 200     | 23          | 39          | 0.100 |

---

## 💡 Key Learnings

- Multi-agent RL is significantly harder than single-agent — both agents are non-stationary targets for each other
- Frame stacking is essential for the agent to perceive movement and direction
- Epsilon decay speed critically affects learning — too fast and agents never explore enough
- Target networks are essential for stable training in competitive environments

---

## 👤 Author

**Smitkumar Velani**
MS Data Science — Northeastern University
[GitHub](https://github.com/Smit-Velani) | [LinkedIn](www.linkedin.com/in/smit-velani)
