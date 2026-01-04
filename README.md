# 🌧️ Neural Network from Scratch: Weather Prediction

> **Deep Learning Assignment 3** — Implementing a 2-layer neural network for rain prediction using temperature and humidity.

---

## 📖 Overview

This project implements a **simple feedforward neural network from scratch** using only NumPy—no deep learning frameworks like TensorFlow or PyTorch. The network learns to predict whether it will rain based on temperature and humidity readings.

The implementation is based on [Victor Zhou's Neural Network Tutorial](https://victorzhou.com/blog/intro-to-neural-networks/), adapted with a **custom weather prediction dataset** to demonstrate understanding of the core concepts.

---

## 🏗️ Network Architecture

```
┌─────────────────┐
│   INPUT LAYER   │
│  (2 neurons)    │
│  • Temperature  │
│  • Humidity     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  HIDDEN LAYER   │
│  (2 neurons)    │
│  • h1 = σ(...)  │
│  • h2 = σ(...)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  OUTPUT LAYER   │
│  (1 neuron)     │
│  • Rain prob.   │
└─────────────────┘
```

| Component          | Description                        |
| ------------------ | ---------------------------------- |
| **Inputs**         | Temperature (°C), Humidity (%)     |
| **Hidden Neurons** | 2 neurons with sigmoid activation  |
| **Output**         | Probability of rain (0-1)          |
| **Activation**     | Sigmoid: `σ(x) = 1 / (1 + e^(-x))` |
| **Loss Function**  | Mean Squared Error (MSE)           |
| **Optimizer**      | Batch Gradient Descent             |

---

## 📊 Dataset

A custom synthetic dataset with 10 training examples relating weather conditions to rain probability:

| Temperature (°C) | Humidity (%) | Rain?  | Description              |
| :--------------: | :----------: | :----: | ------------------------ |
|        30        |      40      | ❌ No  | Hot & dry                |
|        15        |      85      | ✅ Yes | Cool & humid             |
|        25        |      70      | ✅ Yes | Warm & moderate humidity |
|        35        |      30      | ❌ No  | Very hot & dry           |
|        10        |      90      | ✅ Yes | Cold & very humid        |
|        28        |      45      | ❌ No  | Warm & dry               |
|        18        |      80      | ✅ Yes | Cool & humid             |
|        22        |      65      | ✅ Yes | Moderate conditions      |
|        32        |      35      | ❌ No  | Hot & dry                |
|        12        |      88      | ✅ Yes | Cold & humid             |

**Pattern learned:** High humidity (>65%) combined with moderate/low temperature → likely rain.

---

## 🔑 Key Concepts Implemented

### 1. Forward Propagation

Data flows through the network:

```
Input → Hidden Layer → Output Layer → Prediction
```

### 2. Backpropagation

Error propagates backward using the chain rule:

```
Loss → Output Gradients → Hidden Gradients → Weight Updates
```

### 3. Gradient Descent

Weights are updated to minimize loss:

```
new_weight = old_weight - learning_rate × gradient
```

---

## 📈 Results

### Training Performance

- **Initial Loss:** ~0.21
- **Final Loss:** ~0.015 (after 1000 epochs)
- **Accuracy:** 100% on training data

### Loss Curve

The network shows clear convergence with a smooth downward loss curve.

### Predictions on New Data

| Temperature | Humidity | Prediction |  Expected  |
| :---------: | :------: | :--------: | :--------: |
|    20°C     |   75%    |  🌧️ RAIN   | ✅ Correct |
|    33°C     |   38%    | ☀️ NO RAIN | ✅ Correct |
|    16°C     |   82%    |  🌧️ RAIN   | ✅ Correct |
|    27°C     |   50%    | ☀️ NO RAIN | Borderline |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy matplotlib
```

### Run the Notebook

```bash
jupyter notebook 7980-dl-assignment-3.ipynb
```

---

## 📁 Project Structure

```
Assignment 3/
├── 7980-dl-assignment-3.ipynb    # Main implementation notebook
├── 7980_DL_A3.pdf                # Assignment description
├── Assignment 3.pdf              # Additional materials
└── README.md                     # This file
```

---

## 🧠 What I Learned

1. **Neurons compute** weighted sums + bias, then apply activation
2. **Sigmoid activation** enables non-linear learning (values between 0-1)
3. **MSE loss** quantifies prediction error
4. **Backpropagation** efficiently calculates gradients using chain rule
5. **Gradient descent** iteratively minimizes loss
6. **Hidden layers** create useful intermediate representations

---

## 📚 References

- [Victor Zhou: Machine Learning for Beginners: An Introduction to Neural Networks](https://victorzhou.com/blog/intro-to-neural-networks/)
- Course: Deep Learning (7980)

---

## ✍️ Author

Deep Learning Assignment 3 — Term 9

---

<p align="center">
  <i>Built from scratch with ❤️ and NumPy</i>
</p>
