# Perceptron vs Logistic Regression vs Gradient Descent vs Scikit-Learn

This project visually compares how different linear classification algorithms learn a **decision boundary** on the same dataset.

Models compared:

1. Perceptron (Step function)
2. Logistic Regression using Gradient Descent
3. Scikit-Learn Logistic Regression

---

# 1. Dataset Generation

```python
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=1,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=1,
    random_state=41,
    hypercube=False,
    class_sep=10
)
```

### Meaning of parameters

| Parameter       | Purpose                                  |
| --------------- | ---------------------------------------- |
| n_features=2    | 2D data so we can plot decision boundary |
| n_informative=1 | Only 1 feature determines class          |
| class_sep=10    | Makes data almost perfectly separable    |

---

# 2. Perceptron Algorithm

## Step Function

[
f(z) = \begin{cases} 1 & z>0 \ 0 & z\le 0 \end{cases}
]

## Model Equation

[
z = w^T x + b
]

## Weight Update Rule

[
w = w + \eta (y - \hat{y})x
]

Only misclassified points update weights.

---

# 3. Logistic Regression (Gradient Descent)

## Sigmoid Function

[
\sigma(z) = \frac{1}{1 + e^{-z}}
]

## Prediction

[
\hat{y} = \sigma(w^T x + b)
]

## Loss Function (Log Loss)

[
L = - \frac{1}{m} \sum y\log(\hat{y}) + (1-y)\log(1-\hat{y})
]

## Gradient Descent Update

[
w = w + \eta X^T(y - \hat{y})
]

All points influence learning.

---

# 4. Scikit-Learn Logistic Regression

Scikit-Learn solves:
[
J(w) = \text{Log Loss} + \lambda ||w||^2
]

using advanced solvers like **LBFGS**, not manual gradient descent.

---

# 5. Decision Boundary Equation

All models produce a linear boundary:
[
x_2 = -\frac{w_1}{w_2}x_1 - \frac{b}{w_2}
]

---

# 6. What Each Line Represents in Plot

| Color | Model                    |
| ----- | ------------------------ |
| Red   | Perceptron               |
| Brown | Logistic Regression (GD) |
| Black | Sklearn Logistic         |

---

# 7. Key Learning Differences

| Aspect     | Perceptron        | Logistic Regression |
| ---------- | ----------------- | ------------------- |
| Activation | Step              | Sigmoid             |
| Output     | Class label       | Probability         |
| Loss       | Perceptron loss   | Log Loss            |
| Updates    | Only errors       | All samples         |
| Stability  | Poor with overlap | Stable              |

---

# 8. Why All Lines Look Similar Here

Because `class_sep=10` makes the data **linearly separable**, all models find similar boundaries.

Reduce separation to see differences:

```python
class_sep=1
```

---

# 9. Final Takeaway

Perceptron is the foundation of neural networks, but Logistic Regression is more stable and probabilistic. Scikit-Learn’s model is the optimized real-world implementation.
