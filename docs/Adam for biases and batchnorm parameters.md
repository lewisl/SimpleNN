That's a sharp question, and the answer is nuanced. The use of **Adam for biases and BatchNorm parameters (`γ`, `β`)** is still common and **not deprecated**, but it comes with **practical trade-offs** that depend on model architecture, dataset, and training stability.

---

## ✅ **Default Practice Today**

Most modern frameworks (like PyTorch, TensorFlow, and JAX) **do apply Adam updates to all parameters** — including biases, BatchNorm `γ` and `β`. This means that:

* Biases and BN params **get adaptive learning rates** via `m_t`, `v_t` estimation
* But they **do not receive weight decay** (e.g. they are excluded from the `λ * θ` term in AdamW)

This behavior is considered **standard practice** and is baked into tools like:

```python
# PyTorch typical example:
optimizer = AdamW([
    {'params': model.weight_params, 'weight_decay': 0.01},
    {'params': model.bias_and_bn_params, 'weight_decay': 0.0}
])
```

---

## 🧪 **Why Use Adam on Biases / BN Parameters?**

Biases and batchnorm parameters are:

* Low-dimensional (often 1D)
* Crucial for shifting and scaling the activations
* Susceptible to gradient scale mismatches

Using Adam's **adaptive learning rate per parameter** can help mitigate gradient scale imbalances, especially early in training. This can improve stability and convergence speed.

---

## ⚠️ **When It Can Be Harmful**

There are situations where **applying Adam to biases or BN parameters is problematic**:

### 1. **Overfitting or poor generalization**

* Adaptive optimizers like Adam can cause smaller parameters (e.g. biases or BN γ/β) to be updated too aggressively.
* This leads to worse generalization compared to SGD-based methods (see: *Wilson et al., 2017* — “The marginal value of adaptive gradient methods in deep learning”).

### 2. **Vanishing scale (`γ → 0`)**

* Applying weight decay or overly aggressive adaptation to BatchNorm scale (`γ`) can lead it to shrink toward zero, **effectively canceling the normalization effect**.
* This is one reason why BN parameters are typically excluded from weight decay and sometimes from adaptation altogether.

### 3. **Small-batch instability**

* In low-batch-size settings, using Adam on `γ` and `β` can interact badly with noisy BN statistics and hurt convergence.

---

## ✅ **Alternatives and Fine-tuning Strategies**

Some strategies include:

* **Use Adam on all parameters initially**, then switch to SGD for final convergence (e.g. common in fine-tuning).
* **Use different learning rates**:

  ```julia
  γ, β = BN parameters
  if using Adam:
      use η_small = η * 0.1 for γ, β
  ```
* **Freeze BN parameters (`γ`, `β`)** during fine-tuning:

  * Common when transferring a pre-trained model to a new task.

---

## 🔎 Empirical Summary

| Parameter  | Adaptive Updates (Adam) | Weight Decay | Notes                                    |
| ---------- | ----------------------- | ------------ | ---------------------------------------- |
| Weights    | Yes                     | Yes          | Standard practice                        |
| Biases     | Yes                     | No           | Helps early training; can overfit        |
| γ / β (BN) | Yes                     | No           | Works, but requires care; can be fragile |

---

## ✅ Recommendation

* **Training from scratch?** Adam is usually fine for biases and BN parameters — just make sure weight decay is off.
* **Doing fine-tuning or chasing generalization?** Consider:

  * Switching to SGD later
  * Lower learning rates for γ/β
  * Freezing BN parameters

Let me know if you want a Julia version of optimizers with per-parameter group control for decay or learning rates.