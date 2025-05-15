# How AdamW Updates Neural Network Weights, Biases, Gamma, and Beta

**Assumptions:** We start from a point in training where the gradients for all parameters have been computed via backpropagation. Let:

* **g\_weight**, **g\_bias**, **g\_gamma**, **g\_beta** denote the gradients of the loss with respect to a weight W, a bias b, batch norm scale gamma, and batch norm shift beta, respectively (at the current training step).

AdamW is an optimizer that updates each parameter using these gradients, with adaptive moment estimates (like Adam) and **decoupled weight decay** regularization. Below we provide:

* **A simplified explanation:** showing conceptually how each parameter is adjusted using its gradient and learning rate (and how weight decay is applied).
* **A detailed algebraic explanation:** deriving the AdamW update step-by-step, including first/second moment calculations (m\_t, v\_t), bias correction, learning rate eta and epsilon, and the weight decay term. We will highlight the difference in how weights vs. biases (and batch norm parameters) are handled, and explain why **decoupled weight decay** (AdamW) differs from traditional L2 regularization.

## Simplified Parameter Update (Conceptual)

In simple terms, **AdamW updates each parameter by moving it in the opposite direction of its gradient** (gradient descent step), scaled by the learning rate. For weight parameters, an extra **weight decay** term also pulls the weight slightly toward zero on each update. Here’s a conceptual view for each type of parameter:

* **Weights (W):** Updated by subtracting a scaled gradient and a weight decay term.
  W\_new = W\_old - learning\_rate \* (g\_weight + lambda \* W\_old)
  *Here lambda is the weight decay coefficient. The term lambda \* W\_old causes a direct shrinkage of W (decoupled from the gradient). This is equivalent to multiplying W by a factor slightly less than 1 at each step, nudging weights toward 0 over time.*

* **Biases (b):** Updated by subtracting the scaled gradient (usually **no weight decay** on biases):
  b\_new = b\_old - learning\_rate \* g\_bias

* **BatchNorm Scale (gamma):** Updated like a weight **but usually without decay**:
  gamma\_new = gamma\_old - learning\_rate \* g\_gamma

* **BatchNorm Shift (beta):** Updated like a bias (no decay):
  beta\_new = beta\_old - learning\_rate \* g\_beta

In summary, **each parameter is nudged opposite to its gradient**. Weights get an extra nudge toward zero due to weight decay, whereas biases and batch norm parameters (gamma, beta) typically do not.

## Detailed AdamW Update (Step-by-Step Algebra)

AdamW extends the Adam optimizer with decoupled weight decay. It keeps track of two running averages for each parameter: the *first moment* (similar to momentum) and the *second moment* (similar to RMSprop for gradient magnitudes). Let theta represent any parameter (for example, one weight tensor or one bias scalar), and g\_t = dL/dtheta be its gradient at the current time step t. AdamW update for theta involves the following steps:

1. **Initialize moment estimates (at t = 0):** m\_0 = 0, v\_0 = 0.
   *(Before any updates, the first moment m and second moment v are initialized to zero.)*

2. **First moment update (m\_t – gradient mean estimate):**
   m\_t = beta1 \* m\_{t-1} + (1 - beta1) \* g\_t
   Here beta1 (often 0.9) controls the momentum memory. This is an **exponential moving average of gradients**: it blends the previous m\_{t-1} with the current gradient. Over many iterations, m\_t becomes the estimated mean of recent gradients.

3. **Second moment update (v\_t – gradient variance estimate):**
   v\_t = beta2 \* v\_{t-1} + (1 - beta2) \* g\_t^2
   Here g\_t^2 denotes element-wise squaring of the gradient, and beta2 (often 0.999) controls the averaging of squared gradients. v\_t is an **exponential moving average of g^2**, i.e. it estimates the mean squared gradient (uncentered variance). This tracks the scale of recent gradients for each parameter.

4. **Bias correction:**
   Because m\_0 and v\_0 started at 0, the estimates m\_t and v\_t are biased towards zero initially (especially in early steps when t is small). We correct these as:
   m\_hat = m\_t / (1 - beta1^t)
   v\_hat = v\_t / (1 - beta2^t)
   Here beta1^t means beta1 to the power t (and similarly for beta2^t). For example, after one update (t = 1), 1 - beta1^t = 1 - beta1, so we divide m\_1 by (1 - beta1) to remove the initial bias. m\_hat and v\_hat are now unbiased estimates of the true first and second moment of the gradients.

5. **Compute the Adam update term (gradient step):**
   update = -learning\_rate \* m\_hat / (sqrt(v\_hat) + epsilon)
   Here learning\_rate is the step size and epsilon is a small constant (e.g. 1e-8) to prevent division by zero. This formula means we take a step proportional to m\_hat (the sign and scale of recent gradients), divided by the RMS of recent gradients sqrt(v\_hat) (so if a parameter’s gradients have been large, the step is scaled down). This is the core of Adam: **moving in the direction of the gradient momentum, with step sizes normalized by recent gradient magnitudes**.

6. **Decoupled weight decay step:**
   AdamW now applies **weight decay** directly to the parameter. This is **not done through the gradient** (unlike L2 regularization), but as a separate step that shrinks theta. The weight decay update for a weight decay coefficient lambda is:
   decay = -learning\_rate \* lambda \* theta\_t
   This means we subtract learning\_rate \* lambda \* theta from the parameter, which is equivalent to multiplying theta by (1 - learning\_rate \* lambda) to slightly reduce its magnitude. *This step is typically **only applied to weight parameters** (like weight matrices of layers). For biases and batch norm parameters, we usually set lambda = 0 (no decay), as decaying those can hurt model performance.*

7. **Combine steps for the final parameter update:**
   For a weight parameter (where weight decay applies), the AdamW update rule for theta can be written in one equation as:
   theta\_{t+1} = theta\_t - learning\_rate \* (m\_hat / (sqrt(v\_hat) + epsilon) + lambda \* theta\_t)

   The term in parentheses has **two parts**: the first part is the standard Adam adaptive gradient step, and the second part lambda \* theta\_t is the weight decay contribution. This formulation makes it clear that weight decay is **decoupled** from the gradient-based term. In implementation, this might be done in two lines (first the Adam step, then a direct scaling of theta by 1 - learning\_rate \* lambda), but the net effect is the same.

   * For **bias parameters (b)**, or **BatchNorm gamma, beta**, we omit the lambda \* theta\_t term (effectively lambda = 0 for these). Their update is simply:
     theta\_{t+1} = theta\_t - learning\_rate \* (m\_hat / (sqrt(v\_hat) + epsilon))
     (No direct weight decay applied to b, gamma, beta.) In other words, biases and BN parameters still benefit from Adam’s adaptive step, but we do **not** explicitly decay them. This is because decaying these parameters (especially BN’s gamma, beta) can be detrimental – for example, driving BatchNorm’s gamma towards 0 would nullify the normalization effect.

Each parameter of the network (every weight, bias, gamma, beta) has its own m\_t, v\_t that evolve over time, and the above formulas apply to each of them individually. AdamW takes care to **maintain separate moment estimates for each parameter** while updating them in parallel.

## Weight Decay vs. L2 Regularization (Why Decoupled?)

It’s important to understand how **AdamW’s weight decay differs from traditional L2 regularization** (which was used in plain Adam). In L2 regularization, we add a term (lambda / 2) \* ||theta||^2 to the loss, which results in a gradient term lambda \* theta that *gets added to g\_t*. In standard Adam, this means the weight decay effect is intertwined with the gradient and goes through the m\_t, v\_t moving averages. This coupling has side effects: **the adaptive learning rate can counteract the L2 penalty for some weights.** In fact, weights with large historical gradients end up being regularized *less* effectively under L2 (because Adam’s v\_t term will be large, reducing the relative influence of the lambda \* theta term). This can hinder convergence to an optimal generalization.

**AdamW’s decoupled weight decay** fixes this by applying the penalty *directly to the weights*, outside of the gradient/momentum calculation. This ensures the decay is **consistent for all weights** regardless of their gradient history. In simple terms, *AdamW always multiplies the weight by a constant factor (<1) every step, which uniformly pulls weights toward zero*. By not mixing weight decay into m\_t or v\_t, AdamW **preserves the adaptive learning rate behavior** for the actual gradient, while still gradually shrinking the weights. This leads to more stable and predictable regularization. Empirically, decoupling weight decay in this way often yields better generalization than the old L2-in-Adam approach.

**Why apply weight decay directly?** Because it **keeps the adaptive step and regularization separate**. The Adam part focuses on minimizing the loss, adapting for gradient magnitudes, and **the weight decay part consistently constrains weight growth**. This separation means we don’t have to worry about weight decay messing with the per-parameter learning rates that Adam computes. As a result, AdamW tends to converge more reliably and often generalizes better, which is why it’s become a popular default optimizer in deep learning.

---

**Summary:** Using AdamW, each training step adjusts weights, biases, and batch norm parameters via their gradients and running averages. **Weights** get an extra decay towards 0 (decoupled from the gradient), whereas **biases and batch norm gamma, beta** are typically not decayed. The full AdamW update involves computing momentum (m\_t) and RMS (v\_t) estimates of gradients, bias-correcting them, then using those to scale the gradient for the update. Finally, a weight decay term is applied directly to weights. This decoupled approach to weight decay differs from traditional L2 regularization by avoiding interference with Adam’s adaptive learning rates, thus providing more consistent regularization. The end result is an optimizer that combines the advantages of Adam (fast, adaptive learning) with a cleaner and often more effective form of regularization (AdamW-style weight decay). Each of the parameters—weights, biases, gamma, beta—follows this update rule, with weight decay selectively applied as described, to gradually and intelligently improve the model during training.