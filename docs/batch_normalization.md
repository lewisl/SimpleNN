# Batch Normalization in SimpleNN

## Overview
Batch Normalization (BN) is implemented for both fully connected (LinearLayer) and convolutional (ConvLayer) layers, with careful handling of minibatches and allocation‑free operations. BN integrates into each layer via function pointers:
- Forward: `layer.normalizationf(layer, current_batch_size)`
- Backward: `layer.normalization_gradf(layer, ..., current_batch_size)`
- In prediction (inference), BN uses running statistics and does not update parameters.

## Parameters and State
BN parameters/state are held in `BatchNorm{T}`:
- Learnable parameters: `gam` (scale), `bet` (shift)
- Running statistics (for inference): `mu_run`, `std_run`
- Batch statistics (for training): `mu`, `stddev`
- Gradients and Adam buffers: `grad_gam`, `grad_bet`, `grad_m_*`, `grad_v_*`
- `istraining::Ref{Bool}` is `true` during training and set to `false` for inference (handled by `setup_preds`)
- Numerical epsilon `IT` is added to denominators for stability

A `NoNorm` type is used as a no‑op when batch normalization is disabled.

## Forward Pass: LinearLayer
Inputs: `layer.z` (pre‑activation), `layer.z_norm` (buffer), `current_batch_size cb`.

- Uses views over minibatch when `cb <` allocated batch size:
  - `vzn = view_minibatch(layer.z_norm, 1:cb)`
  - `vz  = view_minibatch(layer.z, 1:cb)`

- Training mode:
  - Compute per‑unit stats across the current batch: `mean!(bn.mu, layer.z)`, `bn.stddev .= std(layer.z, dims=2)`
  - Normalize and re‑scale/shift in place:
    - `vzn = (vz − bn.mu) / (bn.stddev + IT)`
    - `vz  = vzn .* bn.gam .+ bn.bet`
  - Exponential running updates: `mu_run`, `std_run` with 0.95/0.05 smoothing

- Inference mode:
  - `vzn = (vz − bn.mu_run) / (bn.std_run + IT)`
  - `vz  = vzn .* bn.gam .+ bn.bet`

Loops use `@turbo`, and views avoid allocations.

## Forward Pass: ConvLayer
Shapes: `z`, `z_norm` are `(H, W, C, B)`; normalization is per channel.

- Uses views of minibatch when needed; otherwise operates on full arrays
- For each channel slice (over `dims=3`):
  - Compute mean/std over `H×W×B`
  - Normalize and re‑scale/shift per channel:
    - `ch_z_norm = (ch_z − mu[c]) / (std[c] + IT)`
    - `ch_z = ch_z_norm * gam[c] + bet[c]`
- Running stats are updated in training; used in inference

## Backward Pass: LinearLayer
- Uses views over current minibatch for `eps_l` (δL/δz) and `z_norm`
- Parameter gradients averaged over minibatch:
  - `grad_bet = sum(eps_l, dims=2) / mb`
  - `grad_gam = sum(eps_l .* z_norm, dims=2) / mb`
- Gradient to inputs follows BN backward identity:
  - First scale by `gam`: `eps_l .= gam .* eps_l` (this is δL/δẑ)
  - Then per unit: `δz = (1/mb)*(1/σ)*(mb*δẑ − sum(δẑ) − ẑ*sum(δẑ.*ẑ))`
- Implemented in‑place, allocation‑free using reductions along `dims=2`

## Backward Pass: ConvLayer
- Uses views over minibatch for:
  - `layer_above.eps_l` → `veps_l_above`
  - `layer.z_norm` → `vzn`
  - destination `layer.pad_above_eps` → `vpa_eps_l`
- Parameter gradients (per channel), averaged over minibatch:
  - `grad_bet = reshape(sum(veps_l_above, dims=(1,2,4)), C) / mb`
  - `grad_gam = reshape(sum(veps_l_above .* vzn, dims=(1,2,4)), C) / mb`
- For each channel `c`:
  - Scale by `gam[c]` to get δẑ
  - Compute `ch_sum = sum(δẑ)`, `ch_prod_sum = sum(δẑ .* ẑ)`
  - Final δz per element: `(1/mb)/(std[c]+IT) * (mb*δẑ − ch_sum − ẑ*ch_prod_sum)`
- Stores δz in `vpa_eps_l` matching convolution backprop tensor shapes

## Minibatch Handling
Across forward and backward passes:
- `current_batch_size` (`cb`) is propagated from training/prediction
- If layer arrays were allocated for a larger batch than `cb`, code uses `view_minibatch(..., 1:cb)`
- This ensures correctness for unequal minibatches (e.g., last smaller batch) and avoids allocations

## Integration
- Layer constructors assign:
  - `normalizationf = batchnorm!` or `noop`
  - `normalization_gradf = batchnorm_grad!` or `noop`
- Forward pass calls `normalizationf(layer, cb)` after computing `layer.z`
- Backward pass calls `normalization_gradf(...)` after activation gradient and before weight gradients

## Numerical Stability and Performance
- Small epsilon `IT` added to denominators
- Population std (uncorrected) used in conv BN for speed
- Means/stds computed per unit/channel
- Running stats smoothed (0.95/0.05)
- Heavy use of `@turbo`, `eachslice`, and views for speed and zero allocations

## Inference Mode
- `setup_preds` sets `bn.istraining[] = false` so forward BN uses `mu_run/std_run`
- No BN parameter updates or batch stat recomputation in prediction

## Reference Snippet (Linear Forward BN)
```julia
@turbo @. @views vzn = (vz - bn.mu) / (bn.stddev + IT)
@turbo @. @views vz  = vzn * bn.gam + bn.bet
@. bn.mu_run  = ifelse(bn.mu_run[1] == ELT(0.0), bn.mu, 0.95 * bn.mu_run + 0.05 * bn.mu)
@. bn.std_run = ifelse(bn.std_run[1] == ELT(0.0), bn.stddev, 0.95 * bn.std_run + 0.05 * bn.stddev)
```

