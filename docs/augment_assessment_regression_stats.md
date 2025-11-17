# Assessment of `docs/analysis of regression stats.md` and `src/regr_fun.jl`

## Scope
- Evaluate the accuracy and usefulness of the existing analysis doc.
- Cross-check claims against the current implementation in `src/regr_fun.jl`.
- Add concise recommendations based on the actual code.

## Summary judgment
- The doc is largely accurate and helpful. It correctly highlights strengths (clear outputs, practical diagnostics) and sensible improvements (numerical stability, type alignment, effective linear parameters for stacked linear layers).
- A few nuances worth clarifying: the partial/semi-partial definitions used here are "no-refit (zero-coefficient)" versions; classical packages typically refit reduced models. R² edge cases when `SST == 0` can produce `NaN` or `±Inf` and are not explicitly handled in code.

## What the doc gets right
- R², partial R², semi-partial R², and correlation matrix are implemented in a way that matches the intended meanings for a single-output linear model.
- The printed reports are readable and useful for quick interpretation.
- Suggestion to prefer QR/`\` over normal equations for analytical comparisons is appropriate.
- Type/precision consistency with project `ELT` would reduce implicit promotions.
- Suggestion to expose effective coefficients for stacked linear layers (if present) is sound.

## Clarifications and corrections
1) Partial/semi-partial methodology (no-refit)
- In `partial_r_squared` and `semi_partial_r_squared`, the reduced model is simulated by zeroing a single coefficient and recomputing predictions without refitting other coefficients.
- Classical partial/semi-partial (e.g., in stats packages) refit the reduced model after dropping the feature. With multicollinearity, no-refit semi-partials will be smaller and do not sum to the overall R²; this is expected and consistent with the code’s behavior.
- Recommendation: explicitly document the "no-refit" approach in docstrings, and keep the "Shared variance (due to correlation)" line in the report (already present) to guide interpretation.

2) Two-layer linear caveat — relevance
- The doc warns that if you have Linear → Linear with no activation, true coefficients relative to original inputs are `W2*W1` and bias `W2*b1 + b2`.
- In the default regression setup used here, there’s a single linear output layer over raw features; the analysis already aligns with original features.
- Recommendation: keep the caveat, but note it applies only when stacked linear layers exist. Consider adding a small `effective_linear_params(layers)` helper if you plan to support deeper linear stacks.

3) R² edge case behavior
- `calculate_r2` does not guard `SST == 0` (constant target). Current behavior:
  - Perfect predictions → `0/0` → `NaN` (tests accept `NaN`)
  - Imperfect predictions → `1 - (positive/0)` → `-Inf`
- Recommendation: explicitly handle `SST == 0` and return `NaN` with a friendly note in printers.

4) Type/precision alignment
- `cross_correlation_matrix` and `TestLayer` use `Float64`. The broader codebase often uses `ELT=Float32`.
- Recommendation: parametric over `T<:AbstractFloat` or promote inputs to a consistent `T`. Not urgent, but avoids silent up/down conversions.

5) Shape robustness
- Implementation assumes single-output regression (`weights :: 1×n`), with `y` viewable as a vector. This is fine for the current scope.
- Recommendation: assert shapes early and/or add a loop to support multi-output if needed later.

## Additional observations from `src/regr_fun.jl`
- `compare_with_analytical_solution` uses normal equations via `(X_with_bias * X_with_bias') \ (...)`. Prefer `X_with_bias' \ y_vec` (QR) or SVD for better conditioning; a ridge option would be a nice extra.
- Top-of-file demo globals (`x_mod`, `slope`, `b`) look like scratch values; consider moving to tests/fixtures to avoid side effects on load.
- `print_variance_analysis` nicely adds "Shared variance (due to correlation)" = `overall_r2 - sum(semi_partials)` — this is exactly what users need when collinearity is present.
- `cross_correlation_matrix` correctly returns `NaN` when a feature or target has zero variance.
- Minor: `print_correlation_report` is readable for small `n_features`. If you plan to expand, consider optional truncation/sorting.

## Lightweight enhancements (optional)
- VIF per feature (using `(X'X)^{-1}` diagonal scaled appropriately) to flag collinearity.
- Signed semi-partial correlation: `sign(coeff_i) * sqrt(semi_partial_r2_i)` for interpretability.
- Printer tweaks: if `overall_r2` is `NaN`/`±Inf`, show an explanatory note (SST=0).

## Conclusion
- The analysis document is solid and mostly accurate. The main practical nuance is the no-refit definition of partial/semi-partial used here; clarifying that in docs/tooltips will align expectations with stats packages. Future-proof by adding an optional refit-based variant and an `effective_linear_params` helper if deeper linear stacks are introduced.

