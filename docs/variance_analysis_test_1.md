=== Quick Variance Analysis Test ===

Test 1: Simple Linear Relationship
----------------------------------------
True coefficient: 2.0
True bias: 1.0
Learned coefficient: 2.0
Learned bias: 1.0
R²: 0.9973
Variance explained: 101.04%

Test 2: Multiple Features
----------------------------------------
True coefficients: [1.0, 0.5, 3.0]
Feature variances: [0.954, 4.444, 0.238]
Learned coefficients: [1.0, 0.5, 3.0]
R²: 0.991
Variance explained by each feature:
  Feature 1: 22.53%
  Feature 2: 26.15%
  Feature 3: 48.34%

Formatted Analysis:
=== Regression Variance Analysis ===

Overall R²: 0.991
Overall variance explained: 99.1%

Variance explained by each coefficient:
----------------------------------------
Feature_1:
  Coefficient: 1.0
  Variance explained: 22.53%

Feature_2:
  Coefficient: 0.5
  Variance explained: 26.15%

Feature_3:
  Coefficient: 3.0
  Variance explained: 48.34%

Bias (intercept): 0.5

Note: Individual variance contributions sum to 97.02%
(This may differ from overall R² due to feature interactions)

==================================================
Running built-in test suite...
Running variance analysis function tests...
Test 1: Perfect linear relationship
✓ Perfect correlation test passed
Test 2: Multiple features with known variance contributions
✓ Multiple features variance attribution test passed
Test 3: R² calculation edge cases
✓ Constant values R² test passed
All variance analysis tests passed! ✓

✓ All variance analysis functions working correctly!

Quick variance analysis test completed!