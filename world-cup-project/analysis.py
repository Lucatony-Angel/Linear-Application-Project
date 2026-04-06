"""
Analysis: Top-5-League Representation and 2022 World Cup Success
Math 2220 Linear Algebra Project

Tools used (pure linear algebra, no scikit-learn):
  - Correlation matrix via mean-centering and matrix multiplication
  - Least squares via the normal equations: beta = (X^T X)^{-1} X^T y
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

data_path = os.path.join(os.path.dirname(__file__), "world_cup_2022_data.csv")

teams          = []
pct_top5       = []
fifa_ranking   = []
stage_reached  = []
goal_diff      = []

with open(data_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        teams.append(row["team"].strip())
        pct_top5.append(float(row["pct_in_top5"]))
        fifa_ranking.append(float(row["fifa_ranking"]))
        stage_reached.append(float(row["stage_reached"]))
        goal_diff.append(float(row["goal_differential"]))

n = len(teams)

# Convert to numpy arrays
p = np.array(pct_top5)        # percent in top 5 leagues
r = np.array(fifa_ranking)    # FIFA ranking (lower = better)
s = np.array(stage_reached)   # success score 1-6
g = np.array(goal_diff)       # goal differential

# ---------------------------------------------------------------------------
# 2. Correlation matrix
# ---------------------------------------------------------------------------
# Variables: pct_in_top5, fifa_ranking, stage_reached, goal_differential
# Note: fifa_ranking is inverted scale (lower rank = stronger team),
#       so its correlation with success will be negative.

var_names = ["pct_in_top5", "fifa_ranking", "stage_reached", "goal_diff"]
M = np.column_stack([p, r, s, g])   # shape (32, 4)

# Mean-center each column
M_centered = M - M.mean(axis=0)

# Covariance matrix (sample, divide by n-1)
cov = (M_centered.T @ M_centered) / (n - 1)

# Standard deviations
std = np.sqrt(np.diag(cov))

# Correlation matrix: C_ij = cov_ij / (std_i * std_j)
corr = cov / np.outer(std, std)

# ---------------------------------------------------------------------------
# 3. Least-squares model
#    y = stage_reached
#    X = [1 | pct_in_top5 | fifa_ranking]   (32 x 3)
#    beta = (X^T X)^{-1} X^T y
# ---------------------------------------------------------------------------

ones = np.ones(n)
X = np.column_stack([ones, p, r])   # design matrix
y = s                                # response vector

XtX = X.T @ X          # (3 x 3)
Xty = X.T @ y          # (3,)
beta = np.linalg.solve(XtX, Xty)   # solves normal equations exactly

# Fitted values and residuals
y_hat = X @ beta
residuals = y - y_hat

# R-squared (how much variance is explained)
ss_res = residuals @ residuals
ss_tot = (y - y.mean()) @ (y - y.mean())
r_squared = 1 - ss_res / ss_tot

# ---------------------------------------------------------------------------
# 4. Write correlation_matrix.csv
# ---------------------------------------------------------------------------

corr_path = os.path.join(os.path.dirname(__file__), "results", "correlation_matrix.csv")

with open(corr_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([""] + var_names)
    for i, name in enumerate(var_names):
        writer.writerow([name] + [f"{corr[i, j]:.4f}" for j in range(4)])

# ---------------------------------------------------------------------------
# 5. Write regression_output.txt
# ---------------------------------------------------------------------------

reg_path = os.path.join(os.path.dirname(__file__), "results", "regression_output.txt")

with open(reg_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("LEAST-SQUARES REGRESSION  (Normal Equations)\n")
    f.write("=" * 60 + "\n\n")
    f.write("Model:  stage_reached = b0 + b1*(pct_in_top5) + b2*(fifa_ranking)\n\n")
    f.write(f"  b0 (intercept)    = {beta[0]:.4f}\n")
    f.write(f"  b1 (pct_in_top5)  = {beta[1]:.4f}\n")
    f.write(f"  b2 (fifa_ranking) = {beta[2]:.4f}\n\n")
    f.write(f"  R-squared         = {r_squared:.4f}\n\n")
    f.write("-" * 60 + "\n")
    f.write("Interpretation:\n")
    f.write(f"  A 10pp increase in top-5 share shifts predicted stage by {beta[1]*0.10:.4f}.\n")
    f.write(f"  Each 1-place drop in FIFA ranking shifts predicted stage by {beta[2]:.4f}.\n")
    f.write(f"  The model explains {r_squared*100:.1f}% of variance in stage reached.\n\n")
    f.write("-" * 60 + "\n")
    f.write("Correlation highlights (from correlation matrix):\n")
    f.write(f"  pct_in_top5  vs stage_reached : {corr[0,2]:.4f}\n")
    f.write(f"  fifa_ranking vs stage_reached : {corr[1,2]:.4f}  (negative = lower rank -> better)\n")
    f.write(f"  pct_in_top5  vs goal_diff     : {corr[0,3]:.4f}\n")
    f.write(f"  pct_in_top5  vs fifa_ranking  : {corr[0,1]:.4f}\n\n")
    f.write("-" * 60 + "\n")
    f.write("Residuals by team (actual - predicted):\n")
    f.write(f"  {'Team':<20} {'Actual':>8} {'Predicted':>10} {'Residual':>10}\n")
    f.write(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10}\n")
    for i in range(n):
        f.write(f"  {teams[i]:<20} {int(s[i]):>8} {y_hat[i]:>10.2f} {residuals[i]:>10.2f}\n")

# ---------------------------------------------------------------------------
# 6. Scatter plot: pct_in_top5 vs stage_reached  (with regression line)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

# Scatter points
ax.scatter(p, s, color="steelblue", zorder=3, s=60)

# Label each point
for i in range(n):
    ax.annotate(teams[i], (p[i], s[i]), textcoords="offset points",
                xytext=(5, 3), fontsize=6.5, color="dimgray")

# Simple univariate regression line for the plot (pct_in_top5 only)
X1 = np.column_stack([ones, p])
beta1 = np.linalg.solve(X1.T @ X1, X1.T @ y)
x_line = np.linspace(p.min() - 0.02, p.max() + 0.02, 200)
y_line = beta1[0] + beta1[1] * x_line
ax.plot(x_line, y_line, color="firebrick", linewidth=1.5,
        label=f"Least-squares fit  (slope = {beta1[1]:.2f})")

# Stage labels on y-axis
stage_labels = {1: "Group Stage", 2: "Round of 16", 3: "Quarterfinal",
                4: "Semifinal", 5: "Finalist", 6: "Champion"}
ax.set_yticks(list(stage_labels.keys()))
ax.set_yticklabels([stage_labels[k] for k in stage_labels])

ax.set_xlabel("% of Squad in Top-5 European Leagues", fontsize=12)
ax.set_ylabel("World Cup Stage Reached", fontsize=12)
ax.set_title("Top-5 League Representation vs 2022 World Cup Success", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plot_path = os.path.join(os.path.dirname(__file__), "results", "scatter_top5_vs_stage.png")
plt.savefig(plot_path, dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 7. Print summary to terminal
# ---------------------------------------------------------------------------

print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"\nSample size: {n} teams\n")
print("Correlation matrix saved to results/correlation_matrix.csv")
print(f"\nKey correlations:")
print(f"  pct_in_top5 vs stage_reached : {corr[0,2]:.4f}")
print(f"  fifa_ranking vs stage_reached : {corr[1,2]:.4f}")
print(f"\nLeast-squares model (2 predictors):")
print(f"  stage = {beta[0]:.3f} + {beta[1]:.3f}*pct_in_top5 + {beta[2]:.3f}*fifa_ranking")
print(f"  R-squared = {r_squared:.4f}")
print(f"\nOutputs written to results/")
print("  - correlation_matrix.csv")
print("  - regression_output.txt")
print("  - scatter_top5_vs_stage.png")
