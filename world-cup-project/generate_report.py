"""
generate_report.py
Produces a self-contained report.html from the project data and results.
Run:  python3 generate_report.py
"""

import csv, base64, datetime, pathlib, numpy as np

ROOT = pathlib.Path(__file__).parent

# ---------------------------------------------------------------------------
# Load & recompute (mirrors analysis.py — keeps this script self-contained)
# ---------------------------------------------------------------------------

rows = list(csv.DictReader(open(ROOT / "world_cup_2022_data.csv")))

teams         = [r["team"].strip()             for r in rows]
pct_top5      = [float(r["pct_in_top5"])       for r in rows]
fifa_ranking  = [float(r["fifa_ranking"])      for r in rows]
stage_reached = [float(r["stage_reached"])     for r in rows]
goal_diff     = [float(r["goal_differential"]) for r in rows]
squad_size    = [int(r["squad_size"])           for r in rows]
num_top5      = [int(r["num_in_top5"])          for r in rows]

n = len(teams)
p, r, s, g = map(np.array, [pct_top5, fifa_ranking, stage_reached, goal_diff])

# Correlation matrix
M = np.column_stack([p, r, s, g])
Mc = M - M.mean(axis=0)
cov = (Mc.T @ Mc) / (n - 1)
std = np.sqrt(np.diag(cov))
corr = cov / np.outer(std, std)
var_names = ["pct_in_top5", "fifa_ranking", "stage_reached", "goal_diff"]

# Least squares  (normal equations)
ones = np.ones(n)
X = np.column_stack([ones, p, r])
y = s
beta = np.linalg.solve(X.T @ X, X.T @ y)
y_hat = X @ beta
residuals = y - y_hat
ss_res = residuals @ residuals
ss_tot = (y - y.mean()) @ (y - y.mean())
r_sq = 1 - ss_res / ss_tot

# Encode scatter plot
img_b64 = base64.b64encode((ROOT / "results/scatter_top5_vs_stage.png").read_bytes()).decode()

# Stage labels
STAGE = {1:"Group Stage", 2:"Round of 16", 3:"Quarterfinal",
         4:"Semifinal", 5:"Finalist", 6:"Champion"}

# ---------------------------------------------------------------------------
# Helper: correlation cell color  (blue → white → red)
# ---------------------------------------------------------------------------
def corr_color(v):
    v = float(v)
    if abs(v) >= 0.999:              # diagonal
        return "#cccccc", "#333"
    if v >= 0:
        r2, g2, b2 = 255, int(255*(1-v)), int(255*(1-v))
    else:
        r2, g2, b2 = int(255*(1+v)), int(255*(1+v)), 255
    fg = "white" if abs(v) > 0.65 else "#222"
    return f"rgb({r2},{g2},{b2})", fg

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: system-ui, -apple-system, sans-serif;
  background: #f0f2f5;
  color: #222;
  line-height: 1.6;
}
a { color: #1a3a5c; }

/* ---- NAV ---- */
nav {
  position: sticky; top: 0; z-index: 100;
  background: #1a3a5c; padding: 0 2rem;
  display: flex; align-items: center; gap: 1.5rem;
  box-shadow: 0 2px 8px rgba(0,0,0,.3);
}
nav a {
  color: #cdd8e8; text-decoration: none;
  padding: .8rem 0; font-size: .88rem; letter-spacing: .03em;
  border-bottom: 2px solid transparent; transition: .2s;
}
nav a:hover { color: #fff; border-bottom-color: #4da3ff; }
nav .brand { color: #fff; font-weight: 700; font-size: 1rem; margin-right: auto; }

/* ---- LAYOUT ---- */
.page { max-width: 960px; margin: 2rem auto; padding: 0 1rem 4rem; }
.card {
  background: #fff; border-radius: 8px;
  box-shadow: 0 1px 6px rgba(0,0,0,.1);
  padding: 2rem 2.2rem; margin-bottom: 2rem;
}
h1 { font-size: 1.7rem; color: #1a3a5c; line-height: 1.3; margin-bottom: .4rem; }
h2 {
  font-size: 1.2rem; color: #1a3a5c; margin-bottom: 1.2rem;
  padding-bottom: .4rem; border-bottom: 3px solid #1a3a5c;
}
h3 { font-size: 1rem; color: #444; margin: 1.2rem 0 .5rem; }
p  { margin-bottom: .75rem; }
ul { margin: .5rem 0 .75rem 1.2rem; }
li { margin-bottom: .3rem; }

/* ---- HERO ---- */
.hero { background: linear-gradient(135deg, #1a3a5c 0%, #2a6099 100%);
  color: #fff; padding: 2.5rem 2.5rem; border-radius: 8px; margin-bottom: 2rem; }
.hero h1 { color: #fff; font-size: 1.8rem; }
.hero .sub { color: #a8c8e8; margin: .3rem 0 1.2rem; font-size: .95rem; }
.rq { background: rgba(255,255,255,.1); border-radius: 6px;
  padding: .9rem 1.2rem; margin-top: .8rem; }
.rq p { margin: .3rem 0; font-size: .92rem; }

/* ---- TABLES ---- */
.tbl-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: .85rem; }
th {
  background: #1a3a5c; color: #fff; text-align: center;
  padding: .55rem .6rem; cursor: pointer; user-select: none;
  white-space: nowrap;
}
th:hover { background: #265480; }
td { padding: .45rem .6rem; border-bottom: 1px solid #eee; text-align: center; }
td:first-child { text-align: left; font-weight: 500; }
tr:hover td { background: #f0f6ff; }
.badge {
  display: inline-block; padding: .15rem .55rem;
  border-radius: 999px; font-size: .78rem; font-weight: 600; color: #fff;
}

/* ---- CORR MATRIX ---- */
.corr-table th, .corr-table td {
  padding: .55rem .7rem; font-size: .83rem;
  border: 1px solid #ddd; white-space: nowrap;
}
.corr-table th { background: #1a3a5c; color: #fff; cursor: default; }
.corr-table td:first-child { font-weight: 600; background: #f5f7fa; color: #333; }

/* ---- MATH ---- */
.math-box {
  background: #f8f9fb; border-left: 4px solid #1a3a5c;
  border-radius: 4px; padding: 1rem 1.4rem; margin: .8rem 0 1.2rem;
  overflow-x: auto;
}
.step { margin-bottom: 1.4rem; }
.step-num {
  display: inline-block; width: 1.6rem; height: 1.6rem;
  background: #1a3a5c; color: #fff; border-radius: 50%;
  text-align: center; line-height: 1.6rem; font-size: .8rem;
  font-weight: 700; margin-right: .5rem;
}

/* ---- SCATTER ---- */
.scatter img { max-width: 100%; border: 1px solid #ddd; border-radius: 6px; }

/* ---- REGRESSION EQ ---- */
.eq-card {
  background: #eef4fb; border: 1px solid #b5cceb;
  border-radius: 6px; padding: 1.1rem 1.4rem; margin: 1rem 0;
  text-align: center; font-size: 1rem;
}
.stat-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.stat { flex: 1; min-width: 140px; background: #f5f7fa;
  border-radius: 6px; padding: .8rem 1rem; text-align: center; }
.stat .val { font-size: 1.5rem; font-weight: 700; color: #1a3a5c; }
.stat .lbl { font-size: .78rem; color: #666; }

/* ---- FOOTER ---- */
footer { text-align: center; color: #aaa; font-size: .8rem; margin-top: 3rem; }
"""

# ---------------------------------------------------------------------------
# NAV
# ---------------------------------------------------------------------------
NAV = """
<nav>
  <span class="brand">&#x26BD; 2022 WC Linear Algebra</span>
  <a href="#data">Data</a>
  <a href="#methods">Methods</a>
  <a href="#results">Results</a>
  <a href="#conclusion">Conclusion</a>
</nav>
"""

# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
HERO = """
<div class="hero">
  <h1>Does Playing in the Top&#8209;5 European Leagues Predict 2022 World Cup Success?</h1>
  <p class="sub">Math 2220 &middot; Linear Algebra Application Project</p>
  <div class="rq">
    <p><strong>RQ 1 &mdash;</strong> Is a higher percentage of players in the top&#8209;5 European leagues
       associated with greater success in the 2022 FIFA World Cup?</p>
    <p><strong>RQ 2 &mdash;</strong> Can stage reached be approximated with a least&#8209;squares model
       using percent in top&#8209;5 leagues and FIFA ranking as predictors?</p>
  </div>
</div>
"""

# ---------------------------------------------------------------------------
# DATA TABLE
# ---------------------------------------------------------------------------
def badge_stage(s):
    colors = {1:"#888", 2:"#3a7bd5", 3:"#9b59b6",
              4:"#e67e22", 5:"#c0392b", 6:"#27ae60"}
    bg = colors.get(int(s), "#888")
    return f'<span class="badge" style="background:{bg};">{STAGE[int(s)]}</span>'

def pct_bar(v):
    pct = float(v)*100
    return (f'<div style="display:flex;align-items:center;gap:.4rem;">'
            f'<div style="width:70px;background:#e0e8f4;border-radius:3px;height:8px;">'
            f'<div style="width:{pct:.1f}%;background:#1a3a5c;height:8px;border-radius:3px;"></div>'
            f'</div><span>{pct:.1f}%</span></div>')

header_cols = [
    ("Team", "team", False),
    ("Squad", "squad", True),
    ("# Top-5", "num5", True),
    ("% Top-5", "pct5", True),
    ("FIFA Rank", "rank", True),
    ("Stage", "stage", True),
    ("GS Pts", "pts", True),
    ("Goal Diff", "gdiff", True),
]

th_row = "".join(
    f'<th onclick="sortTable({i}, {str(num).lower()})">{label} &#x25BF;</th>'
    for i, (label, _, num) in enumerate(header_cols)
)

td_rows = []
for i, row in enumerate(rows):
    s_val = int(float(row["stage_reached"]))
    pct   = float(row["pct_in_top5"])
    gd    = int(float(row["goal_differential"]))
    gd_str = f'+{gd}' if gd > 0 else str(gd)
    gd_color = "#27ae60" if gd > 0 else ("#c0392b" if gd < 0 else "#666")
    td_rows.append(
        f'<tr>'
        f'<td data-val="{row["team"].strip()}">{row["team"].strip()}</td>'
        f'<td data-val="{row["squad_size"]}">{row["squad_size"]}</td>'
        f'<td data-val="{row["num_in_top5"]}">{row["num_in_top5"]}</td>'
        f'<td data-val="{pct}">{pct_bar(pct)}</td>'
        f'<td data-val="{row["fifa_ranking"]}">{int(float(row["fifa_ranking"]))}</td>'
        f'<td data-val="{s_val}">{badge_stage(s_val)}</td>'
        f'<td data-val="{row["group_stage_points"]}">{row["group_stage_points"]}</td>'
        f'<td data-val="{gd}" style="color:{gd_color};font-weight:600">{gd_str}</td>'
        f'</tr>'
    )

SORT_JS = """
<script>
function sortTable(col, numeric) {
  const tbody = document.querySelector('#data-table tbody');
  const rows  = Array.from(tbody.rows);
  const asc   = !(tbody.dataset.sortCol == col && tbody.dataset.sortDir === 'asc');
  rows.sort((a, b) => {
    const av = a.cells[col].dataset.val;
    const bv = b.cells[col].dataset.val;
    if (numeric) return asc ? parseFloat(av) - parseFloat(bv) : parseFloat(bv) - parseFloat(av);
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  });
  rows.forEach(r => tbody.appendChild(r));
  tbody.dataset.sortCol = col;
  tbody.dataset.sortDir = asc ? 'asc' : 'desc';
}
</script>
"""

DATA_SECTION = f"""
<div id="data" class="card">
  <h2>Dataset &mdash; 32 Teams</h2>
  <p>Each row is one national team. The top&#8209;5 leagues are the
     <strong>Premier League, La Liga, Serie A, Bundesliga,</strong> and <strong>Ligue 1</strong>.
     Club assignments are from the official FIFA squad lists (November 2022).
     Click any column header to sort.</p>
  <div class="tbl-wrap">
    <table id="data-table" data-sort-col="" data-sort-dir="">
      <thead><tr>{th_row}</tr></thead>
      <tbody>{"".join(td_rows)}</tbody>
    </table>
  </div>
</div>
{SORT_JS}
"""

# ---------------------------------------------------------------------------
# METHODS
# ---------------------------------------------------------------------------
METHODS_SECTION = f"""
<div id="methods" class="card">
  <h2>Methods &mdash; Linear Algebra Tools</h2>

  <div class="step">
    <h3><span class="step-num">1</span>Variable Definitions</h3>
    <p>For each team we define the <em>predictor</em> as the fraction of squad players
       who played in one of the five major European leagues:</p>
    <div class="math-box">
      $$\\text{{pct\\_in\\_top5}}_i = \\frac{{\\text{{num\\_in\\_top5}}_i}}{{\\text{{squad\\_size}}_i}}$$
    </div>
    <p>The <em>response variable</em> is a numeric success score based on stage reached:</p>
    <ul>
      <li>1 = Group Stage &nbsp;&nbsp; 2 = Round of 16 &nbsp;&nbsp; 3 = Quarterfinal</li>
      <li>4 = Semifinal &nbsp;&nbsp; 5 = Finalist &nbsp;&nbsp; 6 = Champion</li>
    </ul>
  </div>

  <div class="step">
    <h3><span class="step-num">2</span>Correlation Matrix</h3>
    <p>We stack four variables into an $n \\times 4$ data matrix $M$, then <strong>mean-center</strong>
       each column to get $\\tilde M$:</p>
    <div class="math-box">
      $$\\tilde{{M}} = M - \\mathbf{{1}}\\bar{{\\mathbf{{x}}}}^\\top
        \\qquad (\\bar{{\\mathbf{{x}}}} \\text{{ is the column-mean vector}})$$
    </div>
    <p>The <strong>sample covariance matrix</strong> is then computed via a single matrix product:</p>
    <div class="math-box">
      $$C = \\frac{{1}}{{n-1}}\\,\\tilde{{M}}^\\top \\tilde{{M}}$$
    </div>
    <p>Dividing by the outer product of standard deviations gives the
       <strong>correlation matrix</strong>, whose entries lie in $[-1, 1]$:</p>
    <div class="math-box">
      $$r_{{ij}} = \\frac{{C_{{ij}}}}{{\\sigma_i\\,\\sigma_j}}
        \\qquad \\sigma_i = \\sqrt{{C_{{ii}}}}$$
    </div>
    <p>With $n = {n}$ teams and four variables, $C$ is a $4\\times 4$ symmetric matrix.
       The diagonal entries are all $1$ by construction.</p>
  </div>

  <div class="step">
    <h3><span class="step-num">3</span>Least-Squares via Normal Equations</h3>
    <p>We seek the coefficient vector $\\boldsymbol{{\\beta}}$ that minimises the sum of squared residuals
       between the model $\\hat{{y}} = X\\boldsymbol{{\\beta}}$ and the observed stages $y$:</p>
    <div class="math-box">
      $$\\min_{{\\boldsymbol{{\\beta}}}}\\;\\|\\,y - X\\boldsymbol{{\\beta}}\\|^2
        \\qquad X = \\begin{{bmatrix}}1 & p_1 & r_1 \\\\ \\vdots & \\vdots & \\vdots \\\\ 1 & p_n & r_n\\end{{bmatrix}}
        \\in \\mathbb{{R}}^{{{n}\\times 3}}$$
    </div>
    <p>Setting the gradient to zero yields the <strong>normal equations</strong>, which we solve exactly:</p>
    <div class="math-box">
      $$X^\\top X\\,\\boldsymbol{{\\beta}} = X^\\top y
        \\quad\\Longrightarrow\\quad
        \\boldsymbol{{\\beta}} = (X^\\top X)^{{-1}}X^\\top y$$
    </div>
    <p>Here $p_i$ = pct&nbsp;in&nbsp;top&#8209;5 and $r_i$ = FIFA ranking for team $i$.
       Because $X^\\top X$ is a $3\\times 3$ positive-definite matrix, the solution is unique.</p>
  </div>
</div>
"""

# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------
# --- Correlation matrix table
corr_col_labels = ["pct_in_top5", "fifa_ranking", "stage_reached", "goal_diff"]
corr_th = "<tr><th></th>" + "".join(f"<th>{h}</th>" for h in corr_col_labels) + "</tr>"
corr_tds = []
for i, rname in enumerate(corr_col_labels):
    cells = ""
    for j in range(4):
        bg, fg = corr_color(corr[i, j])
        cells += f'<td style="background:{bg};color:{fg};">{corr[i,j]:.4f}</td>'
    corr_tds.append(f"<tr><td>{rname}</td>{cells}</tr>")

# --- Residuals table
res_rows = []
for i in range(n):
    gd = residuals[i]
    color = "#27ae60" if gd > 0.1 else ("#c0392b" if gd < -0.1 else "#666")
    res_rows.append(
        f'<tr><td>{teams[i]}</td>'
        f'<td>{STAGE[int(s[i])]}</td>'
        f'<td>{y_hat[i]:.2f}</td>'
        f'<td style="color:{color};font-weight:600">{gd:+.2f}</td></tr>'
    )

RESULTS_SECTION = f"""
<div id="results" class="card">
  <h2>Results</h2>

  <h3>Correlation Matrix</h3>
  <p>Color scale: <span style="background:#ff6666;padding:1px 8px;border-radius:3px;">red = positive</span>
     &nbsp; <span style="background:#6699ff;color:#fff;padding:1px 8px;border-radius:3px;">blue = negative</span>
     &nbsp; white ≈ 0. Diagonal suppressed (always 1).</p>
  <div class="tbl-wrap">
    <table class="corr-table">
      <thead>{corr_th}</thead>
      <tbody>{"".join(corr_tds)}</tbody>
    </table>
  </div>

  <h3 style="margin-top:1.8rem;">Scatter Plot &mdash; % Top-5 vs Stage Reached</h3>
  <div class="scatter">
    <img src="data:image/png;base64,{img_b64}" alt="Scatter: pct_top5 vs stage_reached">
  </div>

  <h3 style="margin-top:1.8rem;">Least-Squares Regression Model</h3>
  <p>Using two predictors (pct_in_top5 and FIFA ranking), the normal equations give:</p>
  <div class="eq-card">
    $$\\hat{{y}} = {beta[0]:.4f} + {beta[1]:.4f}\\,x_1 + ({beta[2]:.4f})\\,x_2$$
    <p style="font-size:.85rem;margin-top:.5rem;color:#555;">
      $x_1$ = pct_in_top5 &nbsp;|&nbsp; $x_2$ = FIFA ranking &nbsp;|&nbsp; $\\hat{{y}}$ = predicted stage
    </p>
  </div>

  <div class="stat-row">
    <div class="stat">
      <div class="val">{corr[0,2]:.3f}</div>
      <div class="lbl">r(pct_top5, stage)</div>
    </div>
    <div class="stat">
      <div class="val">{corr[1,2]:.3f}</div>
      <div class="lbl">r(fifa_rank, stage)</div>
    </div>
    <div class="stat">
      <div class="val">{r_sq:.3f}</div>
      <div class="lbl">R&#178; (2-predictor model)</div>
    </div>
    <div class="stat">
      <div class="val">{n}</div>
      <div class="lbl">Teams (sample size)</div>
    </div>
  </div>

  <h3 style="margin-top:1rem;">Residuals by Team</h3>
  <p>Positive residual = team outperformed the model; negative = underperformed.</p>
  <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th style="cursor:default;">Team</th>
        <th style="cursor:default;">Actual Stage</th>
        <th style="cursor:default;">Predicted</th>
        <th style="cursor:default;">Residual</th>
      </tr></thead>
      <tbody>{"".join(res_rows)}</tbody>
    </table>
  </div>
</div>
"""

# ---------------------------------------------------------------------------
# CONCLUSION
# ---------------------------------------------------------------------------
CONCLUSION_SECTION = f"""
<div id="conclusion" class="card">
  <h2>Conclusion &amp; Limitations</h2>

  <h3>What the model found</h3>
  <p>There is a <strong>moderate positive correlation</strong> between a team's top&#8209;5 league
     share and its World Cup stage (r&nbsp;=&nbsp;{corr[0,2]:.3f}). FIFA ranking is a slightly
     stronger individual predictor (r&nbsp;=&nbsp;{corr[1,2]:.3f}, negative because a lower
     ranking number means a stronger team). Together the two predictors explain
     <strong>{r_sq*100:.1f}% of the variance</strong> in stage reached (R²&nbsp;=&nbsp;{r_sq:.3f}).</p>

  <p>The regression coefficient on pct_in_top5 is <strong>{beta[1]:.3f}</strong>, meaning a team
     with 100% of players in top&#8209;5 leagues is predicted to reach roughly
     {beta[1]:.2f} stages further than a team with 0%—holding FIFA ranking constant.
     Each one&#8209;place drop in FIFA ranking reduces the predicted stage by {abs(beta[2]):.3f}.</p>

  <h3>Limitations</h3>
  <ul>
    <li><strong>Small sample:</strong> Only 32 data points makes estimates noisy. A few upset
        results (e.g., Saudi Arabia beating Argentina in group stage) can move the regression line
        noticeably.</li>
    <li><strong>Coarse response variable:</strong> <em>Stage reached</em> is ordinal, not
        continuous—jumping from Round&#8209;of&#8209;16 to Quarterfinal is not the same magnitude
        as Champion to Finalist, yet the model treats them equally.</li>
    <li><strong>Collinearity:</strong> pct_in_top5 and FIFA ranking are themselves correlated
        (r&nbsp;=&nbsp;{corr[0,1]:.3f}), so their individual coefficients are harder to interpret
        cleanly.</li>
    <li><strong>Omitted variables:</strong> Injuries, draw luck, home&#8209;continent advantage
        (Morocco, Qatar), and tactical coaching all affect outcomes but are not captured.</li>
    <li><strong>Top&#8209;5 definition:</strong> Playing in a top&#8209;5 league does not
        guarantee playing time. A bench player at PSG contributes less than a starter at a
        mid&#8209;table club.</li>
  </ul>

  <h3>Takeaway</h3>
  <p>The linear algebra tools—<strong>correlation matrices</strong> and <strong>least&#8209;squares
     via normal equations</strong>—give a clean quantitative picture: top&#8209;5 league experience
     is meaningfully associated with World Cup success, but it explains only about a third of the
     variation. Tournament football remains genuinely unpredictable, and that is precisely what makes
     it worth watching.</p>
</div>
"""

# ---------------------------------------------------------------------------
# ASSEMBLE & WRITE
# ---------------------------------------------------------------------------
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>2022 FIFA WC — Linear Algebra Analysis</title>
  <style>{CSS}</style>
  <script>MathJax = {{ tex: {{ inlineMath: [['$','$']] }} }};</script>
  <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head>
<body>
{NAV}
<div class="page">
  {HERO}
  {DATA_SECTION}
  {METHODS_SECTION}
  {RESULTS_SECTION}
  {CONCLUSION_SECTION}
  <footer><p>Generated {datetime.date.today()} &middot; Math 2220 Linear Algebra</p></footer>
</div>
</body>
</html>"""

out = ROOT / "report.html"
out.write_text(HTML, encoding="utf-8")
print(f"report.html written ({out.stat().st_size // 1024} KB)")
print(f"Open with:  open {out}")
