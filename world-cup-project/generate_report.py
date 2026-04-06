"""
generate_report.py
Outputs a multi-page site to docs/ — ready for GitHub Pages.
Run:  python3 generate_report.py
Then push the docs/ folder to GitHub and enable Pages (branch: main, folder: /docs).
"""

import csv, base64, datetime, pathlib, shutil, numpy as np
from collections import defaultdict

ROOT = pathlib.Path(__file__).parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data + linear algebra (self-contained)
# ---------------------------------------------------------------------------
rows          = list(csv.DictReader(open(ROOT / "world_cup_2022_data.csv")))
teams         = [r["team"].strip()             for r in rows]
pct_top5      = [float(r["pct_in_top5"])       for r in rows]
fifa_ranking  = [float(r["fifa_ranking"])      for r in rows]
stage_reached = [float(r["stage_reached"])     for r in rows]
goal_diff     = [float(r["goal_differential"]) for r in rows]

n = len(teams)
p, rk, s, g = map(np.array, [pct_top5, fifa_ranking, stage_reached, goal_diff])

M  = np.column_stack([p, rk, s, g])
Mc = M - M.mean(axis=0)
cov  = (Mc.T @ Mc) / (n - 1)
std  = np.sqrt(np.diag(cov))
corr = cov / np.outer(std, std)

ones  = np.ones(n)
X     = np.column_stack([ones, p, rk])
y     = s
beta  = np.linalg.solve(X.T @ X, X.T @ y)
y_hat = X @ beta
resid = y - y_hat
r_sq  = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))

# Copy scatter plot into docs/
shutil.copy(ROOT / "results/scatter_top5_vs_stage.png", DOCS / "scatter.png")

STAGE = {1:"Group Stage", 2:"Round of 16", 3:"Quarterfinal",
         4:"Semifinal", 5:"Finalist", 6:"Champion"}
VAR_NAMES = ["pct_in_top5", "fifa_ranking", "stage_reached", "goal_diff"]

# ---------------------------------------------------------------------------
# Shared CSS  (one string, injected into every page)
# ---------------------------------------------------------------------------
CSS = """
:root {
  --navy: #1a3a5c;
  --navy-light: #2a6099;
  --bg: #f0f2f5;
  --card: #fff;
  --accent: #4da3ff;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
  font-family: system-ui, -apple-system, sans-serif;
  background: var(--bg); color: #222; line-height: 1.65;
  min-height: 100vh; display: flex; flex-direction: column;
}
a { color: var(--navy); }

/* NAV */
nav {
  position: sticky; top: 0; z-index: 200;
  background: var(--navy); padding: 0 2rem;
  display: flex; align-items: center; gap: 0;
  box-shadow: 0 2px 10px rgba(0,0,0,.35);
}
.brand {
  color: #fff; font-weight: 700; font-size: .95rem;
  margin-right: auto; padding: .9rem 0; text-decoration: none;
  letter-spacing: .02em;
}
nav a {
  color: #b0c8e0; text-decoration: none;
  padding: .9rem 1.1rem; font-size: .85rem; letter-spacing: .03em;
  border-bottom: 3px solid transparent; transition: .18s;
  white-space: nowrap;
}
nav a:hover          { color: #fff; border-bottom-color: var(--accent); }
nav a.active         { color: #fff; border-bottom-color: var(--accent); font-weight: 600; }

/* LAYOUT */
.page { max-width: 980px; margin: 2.5rem auto; padding: 0 1.2rem 5rem; flex: 1; }
.card {
  background: var(--card); border-radius: 10px;
  box-shadow: 0 1px 8px rgba(0,0,0,.09);
  padding: 2.2rem 2.4rem; margin-bottom: 2rem;
}

/* TYPOGRAPHY */
h1 { font-size: 1.75rem; color: var(--navy); line-height: 1.3; margin-bottom: .5rem; }
h2 { font-size: 1.18rem; color: var(--navy); margin-bottom: 1.2rem;
     padding-bottom: .45rem; border-bottom: 3px solid var(--navy); }
h3 { font-size: 1rem; color: #444; margin: 1.3rem 0 .5rem; }
p  { margin-bottom: .8rem; }
ul { margin: .5rem 0 .8rem 1.3rem; }
li { margin-bottom: .35rem; }

/* HERO */
.hero {
  background: linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%);
  color: #fff; padding: 3rem 2.8rem; border-radius: 10px; margin-bottom: 2rem;
}
.hero h1 { color: #fff; font-size: 2rem; margin-bottom: .5rem; }
.hero .sub { color: #a8c8e8; font-size: .95rem; margin-bottom: 1.4rem; }
.rq { background: rgba(255,255,255,.12); border-radius: 8px; padding: 1rem 1.4rem; }
.rq p { margin: .4rem 0; font-size: .92rem; }
.hero-links { display: flex; gap: .8rem; flex-wrap: wrap; margin-top: 1.8rem; }
.btn {
  display: inline-block; padding: .6rem 1.4rem; border-radius: 6px;
  font-size: .88rem; font-weight: 600; text-decoration: none; transition: .18s;
}
.btn-white  { background: #fff; color: var(--navy); }
.btn-white:hover  { background: #e8f0ff; }
.btn-ghost  { background: rgba(255,255,255,.15); color: #fff; border: 1px solid rgba(255,255,255,.4); }
.btn-ghost:hover  { background: rgba(255,255,255,.25); }

/* STAT CARDS (index page) */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); gap: 1rem; margin: 1.5rem 0; }
.stat-card { background: #f4f7fb; border-radius: 8px; padding: 1.1rem 1rem; text-align: center; }
.stat-card .val { font-size: 1.9rem; font-weight: 700; color: var(--navy); }
.stat-card .lbl { font-size: .76rem; color: #666; margin-top: .2rem; }

/* PAGE CARD GRID (index nav cards) */
.page-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap: 1rem; margin-top: 1.2rem; }
.page-card {
  border: 1.5px solid #dde6f0; border-radius: 8px; padding: 1.2rem;
  text-decoration: none; color: #333; transition: .18s;
  display: block;
}
.page-card:hover { border-color: var(--navy); box-shadow: 0 2px 12px rgba(26,58,92,.12); transform: translateY(-1px); }
.page-card h3 { color: var(--navy); margin: 0 0 .3rem; font-size: .95rem; }
.page-card p  { font-size: .8rem; color: #666; margin: 0; }

/* TABLES */
.tbl-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: .84rem; }
th {
  background: var(--navy); color: #fff; text-align: center;
  padding: .6rem .65rem; cursor: pointer; user-select: none; white-space: nowrap;
}
th:hover { background: #265480; }
td { padding: .48rem .65rem; border-bottom: 1px solid #eee; text-align: center; }
td:first-child { text-align: left; font-weight: 500; }
tr:hover td { background: #f0f6ff; }
.badge {
  display: inline-block; padding: .15rem .6rem;
  border-radius: 999px; font-size: .76rem; font-weight: 600; color: #fff;
}

/* CORR TABLE */
.corr-table th, .corr-table td {
  padding: .55rem .7rem; font-size: .82rem;
  border: 1px solid #ddd; white-space: nowrap;
}
.corr-table th { background: var(--navy); color: #fff; cursor: default; }
.corr-table td:first-child { font-weight: 600; background: #f5f7fa; color: #333; }

/* MATH */
.math-box {
  background: #f8f9fb; border-left: 4px solid var(--navy);
  border-radius: 4px; padding: 1rem 1.5rem; margin: .9rem 0 1.3rem;
  overflow-x: auto;
}
.step { margin-bottom: 1.6rem; }
.step-num {
  display: inline-flex; align-items: center; justify-content: center;
  width: 1.7rem; height: 1.7rem; background: var(--navy); color: #fff;
  border-radius: 50%; font-size: .8rem; font-weight: 700; margin-right: .5rem;
  flex-shrink: 0; vertical-align: middle;
}

/* RESULTS */
.scatter img { max-width: 100%; border: 1px solid #ddd; border-radius: 6px; }
.eq-card {
  background: #eef4fb; border: 1px solid #b5cceb;
  border-radius: 8px; padding: 1.2rem 1.5rem; margin: 1rem 0; text-align: center;
}
.key-stats { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.2rem 0; }
.ks { flex: 1; min-width: 140px; background: #f4f7fb;
  border-radius: 8px; padding: .9rem 1rem; text-align: center; }
.ks .val { font-size: 1.6rem; font-weight: 700; color: var(--navy); }
.ks .lbl { font-size: .76rem; color: #666; }

/* FOOTER */
footer { background: var(--navy); color: #7090b0; text-align: center;
  font-size: .78rem; padding: 1.2rem; margin-top: auto; }
footer a { color: #a0c0e0; }
"""

MATHJAX = """
  <script>MathJax = { tex: { inlineMath: [['$','$']] } };</script>
  <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>"""

SORT_JS = """
<script>
function sortTable(tid, col, numeric) {
  const tbody = document.querySelector('#'+tid+' tbody');
  const rows  = Array.from(tbody.rows);
  const asc   = !(tbody.dataset.sortCol == col && tbody.dataset.sortDir === 'asc');
  rows.sort((a, b) => {
    const av = a.cells[col].dataset.val ?? a.cells[col].textContent.trim();
    const bv = b.cells[col].dataset.val ?? b.cells[col].textContent.trim();
    if (numeric) return asc ? parseFloat(av)-parseFloat(bv) : parseFloat(bv)-parseFloat(av);
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  });
  rows.forEach(r => tbody.appendChild(r));
  tbody.dataset.sortCol = col; tbody.dataset.sortDir = asc ? 'asc' : 'desc';
}
</script>"""

# ---------------------------------------------------------------------------
# Shared page shell
# ---------------------------------------------------------------------------
def nav(active):
    pages = [("index","Home"), ("data","Data"),
             ("methods","Methods"), ("results","Results"),
             ("conclusion","Conclusion"), ("predictions","26' Predictions")]
    links = "".join(
        f'<a href="{slug}.html" class="{"active" if slug==active else ""}">{label}</a>'
        for slug, label in pages
    )
    return f'<nav><a class="brand" href="index.html">WC 2022 &mdash; Linear Algebra</a>{links}</nav>'

def page(slug, title, body, extra_head=""):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} | 2022 WC Linear Algebra</title>
  <style>{CSS}</style>{extra_head}
</head>
<body>
{nav(slug)}
<div class="page">
{body}
</div>
<footer>Math 2220 Linear Algebra &middot; Generated {datetime.date.today()}
  &nbsp;&mdash;&nbsp; <a href="index.html">Home</a></footer>
</body>
</html>"""

# ---------------------------------------------------------------------------
# index.html
# ---------------------------------------------------------------------------
index_body = f"""
<div class="hero">
  <h1>Does Playing in the Top&#8209;5 European Leagues Predict 2022 World Cup Success?</h1>
  <p class="sub">Math 2220 &middot; Linear Algebra Application Project</p>
  <div class="rq">
    <p><strong>RQ&nbsp;1 &mdash;</strong> Is a higher percentage of players in the top&#8209;5 European leagues
       associated with greater World Cup success?</p>
    <p><strong>RQ&nbsp;2 &mdash;</strong> Can stage reached be approximated with a least&#8209;squares model
       using percent in top&#8209;5 leagues and FIFA ranking as predictors?</p>
  </div>
  <div class="hero-links">
    <a class="btn btn-white" href="data.html">Explore the Data</a>
    <a class="btn btn-ghost" href="results.html">Jump to Results</a>
  </div>
</div>

<div class="card">
  <h2>Key Findings</h2>
  <div class="stat-grid">
    <div class="stat-card"><div class="val">{corr[0,2]:.3f}</div><div class="lbl">r(pct_top5, stage)</div></div>
    <div class="stat-card"><div class="val">{corr[1,2]:.3f}</div><div class="lbl">r(fifa_rank, stage)</div></div>
    <div class="stat-card"><div class="val">{r_sq:.3f}</div><div class="lbl">R&sup2; (2-predictor model)</div></div>
    <div class="stat-card"><div class="val">{n}</div><div class="lbl">Teams in sample</div></div>
    <div class="stat-card"><div class="val">5</div><div class="lbl">Top-5 leagues analysed</div></div>
  </div>
  <p>There is a <strong>moderate positive correlation</strong> (r&nbsp;=&nbsp;{corr[0,2]:.3f}) between
     a team&rsquo;s share of players in the top&#8209;5 European leagues and how far it advanced
     in Qatar. The two-predictor least-squares model explains <strong>{r_sq*100:.1f}%</strong> of
     the variance in stage reached, with FIFA ranking as the stronger individual predictor.</p>
</div>

<div class="card">
  <h2>Pages</h2>
  <div class="page-grid">
    <a class="page-card" href="data.html">
      <h3>Data</h3>
      <p>All 32 team squad stats &mdash; sortable table with % top-5, stage reached, goal diff.</p>
    </a>
    <a class="page-card" href="methods.html">
      <h3>Methods</h3>
      <p>Step-by-step linear algebra: variable definitions, covariance matrix, normal equations.</p>
    </a>
    <a class="page-card" href="results.html">
      <h3>Results</h3>
      <p>Color-coded correlation matrix, scatter plot, regression equation and residuals.</p>
    </a>
    <a class="page-card" href="conclusion.html">
      <h3>Conclusion</h3>
      <p>Interpretation, model limitations, and what drives the remaining unexplained variance.</p>
    </a>
    <a class="page-card" href="predictions.html">
      <h3>26' Predictions</h3>
      <p>Simulated 2026 World Cup bracket &mdash; winner decided by % top-5 league representation.</p>
    </a>
  </div>
</div>
"""

# ---------------------------------------------------------------------------
# data.html
# ---------------------------------------------------------------------------
def badge_stage(sv):
    colors = {1:"#888",2:"#3a7bd5",3:"#9b59b6",4:"#e67e22",5:"#c0392b",6:"#27ae60"}
    bg = colors.get(int(sv), "#888")
    return f'<span class="badge" style="background:{bg};">{STAGE[int(sv)]}</span>'

def pct_bar(v):
    pct = float(v)*100
    return (f'<div style="display:flex;align-items:center;gap:.4rem;">'
            f'<div style="width:66px;background:#dde8f4;border-radius:3px;height:7px;">'
            f'<div style="width:{pct:.1f}%;background:var(--navy);height:7px;border-radius:3px;"></div>'
            f'</div><span>{pct:.1f}%</span></div>')

header_cols = [
    ("Team","team",False), ("Squad","squad",True), ("# Top-5","num5",True),
    ("% Top-5","pct5",True), ("FIFA Rank","rank",True), ("Stage","stage",True),
    ("GS Pts","pts",True), ("Goal Diff","gdiff",True),
]
th_row = "".join(
    f'<th onclick="sortTable(\'data-table\',{i},{str(num).lower()})">{label} &#x25BF;</th>'
    for i,(label,_,num) in enumerate(header_cols)
)
td_rows = []
for row in rows:
    sv  = int(float(row["stage_reached"]))
    pct = float(row["pct_in_top5"])
    gd  = int(float(row["goal_differential"]))
    gd_str   = f'+{gd}' if gd > 0 else str(gd)
    gd_color = "#27ae60" if gd > 0 else ("#c0392b" if gd < 0 else "#666")
    td_rows.append(
        f'<tr>'
        f'<td data-val="{row["team"].strip()}">{row["team"].strip()}</td>'
        f'<td data-val="{row["squad_size"]}">{row["squad_size"]}</td>'
        f'<td data-val="{row["num_in_top5"]}">{row["num_in_top5"]}</td>'
        f'<td data-val="{pct}">{pct_bar(pct)}</td>'
        f'<td data-val="{row["fifa_ranking"]}">{int(float(row["fifa_ranking"]))}</td>'
        f'<td data-val="{sv}">{badge_stage(sv)}</td>'
        f'<td data-val="{row["group_stage_points"]}">{row["group_stage_points"]}</td>'
        f'<td data-val="{gd}" style="color:{gd_color};font-weight:600">{gd_str}</td>'
        f'</tr>'
    )

data_body = f"""
<div class="card">
  <h2>Dataset &mdash; 32 Teams, 2022 FIFA World Cup</h2>
  <p>Each row is one national team. The <strong>top&#8209;5 leagues</strong> are the
     Premier League, La Liga, Serie A, Bundesliga, and Ligue 1.
     Club assignments come from official FIFA squad lists (November 2022).
     <strong>Click any column header to sort.</strong></p>
  <div class="tbl-wrap">
    <table id="data-table" data-sort-col="" data-sort-dir="">
      <thead><tr>{th_row}</tr></thead>
      <tbody>{"".join(td_rows)}</tbody>
    </table>
  </div>
</div>
{SORT_JS}"""

# ---------------------------------------------------------------------------
# methods.html
# ---------------------------------------------------------------------------
methods_body = f"""
<div class="card">
  <h2>Methods &mdash; Linear Algebra Tools</h2>

  <div class="step">
    <h3><span class="step-num">1</span>Variable Definitions</h3>
    <p>For each team $i$ we define the primary <em>predictor</em> as the fraction of squad
       players who played in one of the five major European leagues:</p>
    <div class="math-box">
      $$p_i = \\frac{{\\text{{num\\_in\\_top5}}_i}}{{\\text{{squad\\_size}}_i}} \\in [0,1]$$
    </div>
    <p>The <em>response variable</em> is a numeric success score based on stage reached:</p>
    <ul>
      <li><strong>1</strong> = Group Stage &nbsp; <strong>2</strong> = Round of 16 &nbsp;
          <strong>3</strong> = Quarterfinal</li>
      <li><strong>4</strong> = Semifinal &nbsp; <strong>5</strong> = Finalist &nbsp;
          <strong>6</strong> = Champion</li>
    </ul>
    <p>A second predictor is the <strong>FIFA ranking</strong> $r_i$ (October 2022).
       Note that a <em>lower</em> ranking number indicates a stronger team, so we expect
       a negative correlation with stage reached.</p>
  </div>

  <div class="step">
    <h3><span class="step-num">2</span>Correlation Matrix</h3>
    <p>We stack four variables into an $n \\times 4$ data matrix $M$
       (pct_in_top5, fifa_ranking, stage_reached, goal_diff), then
       <strong>mean-center</strong> each column:</p>
    <div class="math-box">
      $$\\tilde{{M}} = M - \\mathbf{{1}}\\bar{{\\mathbf{{x}}}}^\\top
        \\qquad (\\bar{{\\mathbf{{x}}}} = \\text{{column-mean vector}})$$
    </div>
    <p>The <strong>sample covariance matrix</strong> is a single matrix product &mdash;
       this is the key linear algebra step:</p>
    <div class="math-box">
      $$C = \\frac{{1}}{{n-1}}\\,\\tilde{{M}}^\\top \\tilde{{M}} \\in \\mathbb{{R}}^{{4\\times 4}}$$
    </div>
    <p>Normalising by the outer product of standard deviations gives the
       <strong>Pearson correlation matrix</strong> $R$, with all entries in $[-1,1]$:</p>
    <div class="math-box">
      $$R_{{ij}} = \\frac{{C_{{ij}}}}{{\\sigma_i\\,\\sigma_j}},
        \\qquad \\sigma_i = \\sqrt{{C_{{ii}}}}$$
    </div>
    <p>With $n={n}$ teams, $C$ is $4\\times 4$ and symmetric.
       Its diagonal is all ones by construction. See the
       <a href="results.html">Results page</a> for the computed matrix.</p>
  </div>

  <div class="step">
    <h3><span class="step-num">3</span>Least-Squares via Normal Equations</h3>
    <p>We seek $\\boldsymbol{{\\beta}} \\in \\mathbb{{R}}^3$ that minimises the
       squared residual between the linear model and the observed stages $y$:</p>
    <div class="math-box">
      $$\\min_{{\\boldsymbol{{\\beta}}}}\\;\\|y - X\\boldsymbol{{\\beta}}\\|^2,
        \\qquad X = \\begin{{bmatrix}}1 & p_1 & r_1 \\\\ \\vdots & \\vdots & \\vdots \\\\
        1 & p_{{n}} & r_{{n}}\\end{{bmatrix}} \\in \\mathbb{{R}}^{{{n}\\times 3}}$$
    </div>
    <p>Setting $\\nabla_{{\\boldsymbol{{\\beta}}}}\\|y-X\\boldsymbol{{\\beta}}\\|^2 = 0$
       yields the <strong>normal equations</strong>:</p>
    <div class="math-box">
      $$X^\\top X\\,\\boldsymbol{{\\beta}} = X^\\top y
        \\quad\\Longrightarrow\\quad
        \\boldsymbol{{\\beta}} = (X^\\top X)^{{-1}}X^\\top y$$
    </div>
    <p>Here $X^\\top X$ is a $3\\times 3$ symmetric positive-definite matrix, so the
       solution is unique and computed exactly via
       <code>numpy.linalg.solve</code> — no iteration required.</p>
  </div>

  <div class="step">
    <h3><span class="step-num">4</span>Goodness of Fit (R&sup2;)</h3>
    <p>We measure how much variance in stage reached the model explains using:</p>
    <div class="math-box">
      $$R^2 = 1 - \\frac{{\\|y - \\hat{{y}}\\|^2}}{{\\|y - \\bar{{y}}\\mathbf{{1}}\\|^2}}
        = 1 - \\frac{{SS_{{\\text{{res}}}}}}{{SS_{{\\text{{tot}}}}}}$$
    </div>
    <p>$R^2=1$ means a perfect fit; $R^2=0$ means the model does no better than
       simply predicting the mean for every team.</p>
  </div>
</div>"""

# ---------------------------------------------------------------------------
# results.html
# ---------------------------------------------------------------------------
corr_col_labels = VAR_NAMES
corr_th = "<tr><th></th>" + "".join(f"<th>{h}</th>" for h in corr_col_labels) + "</tr>"

def corr_color(v):
    v = float(v)
    if abs(v) >= 0.999:
        return "#cccccc", "#333"
    if v >= 0:
        r2,g2,b2 = 255, int(255*(1-v)), int(255*(1-v))
    else:
        r2,g2,b2 = int(255*(1+v)), int(255*(1+v)), 255
    fg = "white" if abs(v) > 0.65 else "#222"
    return f"rgb({r2},{g2},{b2})", fg

corr_tds = []
for i, rname in enumerate(corr_col_labels):
    cells = ""
    for j in range(4):
        bg, fg = corr_color(corr[i,j])
        cells += f'<td style="background:{bg};color:{fg};">{corr[i,j]:.4f}</td>'
    corr_tds.append(f"<tr><td>{rname}</td>{cells}</tr>")

res_rows = []
for i in range(n):
    gd = resid[i]
    color = "#27ae60" if gd > 0.1 else ("#c0392b" if gd < -0.1 else "#666")
    res_rows.append(
        f'<tr><td>{teams[i]}</td><td>{STAGE[int(s[i])]}</td>'
        f'<td>{y_hat[i]:.2f}</td>'
        f'<td style="color:{color};font-weight:600">{gd:+.2f}</td></tr>'
    )

results_body = f"""
<div class="card">
  <h2>Correlation Matrix</h2>
  <p>Color scale: <span style="background:#ff8888;padding:1px 10px;border-radius:3px;">red&nbsp;=&nbsp;positive</span>
     &nbsp; <span style="background:#8899ff;color:#fff;padding:1px 10px;border-radius:3px;">blue&nbsp;=&nbsp;negative</span>
     &nbsp; white&nbsp;&asymp;&nbsp;0. Diagonal (always 1) is grey.</p>
  <div class="tbl-wrap" style="margin-top:1rem;">
    <table class="corr-table">
      <thead>{corr_th}</thead>
      <tbody>{"".join(corr_tds)}</tbody>
    </table>
  </div>
  <p style="margin-top:1rem;font-size:.88rem;color:#555;">
    Key takeaways: <strong>pct_in_top5 &harr; stage_reached&nbsp;r={corr[0,2]:.3f}</strong>&nbsp;
    (moderate positive); &nbsp;
    <strong>fifa_ranking &harr; stage_reached&nbsp;r={corr[1,2]:.3f}</strong>&nbsp;
    (moderate negative &mdash; lower rank&nbsp;=&nbsp;stronger team).
  </p>
</div>

<div class="card">
  <h2>Scatter Plot &mdash; % Top-5 vs Stage Reached</h2>
  <div class="scatter">
    <img src="scatter.png" alt="Scatter: pct top-5 vs stage reached">
  </div>
  <p style="margin-top:.8rem;font-size:.88rem;color:#555;">
    The red line is the simple (one-predictor) least-squares fit.
    The positive slope confirms the correlation above.
  </p>
</div>

<div class="card">
  <h2>Least-Squares Regression Model</h2>
  <p>Solving the normal equations with two predictors gives:</p>
  <div class="eq-card">
    $$\\hat{{y}} = {beta[0]:.4f} + {beta[1]:.4f}\\,x_1 + ({beta[2]:.4f})\\,x_2$$
    <p style="font-size:.85rem;margin-top:.6rem;color:#555;">
      $x_1$&nbsp;=&nbsp;pct_in_top5 &nbsp;|&nbsp;
      $x_2$&nbsp;=&nbsp;FIFA ranking &nbsp;|&nbsp;
      $\\hat{{y}}$&nbsp;=&nbsp;predicted stage reached
    </p>
  </div>
  <div class="key-stats">
    <div class="ks"><div class="val">{corr[0,2]:.3f}</div><div class="lbl">r(pct_top5, stage)</div></div>
    <div class="ks"><div class="val">{corr[1,2]:.3f}</div><div class="lbl">r(fifa_rank, stage)</div></div>
    <div class="ks"><div class="val">{r_sq:.3f}</div><div class="lbl">R&sup2;</div></div>
    <div class="ks"><div class="val">{beta[1]:.3f}</div><div class="lbl">slope (pct_top5)</div></div>
    <div class="ks"><div class="val">{beta[2]:.3f}</div><div class="lbl">slope (fifa_rank)</div></div>
  </div>
</div>

<div class="card">
  <h2>Residuals by Team</h2>
  <p><span style="color:#27ae60;font-weight:600">Green</span> = outperformed model &nbsp;
     <span style="color:#c0392b;font-weight:600">Red</span> = underperformed model.</p>
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
</div>"""

# ---------------------------------------------------------------------------
# conclusion.html
# ---------------------------------------------------------------------------
conclusion_body = f"""
<div class="card">
  <h2>What the Model Found</h2>
  <p>There is a <strong>moderate positive correlation</strong> (r&nbsp;=&nbsp;{corr[0,2]:.3f})
     between a team&rsquo;s share of players in the top&#8209;5 European leagues and how far it
     advanced in Qatar 2022. FIFA ranking is a slightly stronger individual predictor
     (r&nbsp;=&nbsp;{corr[1,2]:.3f}), with the negative sign reflecting that a lower rank
     number means a stronger team.</p>
  <p>Together the two predictors explain <strong>{r_sq*100:.1f}% of the variance</strong>
     in stage reached (R²&nbsp;=&nbsp;{r_sq:.3f}). The regression coefficient on pct_in_top5
     is <strong>{beta[1]:.3f}</strong>: a team with 100% of players in top&#8209;5 leagues is
     predicted to reach {beta[1]:.2f} stages further than a team with 0%, holding FIFA ranking
     constant. Each one&#8209;place drop in FIFA ranking shifts the predicted stage by
     {beta[2]:.3f}.</p>
</div>

<div class="card">
  <h2>Limitations</h2>
  <ul>
    <li><strong>Small sample (n={n}).</strong> A handful of upsets &mdash; Saudi Arabia beating
        Argentina in the group stage, Morocco reaching the semi-finals &mdash; can shift the
        regression line noticeably at this sample size.</li>
    <li><strong>Ordinal response.</strong> <em>Stage reached</em> is an ordered category, not a
        continuous quantity. The gap between Group Stage and Round of 16 is not necessarily the
        same &ldquo;distance&rdquo; as between Semifinal and Finalist, yet the model treats them
        equally.</li>
    <li><strong>Multicollinearity.</strong> pct_in_top5 and FIFA ranking are themselves correlated
        (r&nbsp;=&nbsp;{corr[0,1]:.3f}), making it harder to isolate the individual contribution
        of each predictor.</li>
    <li><strong>Omitted variables.</strong> Injuries, tournament draw luck, home&#8209;continent
        advantage (Qatar, Morocco), squad chemistry, and coaching quality all influence results but
        are absent from the model.</li>
    <li><strong>Playing time not captured.</strong> Being registered at a top&#8209;5 club does not
        mean playing time. A bench player at PSG is counted the same as a starter at Bayern.</li>
  </ul>
</div>

<div class="card">
  <h2>Takeaway</h2>
  <p>The two linear algebra tools &mdash; <strong>correlation matrices</strong> (built via
     mean&#8209;centering and matrix multiplication) and <strong>least&#8209;squares regression</strong>
     (solved exactly through the normal equations) &mdash; give a clean quantitative picture of the
     relationship between club&#8209;league environment and international success.</p>
  <p>Top&#8209;5 league experience is <em>meaningfully</em> associated with World Cup success, but it
     explains only about a third of the variation. The remaining two&#8209;thirds is a reminder that
     tournament football is genuinely unpredictable &mdash; and that is precisely what makes it worth
     watching.</p>
</div>"""

# ---------------------------------------------------------------------------
# 2026 Predictions simulation
# ---------------------------------------------------------------------------
DATA_2026 = ROOT.parent / "world_cup_2026_data.csv"
rows_26 = list(csv.DictReader(open(DATA_2026)))

def team_info(row):
    return {
        "name": row["team"].strip(),
        "pct":  float(row["pct_in_top5"]),
        "rank": float(row["fifa_ranking"]),
        "group": row["group_id"].strip(),
    }

teams_26 = [team_info(r) for r in rows_26]

def match_winner(a, b):
    """Return winner dict; higher pct wins, lower rank number breaks ties."""
    if a["pct"] != b["pct"]:
        return a if a["pct"] > b["pct"] else b
    return a if a["rank"] < b["rank"] else b

# --- Group stage ---
groups = defaultdict(list)
for t in teams_26:
    groups[t["group"]].append(t)

group_results = {}   # group_id -> sorted list of team dicts with wins/pct_sum
group_stage_eliminated = []
group_winners = []    # 1st-place teams
group_runners = []    # 2nd-place teams
third_place_teams = []  # 3rd-place teams (need best 8)

for gid in sorted(groups.keys()):
    members = groups[gid]
    wins = defaultdict(int)
    pct_sum = defaultdict(float)   # use pct_in_top5 margin as tiebreaker proxy
    # round-robin
    for i in range(len(members)):
        for j in range(i+1, len(members)):
            w = match_winner(members[i], members[j])
            l = members[j] if w is members[i] else members[i]
            wins[w["name"]] += 1
            pct_sum[w["name"]] += (w["pct"] - l["pct"])
    # sort: wins desc, then pct_sum desc, then rank asc
    ranked = sorted(members,
                    key=lambda t: (-wins[t["name"]], -pct_sum[t["name"]], t["rank"]))
    group_results[gid] = ranked
    group_winners.append(ranked[0])
    group_runners.append(ranked[1])
    third_place_teams.append(ranked[2])
    group_stage_eliminated.extend(ranked[3:])

# Pick best 8 third-place teams (by pct then rank)
third_sorted = sorted(third_place_teams,
                      key=lambda t: (-t["pct"], t["rank"]))
best_thirds = third_sorted[:8]
group_stage_eliminated.extend(third_sorted[8:])

# --- Build Round of 32 (32 teams) ---
# 24 group advancers + 8 best thirds = 32 teams total
# Seeded bracket: sort all 32 by pct desc (rank asc tiebreak),
# then classic 1v32, 2v31, ... 16v17 pairing
group_ids = sorted(groups.keys())  # A-L
all_r32 = []
for gid in group_ids:
    all_r32.append(group_results[gid][0])  # winners
    all_r32.append(group_results[gid][1])  # runners
all_r32 += best_thirds                     # 8 best thirds

# sort by pct desc, rank asc (same as match_winner logic)
all_r32_sorted = sorted(all_r32, key=lambda t: (-t["pct"], t["rank"]))

# classic seeded bracket: seed 1 vs seed 32, seed 2 vs seed 31, ...
r32_pairs = [(all_r32_sorted[i], all_r32_sorted[31 - i]) for i in range(16)]

def run_bracket_tracked(pairs):
    """Run bracket tracking full match results (team_a, team_b, winner, loser)."""
    def mr(a, b):
        w = match_winner(a, b)
        return {"team_a": a, "team_b": b, "winner": w, "loser": b if w is a else a}

    r32_res = [mr(a, b) for a, b in pairs]
    r16t = [m["winner"] for m in r32_res]
    r16_res = [mr(r16t[i], r16t[i+1]) for i in range(0, 16, 2)]
    qft = [m["winner"] for m in r16_res]
    qf_res = [mr(qft[i], qft[i+1]) for i in range(0, 8, 2)]
    sft = [m["winner"] for m in qf_res]
    sf_res = [mr(sft[i], sft[i+1]) for i in range(0, 4, 2)]
    third_teams = [m["loser"] for m in sf_res]
    third_res = mr(third_teams[0], third_teams[1])
    finalists = [m["winner"] for m in sf_res]
    final_res = mr(finalists[0], finalists[1])
    return r32_res, r16_res, qf_res, sf_res, final_res, third_res

r32_res, r16_res, qf_res, sf_res, final_res, third_res = run_bracket_tracked(r32_pairs)

champion   = final_res["winner"]
runner_up  = final_res["loser"]
third_place = third_res["winner"]
sf_losers  = [m["loser"] for m in sf_res]
qf_losers  = [m["loser"] for m in qf_res]
r16_losers = [m["loser"] for m in r16_res]
r32_losers = [m["loser"] for m in r32_res]


# SVG Bracket generator

def gen_bracket_svg(r32_res, r16_res, qf_res, sf_res, final_res, third_res):
    SH   = 18 
    GAP  = 3
    UNIT = SH + GAP
    SW   = 72
    CONN = 7
    RS   = SW + 2*CONN

    N      = 4
    PAD_T  = 42
    PAD_H  = 8
    CENTER_W = 120

    BRACKET_H = 16 * UNIT
    SVG_W = PAD_H + N*RS + CENTER_W + N*RS + PAD_H
    SVG_H = PAD_T + BRACKET_H + 90 

    CX = PAD_H + N*RS + CENTER_W//2

    def r32y(k):
        return PAD_T + k*UNIT + SH//2

    def tcy(r, m, t):
        """Center-y of team t in round r, match m (shared for both sides)."""
        if r == 0:
            return r32y(m*2 + t)
        pm = m*2 + t
        return (tcy(r-1, pm, 0) + tcy(r-1, pm, 1)) // 2

    def mcy(r, m):
        return (tcy(r, m, 0) + tcy(r, m, 1)) // 2


    def left_sx(r):          return PAD_H + r*RS
    def left_vbar(r):        return left_sx(r) + SW + CONN

    def right_sx(r):         return PAD_H + N*RS + CENTER_W + (N-1-r)*RS
    def right_vbar(r):       return right_sx(r) - CONN

    out = []

    WIN_BG  = "#d4edda"
    LOSE_BG = "#fff8e1"
    LC      = "#6090b8"   
    NAVY    = "#1a3a5c"

    def rect(x, y, w, h, fill, stroke=NAVY, rx=2, sw=1):
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                   f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" rx="{rx}"/>')

    def line(x1, y1, x2, y2, color=LC, sw=1.5):
        out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                   f'stroke="{color}" stroke-width="{sw}"/>')

    def txt(x, y, s, anchor="start", size=9, bold=False, color=NAVY):
        fw = "bold" if bold else "normal"
        out.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{fw}" '
                   f'fill="{color}" text-anchor="{anchor}" '
                   f'font-family="Arial,sans-serif">{s}</text>')

    def trunc(name, n=11):
        return name if len(name) <= n else name[:n-1] + "…"

    def draw_slot(x, yc, team, is_win, side="left"):
        yt = yc - SH//2
        rect(x, yt, SW, SH, WIN_BG if is_win else LOSE_BG)
        nm  = trunc(team["name"])
        pct = f'{team["pct"]*100:.0f}%'
        base_y = yt + SH - 5
        if side == "left":
            txt(x+3,    base_y, nm,  bold=is_win, size=9)
            txt(x+SW-3, base_y, pct, anchor="end", size=8, color="#555")
        else:
            txt(x+SW-3, base_y, nm,  anchor="end", bold=is_win, size=9)
            txt(x+3,    base_y, pct, anchor="start", size=8, color="#555")

    def draw_left(r, m, md):
        sx   = left_sx(r)
        t0y, t1y = tcy(r,m,0), tcy(r,m,1)
        mc   = mcy(r, m)
        vx   = left_vbar(r)

        draw_slot(sx, t0y, md["team_a"], md["winner"] is md["team_a"])
        draw_slot(sx, t1y, md["team_b"], md["winner"] is md["team_b"])

        line(sx+SW, t0y, vx, t0y)
        line(sx+SW, t1y, vx, t1y)
        line(vx, t0y, vx, t1y)

        if r < N-1:
            line(vx, mc, left_sx(r+1), mc)
        else:
            box_left = CX - 70
            line(vx, mc, box_left, mc)

    def draw_right(r, m, md):
        sx   = right_sx(r)
        t0y, t1y = tcy(r,m,0), tcy(r,m,1)
        mc   = mcy(r, m)
        vx   = right_vbar(r)

        draw_slot(sx, t0y, md["team_a"], md["winner"] is md["team_a"], side="right")
        draw_slot(sx, t1y, md["team_b"], md["winner"] is md["team_b"], side="right")

        line(sx, t0y, vx, t0y)
        line(sx, t1y, vx, t1y)
        line(vx, t0y, vx, t1y)

        if r < N-1:
            line(vx, mc, right_sx(r+1)+SW, mc)
        else:
            box_right = CX + 70
            line(vx, mc, box_right, mc)

    labels = ["R32", "R16", "QF", "SF"]
    for r, lb in enumerate(labels):
        lx = left_sx(r) + SW//2
        txt(lx, PAD_T-10, lb, anchor="middle", size=9, bold=True)
        rx_col = right_sx(r) + SW//2
        txt(rx_col, PAD_T-10, lb, anchor="middle", size=9, bold=True)
    txt(CX, PAD_T-10, "FINAL", anchor="middle", size=9, bold=True, color="#b8860b")

    left_data = [r32_res[0:8], r16_res[0:4], qf_res[0:2], [sf_res[0]]]
    for r, matches in enumerate(left_data):
        for m, md in enumerate(matches):
            draw_left(r, m, md)

    right_data = [r32_res[8:16], r16_res[4:8], qf_res[2:4], [sf_res[1]]]
    for r, matches in enumerate(right_data):
        for m, md in enumerate(matches):
            draw_right(r, m, md)

    sf_y  = mcy(3, 0)
    BW, BH = 108, 80
    bx = CX - BW//2
    by = sf_y - BH//2

    rect(bx, by, BW, BH, "#eef3fc", stroke="#1a3a5c", rx=6, sw=2)

    fa, fb = final_res["team_a"], final_res["team_b"]
    champ_fin = final_res["winner"]

    slot_y_a = by + 14
    slot_y_b = slot_y_a + SH + 4

    rect(bx+6, slot_y_a, BW-12, SH, WIN_BG if champ_fin is fa else LOSE_BG, rx=2)
    txt(CX, slot_y_a+SH-5,
        f'{trunc(fa["name"],14)}  {fa["pct"]*100:.0f}%',
        anchor="middle", size=8, bold=(champ_fin is fa))

    rect(bx+6, slot_y_b, BW-12, SH, WIN_BG if champ_fin is fb else LOSE_BG, rx=2)
    txt(CX, slot_y_b+SH-5,
        f'{trunc(fb["name"],14)}  {fb["pct"]*100:.0f}%',
        anchor="middle", size=8, bold=(champ_fin is fb))

    txt(CX, slot_y_b+SH+13,
        f'CHAMPION: {champ_fin["name"]}',
        anchor="middle", size=9, bold=True, color="#b8860b")

    tp_y = PAD_T + BRACKET_H + 18
    txt(CX, tp_y, "3rd Place Match", anchor="middle", size=9, bold=True, color="#666")

    t3a, t3b = third_res["team_a"], third_res["team_b"]
    t3w = third_res["winner"]
    half = SW + 6
    tp_ax = CX - half - 14
    tp_bx = CX + 14

    rect(tp_ax, tp_y+6, SW, SH, WIN_BG if t3w is t3a else LOSE_BG)
    txt(tp_ax+3, tp_y+6+SH-5,
        f'{trunc(t3a["name"])}  {t3a["pct"]*100:.0f}%',
        size=8, bold=(t3w is t3a))

    txt(CX, tp_y+6+SH//2+4, "vs", anchor="middle", size=8, color="#888")

    rect(tp_bx, tp_y+6, SW, SH, WIN_BG if t3w is t3b else LOSE_BG)
    txt(tp_bx+3, tp_y+6+SH-5,
        f'{trunc(t3b["name"])}  {t3b["pct"]*100:.0f}%',
        size=8, bold=(t3w is t3b))

    txt(CX, tp_y+6+SH+16,
        f'3rd Place: {t3w["name"]}',
        anchor="middle", size=9, bold=True, color="#e67e22")

    return (f'<svg viewBox="0 0 {SVG_W} {SVG_H}" '
            f'xmlns="http://www.w3.org/2000/svg" style="display:block;width:100%;height:auto;">'
            + "\n".join(out) + "</svg>")

bracket_svg = gen_bracket_svg(r32_res, r16_res, qf_res, sf_res, final_res, third_res)

# Pill summary helpers
def team_pill(t, color, size="normal"):
    font_size = ".78rem" if size == "small" else ".85rem"
    pad = ".2rem .6rem" if size == "small" else ".3rem .8rem"
    return (f'<span style="display:inline-block;background:{color};color:#fff;'
            f'border-radius:999px;padding:{pad};font-size:{font_size};'
            f'font-weight:600;margin:.15rem .1rem;">{t["name"]}'
            f'<span style="font-weight:400;font-size:.75em;opacity:.85;">'
            f' {t["pct"]*100:.0f}%</span></span>')

def round_block(title, teams_list, color, bg):
    pills = "".join(team_pill(t, color) for t in teams_list)
    count = len(teams_list)
    return f'''
<div style="background:{bg};border-radius:10px;padding:1.2rem 1.5rem;margin-bottom:1rem;">
  <div style="font-weight:700;color:{color};font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.6rem;">{title} &nbsp;<span style="font-weight:400;color:#888;">({count} teams)</span></div>
  <div>{pills}</div>
</div>'''

sf_list = [t for t in sf_losers if t is not third_place and t is not runner_up]

pill_html = (
    round_block("Finalist (Runner-Up)", [runner_up],  "#c0392b", "#fdf3f3") +
    round_block("3rd Place",            [third_place], "#e67e22", "#fef9f3") +
    round_block("Semifinalists (out in SF)", sf_list,  "#9b59b6", "#f8f3fd") +
    round_block("Quarterfinalists (out in QF)", qf_losers, "#3a7bd5", "#f0f5ff") +
    round_block("Round of 16 Eliminated",   r16_losers,  "#2980b9", "#f0f8ff") +
    round_block("Round of 32 Eliminated",   r32_losers,  "#7f8c8d", "#f6f7f8") +
    round_block("Group Stage Eliminated", group_stage_eliminated, "#95a5a6", "#f8f9fa")
)

# group stage table
group_table_rows = ""
for gid in sorted(groups.keys()):
    ranked = group_results[gid]
    for pos, t in enumerate(ranked):
        if pos == 0:
            status = "1st - Advanced"
            sc = "#27ae60"
        elif pos == 1:
            status = "2nd - Advanced"
            sc = "#3a7bd5"
        elif t in best_thirds:
            status = "3rd - Advanced (best 3rd)"
            sc = "#e67e22"
        else:
            status = "3rd/4th - Eliminated"
            sc = "#c0392b"
        group_table_rows += (
            f'<tr><td>{gid}</td><td>{t["name"]}</td>'
            f'<td>{t["pct"]*100:.1f}%</td>'
            f'<td>#{int(t["rank"])}</td>'
            f'<td><span style="color:{sc};font-weight:600">{status}</span></td></tr>'
        )

champ_html = f'''
<div style="text-align:center;padding:2rem 1rem;">
  <div style="font-size:.85rem;color:#888;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem;">2026 World Cup Champion</div>
  <div style="font-size:2.4rem;font-weight:800;color:var(--navy);letter-spacing:-.01em;">{champion["name"]}</div>
  <div style="color:#888;font-size:.85rem;margin-top:.3rem;">{champion["pct"]*100:.1f}% top-5 &nbsp;&middot;&nbsp; FIFA #{int(champion["rank"])}</div>
</div>
'''

predictions_body = f"""
<div class="card">
  <h2>2026 World Cup Predictions</h2>
  <p>Each match is decided purely by <strong>% of squad in the top&#8209;5 European leagues</strong>.
     Where two teams are equal, the lower FIFA ranking number (i.e., stronger team) advances.
     The 2026 tournament has 48 teams in 12 groups of 4 &mdash; top&#8209;2 from each group
     (24 teams) plus the 8 best third&#8209;place finishers advance to a Round&#8209;of&#8209;32
     single&#8209;elimination bracket.</p>
  <p>Green slots = match winner &nbsp;&bull;&nbsp; Yellow slots = eliminated &nbsp;&bull;&nbsp;
     Each team label shows name and % top&#8209;5.</p>
</div>

<div class="card" style="text-align:center;">
  <h2>Predicted Champion</h2>
  {champ_html}
</div>

<div class="card">
  <h2>Tournament Bracket</h2>
  <div style="overflow-x:auto;padding-bottom:.5rem;">
    {bracket_svg}
  </div>
</div>

<div class="card">
  <h2>Results by Round</h2>
  {pill_html}
</div>

<div class="card">
  <h2>Group Stage Standings</h2>
  <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th>Group</th><th>Team</th><th>% Top-5</th><th>FIFA Rank</th><th>Result</th>
      </tr></thead>
      <tbody>{group_table_rows}</tbody>
    </table>
  </div>
</div>
"""

# Write all pages
pages = [
    ("index",       "Home",           index_body,       ""),
    ("data",        "Data",           data_body,        ""),
    ("methods",     "Methods",        methods_body,      MATHJAX),
    ("results",     "Results",        results_body,      MATHJAX),
    ("conclusion",  "Conclusion",     conclusion_body,   MATHJAX),
    ("predictions", "26' Predictions",predictions_body,  ""),
]

for slug, title, body, extra in pages:
    html = page(slug, title, body, extra)
    (DOCS / f"{slug}.html").write_text(html, encoding="utf-8")
    size = (DOCS / f"{slug}.html").stat().st_size // 1024
    print(f"  {slug}.html  ({size} KB)")

print(f"\nSite written to {DOCS}/")
print("\nTo go live with GitHub Pages:")
print("  1. git init  (if not already a repo)")
print("  2. git add docs/")
print('  3. git commit -m "Add project site"')
print("  4. git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git")
print("  5. git push -u origin main")
print("  6. GitHub repo Settings → Pages → Branch: main, Folder: /docs → Save")
print("  Your site will be at: https://YOUR_USERNAME.github.io/YOUR_REPO/")
