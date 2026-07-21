# Commands and Tools Explained — a senior-level reference to this project's codebase

This file explains every non-trivial command, function, and API call used across the
project's Python scripts — not just *what* each one does, but *why it was the right tool
here*, and what a senior engineer would watch out for. Organized by library. Read this
alongside the actual `step*.py` files, not instead of them.

---

## 1. pandas

### `df.groupby(col).agg(...)`
Used everywhere feature engineering happens. The critical thing to understand: `.agg()`
with a dict/kwargs lets you apply a **different** aggregation function to each column in
one pass, which is both faster and clearer than looping.
```python
stops = raw.groupby("STOPCODE").agg(
    stop_name=("STOPNAME", "first"),
    total_boardings=("Boardings", "sum"),
    n_route_dir_qhr_rows=("Boardings", "size"),
)
```
`"first"` grabs one representative value (assumes all rows for a STOPCODE share the same
name — true here, but this is a silent-failure risk if that assumption ever breaks: `"first"`
won't error if names differ, it'll just silently pick one). `"size"` counts rows regardless
of value — this is what makes `service_coverage` a *supply-side row count*, not a
demand-weighted quantity (see `PROJECT_HANDOFF.md` / `experiment_log.md` for why that
distinction matters for the cold-start validity argument).

**Senior gotcha:** `.agg` silently drops rows with NaN in the grouping key. Always check
`len(before)` vs `len(after)` when you group — this project's scripts don't always do that
explicitly, relying instead on printed row counts as a manual sanity check.

### `df.merge(other, on=col, how="left")`
Standard join. `how="left"` is used deliberately throughout (e.g. joining AI23 features onto
stops) to preserve every stop even if a lookup fails — a stop with no OSM POIs nearby still
needs to exist in the output with NaN/0, not silently vanish. Always check `df.merge(...).isna().sum()`
after a left join — this project does that (see `step3_lsoa_features.py`'s postcode match-rate
prints), which is correct practice.

### `df.pivot(index=, columns=, values=)`
Used constantly to turn a long-format CV results table (`borough, model, WMAPE` — one row
per borough-model pair) into a wide format (`borough` as rows, one column per model) for
correlation analysis and per-borough comparison. This is the standard "tidy data → analysis
shape" transform. **Gotcha:** `pivot` (not `pivot_table`) throws if there are duplicate
`(index, columns)` pairs — this project relies on that as an implicit assertion that each
(borough, model) combination appears exactly once, which is a reasonable use of a strict
function as a correctness check.

### `pd.qcut(x, q=5, labels=False, duplicates="drop")`
Bins a continuous variable into 5 equal-*frequency* quantile buckets (not equal-width —
that's `pd.cut`). Used for the **stratified validation split**: sampling evenly across
5 demand quantiles so early-stopping's validation signal sees the full boarding-demand
range, not just the majority of low-demand stops a random sample would over-represent.
`duplicates="drop"` handles the case where quantile boundaries collide (common with
skewed/discrete data) by merging bins rather than erroring.

### `df.groupby(...).size()` / `.nunique()`
`.size()` counts rows per group (including duplicates); `.nunique()` counts distinct values
per group. The LSOA-nests-within-borough verification used exactly this:
```python
lsoa_to_boroughs = df.groupby("lsoa11cd")["lad_name"].nunique()
crossing = (lsoa_to_boroughs > 1).sum()   # 3 of 4222 — see experiment_log.md
```
This is the correct, minimal-code way to empirically check a hierarchical-nesting
assumption on real data rather than trusting it by construction.

---

## 2. NumPy

### `np.log1p(x)` / `np.expm1(x)`
`log1p(x) = log(1+x)`, `expm1(x) = exp(x)-1`. Used instead of plain `log`/`exp` because the
data contains true zeros (zero-boardings stops, zero-count POI categories) — `log(0)` is
`-inf`, but `log1p(0) = 0`. This is the standard, correct transform for right-skewed count
data with zeros, used consistently for every feature *and* the target throughout the
project.

### `np.radians(x)` + `BallTree(..., metric="haversine")`
Haversine distance (great-circle distance on a sphere) requires coordinates in **radians**,
not degrees — a very common silent-bug source (forgetting to convert gives distances that
are off by a factor of ~57, since 1 radian ≈ 57.3°). This project converts explicitly before
every `BallTree` call. `BallTree` itself is a spatial index structure (like a k-d tree, but
supports non-Euclidean metrics like haversine) — using it instead of computing an $O(n^2)$
distance matrix is what makes the K=5 KNN graph construction over 17,943 nodes take ~1
second instead of minutes.

### `np.argpartition(-arr, k, axis=1)`
Used in the functional-similarity edge builder to get each row's top-`k` highest values
**without fully sorting** — `argpartition` is $O(n)$ average case vs.\ `argsort`'s
$O(n \log n)$, and you only need the top-k unordered, not a full ranking. A senior-level
micro-optimization that mattered here because the naive $\rho{>}0.8$ approach already
produced a 48-million-edge graph — every bit of the sparsification path needed to be fast.

### `np.corrcoef` — NOT used directly; understand why
The functional-similarity edge builder deliberately avoids `np.corrcoef` on the full
17,943×17,943 matrix (would be a ~2.5GB dense array) and instead standardizes each row to
zero-mean/unit-norm, then computes correlation via **batched matrix multiplication**
(`U[start:end] @ U.T`) in chunks of 2000 rows. This is the standard trick for "I need a
correlation/similarity matrix too large to materialize at once" — process in blocks, extract
what you need (top-k, thresholded edges), discard the block. Worth understanding even
outside this project: it's the same pattern behind flash-attention-style chunked computation
in transformers.

---

## 3. scikit-learn

### `StandardScaler().fit_transform(X_train)` / `.transform(X_test)`
**The single most important correctness pattern in the whole codebase.** `fit` computes
mean/std from training data only; `transform` applies those *same* stored values to test
data. Calling `fit_transform` on test data (or on the whole dataset before splitting) is the
classic silent leakage bug — the scaler would "see" the test distribution. Every single fold
in this project re-fits a fresh `StandardScaler` on that fold's training stops only. This is
correct and was explicitly called out as a fix in the bug log (fitting globally shifts the
mean by up to 4.4% for extreme boroughs like City of London).

### `RandomForestRegressor(n_estimators=150, max_features="sqrt", min_samples_leaf=5, random_state=SEED)`
`max_features="sqrt"` (each split considers $\sqrt{p}$ random features, not all $p$) is what
makes it a *random* forest rather than bagged decision trees — it decorrelates the trees.
`min_samples_leaf=5` is a regularization knob (prevents leaves from memorizing single
outlier stops). `random_state` fixes RF's internal bootstrap-sampling and feature-subset
randomness — but note this does **not** make RF's *output* deterministic across different
machines/sklearn versions in all cases; it makes it reproducible *given* the same environment.

### `Ridge(alpha=1.0)`
L2-regularized linear regression. `alpha` is the regularization strength — `alpha=1.0` is a
reasonable, unturned default (see the senior critique above: this was never tuned). Chosen
over plain `LinearRegression` specifically *because* the AI23 features are severely
multicollinear (VIF up to 37.7) — Ridge's L2 penalty is the standard remedy for
multicollinearity's effect on coefficient variance, even though (as this project found) it
doesn't necessarily change *predictive* WMAPE much once other regularized/robust models are
already in the comparison.

### `PCA(n_components=3, random_state=SEED)`
Principal Component Analysis — finds the orthogonal directions of maximum variance in the
(standardized) 8-dimensional AI23 feature block. `random_state` matters here because PCA's
solver can involve randomized SVD for large matrices, though for an 8-column input this is
mostly for reproducibility hygiene rather than because the algorithm is meaningfully
stochastic at this scale.

### `BallTree` (`sklearn.neighbors`)
Covered above under NumPy/spatial — worth re-noting here since it's the sklearn API. Used
identically for both the geographic KNN graph and the IDW baseline's neighbour lookup.

### `mean_squared_error`, `mean_absolute_error` (`sklearn.metrics`)
Standard metric functions, used to report RMSE/MAE alongside the project's primary metric
(WMAPE, which is hand-rolled — see below — because sklearn has no built-in weighted MAPE).

---

## 4. XGBoost

### `XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=SEED, verbosity=0)`
Gradient-boosted trees. `n_estimators` (number of boosting rounds) and `learning_rate`
trade off against each other (more rounds at a lower rate = smoother, less overfit-prone
convergence, at higher compute cost); `max_depth=6` is XGBoost's own commonly-cited
reasonable default, limiting how complex any single tree can get. `verbosity=0` suppresses
XGBoost's per-round training log, which would otherwise flood stdout across 33 folds ×
however many experiments. Chosen specifically because Yusuf et al. (2025) found gradient
boosting best-in-class for stop-level transit prediction — this project's config choices are
literature-aligned, not arbitrary.

---

## 5. SciPy statistics

### `scipy.stats.wilcoxon(a, b)`
The **Wilcoxon signed-rank test** — a non-parametric test for whether paired differences
(here: same 33 boroughs, two models) are symmetrically distributed around zero. Used instead
of a paired t-test because WMAPE differences across boroughs are not assumed normally
distributed (33 samples, some boroughs are structural outliers like City of London) — the
Wilcoxon test only assumes the differences are symmetric, a much weaker assumption, and it's
robust to outliers since it works on **ranks**, not raw magnitudes.

### `scipy.stats.pearsonr(x, y)` / `scipy.stats.spearmanr(x, y)`
Pearson measures *linear* correlation; Spearman measures *monotonic* correlation via ranks
(robust to outliers and non-linear-but-monotonic relationships). This project deliberately
runs **both** on every mechanism-hypothesis test and treats a result as robust only if it
survives both — this caught (and correctly discounted) two hypotheses (sample-size artifact,
geographic-scale artifact) whose apparent Pearson significance did not survive the
rank-based test, indicating outlier-driven leverage rather than a real relationship.

### Holm-Bonferroni correction (hand-implemented, not a library call)
Multiple-comparison correction: when running $k$ significance tests in one "family," the
chance that at least one shows $p<0.05$ purely by chance grows with $k$. Holm-Bonferroni
sorts p-values ascending and requires the $i$-th smallest to beat $\alpha/(k-i+1)$, stopping
at the first failure — strictly more powerful than plain Bonferroni ($\alpha/k$ for every
test) while still controlling the family-wise error rate. Implemented by hand here (see
`step4g_gated_analysis.py`) rather than via `statsmodels.stats.multitest` — a fine choice for
a 2-test family, but `statsmodels`' `multipletests(pvals, method="holm")` would be the
standard library call for a larger family, worth knowing for future work.

---

## 6. PyTorch

### `torch.manual_seed(seed)` / `np.random.seed(seed)`
Sets the global RNG state for PyTorch and NumPy respectively. **Important limitation this
project discovered empirically**: this does *not* guarantee bit-identical results across
runs unless combined with `torch.use_deterministic_algorithms(True)` (not set here) — some
GPU/CPU kernels use non-deterministic reduction orders for performance. The 0.63pp swing
between two "identical-seed" MLP runs (Experiment Set 8's follow-up) is a direct, empirical
demonstration of this gap between "seeded" and "deterministic."

### `nn.Module`, `nn.Linear`, `nn.Sequential`, `.forward()`
Standard PyTorch model-building blocks. `nn.Module` subclasses define parameters in
`__init__` and the computation graph in `forward()`; PyTorch's autograd traces every
operation on tensors with `requires_grad=True` (the default for `nn.Parameter`s) to build a
computation graph automatically, which `loss.backward()` then differentiates.

### `F.elu`, `F.dropout(x, p=, training=self.training)`
`F.elu` (exponential linear unit) is the activation function used throughout instead of
ReLU — ELU is smooth and non-zero for negative inputs, which can help gradient flow;
consistent choice across every model in this project (GATv2Model, MLPModel, the fusion and
gated variants) so no comparison is confounded by activation-function differences.
`training=self.training` is critical: dropout must be **active** during training and
**disabled** during evaluation, and `self.training` is a flag PyTorch sets automatically via
`model.train()` / `model.eval()`. Getting this wrong in the *validation loss* computation
specifically was Bug 3 — the single most consequential bug in the project.

### `torch.optim.Adam(params, lr=5e-4, weight_decay=1e-4)`
Adam is an adaptive-learning-rate optimizer (maintains per-parameter momentum and variance
estimates). `weight_decay` adds L2 regularization directly into the optimizer step. `lr=5e-4`
is on the conservative side, tuned down from an initial `1e-3` specifically as part of the
Bug 5 fix (residual skip connection needed a lower learning rate to train stably).

### `F.huber_loss(pred, target, delta=0.5)`
Huber loss is quadratic (like MSE) for small errors and linear (like MAE) for large ones,
switching at `delta`. Chosen over plain MSE specifically because `log1p(boardings)` still
has a residual heavy tail after the log transform — Huber makes training less sensitive to
the handful of very-high-demand outlier stops that would otherwise dominate an MSE loss's
gradient.

### `torch.no_grad()`
Context manager that disables gradient tracking — used for every inference/evaluation pass.
Not just a performance optimization (though it is one): using it during evaluation also
prevents accidentally accumulating a computation graph across the training loop, which would
leak memory over 33 folds × multiple models × multiple seeds if forgotten.

### `model.state_dict()` / `.load_state_dict()`
Used to implement manual early stopping: on each validation improvement, `{k: v.clone() for
k, v in model.state_dict().items()}` snapshots every parameter tensor (the `.clone()` is
essential — without it you'd store references that keep changing as training continues);
after the patience limit is hit, `load_state_dict()` restores the best-seen weights.

---

## 7. PyTorch Geometric (PyG)

### `GATv2Conv(in_channels, out_channels, heads=4, dropout=0.15, concat=True)`
The core graph attention layer (Brody et al. 2022's fix to the original GAT's static
attention). `heads=4` runs 4 independent attention mechanisms in parallel (multi-head
attention, same idea as in transformers); `concat=True` concatenates their outputs
(`4 × 64 = 256`-dim) rather than averaging them, giving the next layer more information to
work with. `dropout` here specifically means "randomly drop attention *edges* during
training," not standard neuron dropout — a graph-specific regularization.

### `torch_geometric.utils.subgraph(node_idx, edge_index, relabel_nodes=True, num_nodes=N)`
Extracts the induced subgraph on a given node subset (both endpoints of an edge must be in
`node_idx` for the edge to survive). `relabel_nodes=True` remaps the surviving nodes to a
dense `0..len(subset)-1` index — **essential** and easy to get wrong: if two subgraph calls
(e.g. `ei_tr` for training and `ei_ctx` for train+test context) are built from
*differently-ordered* node lists, the relabeled indices won't correspond to the same feature
rows, silently corrupting every downstream computation. This project consistently passes the
same ordered index tensor (`tr_idx` then `all_idx = concat([tr_idx, te_idx])`) to keep
relabeling consistent.

### `edge_index` tensor format: `torch.tensor([src_list, dst_list])`, shape `(2, E)`
PyG's standard sparse-graph representation: row 0 is source node indices, row 1 is
destination node indices, column $j$ is one directed edge. Undirected graphs are represented
by including both `(i,j)` and `(j,i)` explicitly (this project does so everywhere — "bidirectional"
in the code comments means literally appending both directions to the edge list).
`torch.unique(edge_index, dim=1)` deduplicates identical edges after combining multiple edge
sources (KNN + route, or KNN + route + functional-similarity).

---

## 8. Command-line / environment

### `python -u script.py`
`-u` forces unbuffered stdout/stderr — without it, `print()` output can sit in a buffer and
not appear until the buffer fills or the process exits, which is disastrous for monitoring a
2-hour training run's progress in real time (you'd see nothing, then everything, at the end).
Used on every long-running script in this project.

### `pip install package` vs.\ `tlmgr install package`
Two independent package managers for two independent toolchains: `pip` installs Python
packages (xgboost, etc.) into the Python environment; `tlmgr` installs LaTeX packages into
the TinyTeX distribution used to compile the PDF report. Conflating them (e.g. trying `pip
install titlesec`) would fail silently-confusingly since they manage entirely different
filesystems.

### `pdflatex -interaction=nonstopmode file.tex`
Compiles LaTeX to PDF. `-interaction=nonstopmode` prevents the compiler from **halting and
waiting for keyboard input** on an error (its default behavior) — critical when running
non-interactively, since without it a missing-package error would hang the process forever
rather than failing fast with a readable log. LaTeX documents with a table of contents or
cross-references (`\ref{}`) require **two compilation passes**: the first pass writes label
positions to an `.aux` file, the second pass reads that `.aux` file to resolve the
references — this is why "compile twice" is standard practice, not superstition.

### `BallTree(...).query(coords, k=k+1)` pattern (`k+1`, then drop the first column)
A recurring idiom in this codebase: querying `k+1` nearest neighbours and discarding the
first result, because a point's nearest neighbour to *itself* (distance 0) is always
returned first by a spatial index query against the same point set. Forgetting the `+1` is
a classic off-by-one that would silently give you `k-1` real neighbours plus yourself.

---

## Closing note

If you take one thing from this file into how you think about your own code going forward,
make it this: nearly every choice documented above was made *because* of a specific,
nameable failure mode it avoids (leakage, off-by-one, silent non-determinism, outlier
leverage, multicollinearity). A senior engineer's code doesn't look different because it uses
fancier tools — it looks different because every non-obvious choice has a one-sentence reason
attached to it, the way this file (and, credit where due, this project's own code comments)
tries to make explicit.
