# A Lightweight Attention LSTM Filter in the Momentum Transformer

An end-to-end research implementation of a Momentum Transformer for cross-sectional
equity momentum, built to answer one question honestly: **does adding a lightweight
attention path to the LSTM filter, and learning to blend it against a vanilla filter
by market regime, produce economically meaningful out-of-sample alpha?**

On this dataset, over this universe, under realistic costs, the answer is **no**.

That answer is the contribution. What follows documents the architecture, the data
pipeline, the evaluation protocol that produced it, and the diagnostics that make it
credible rather than merely negative.

---

## Table of Contents

1. [Headline Result](#headline-result)
2. [What This Repository Actually Establishes](#what-this-repository-actually-establishes)
3. [Inspiration and Intellectual Provenance](#inspiration-and-intellectual-provenance)
4. [Repository Layout](#repository-layout)
5. [Data Pipeline](#data-pipeline)
6. [Architecture](#architecture)
7. [Feature Engineering](#feature-engineering)
8. [Training Methodology](#training-methodology)
9. [Evaluation Protocol](#evaluation-protocol)
10. [Results](#results)
11. [Discussion: Reading the Null](#discussion-reading-the-null)
12. [Limitations and Threats to Validity](#limitations-and-threats-to-validity)
13. [Reproducing These Results](#reproducing-these-results)
14. [Test Suite](#test-suite)
15. [References](#references)

---

## Headline Result

Single chronological split. Train through 2020-12-31, validate on 2021, test on
everything after. Out-of-sample window **2022-02-28 → 2025-11-26** (6,594 hourly bars,
942 sessions, 53 symbols). All figures net of 6 bps one-way costs on realized turnover.

| Model | Sharpe (daily agg.) | Sharpe (per-bar ann.) | CAGR | Vol | Max DD | Total return | Deflated Sharpe |
|---|---|---|---|---|---|---|---|
| Vanilla | 0.328 | 0.340 | 4.34% | 16.45% | −23.46% | 17.31% | 0.033 |
| Attention | **0.513** | 0.532 | 7.58% | 16.22% | −22.63% | 31.63% | 0.070 |
| Regime ensemble | 0.507 | 0.526 | 7.11% | 15.29% | −25.18% | 29.49% | 0.068 |
| — | | | | | | | |
| SPY (buy & hold) | **0.783** | 0.783 | 12.85% | 17.36% | −23.81% | 57.54% | — |
| Equal-weight universe | 0.729 | 0.769 | 17.33% | 24.77% | −35.30% | 82.42% | — |
| Naive TSMOM | −0.018 | −0.063 | −15.96% | 52.84% | −75.16% | −48.01% | — |

Three facts settle the question:

1. **Every learned variant loses to buying SPY** over the same window, on both Sharpe
   conventions, before any adjustment for search effort.
2. **There is no alpha.** Regressed on SPY, the ensemble's annualized alpha is
   **−0.45% (t = −0.08)** with **beta 0.625** and **correlation 0.709**. The strategy is
   a concentrated, slightly-worse proxy for the index it is supposed to beat.
3. **The Sharpe does not survive multiple-testing adjustment.** Counting 93
   configurations, the expected maximum Sharpe under the null of zero skill is
   **1.295** against a realized 0.526 and a Sharpe standard error of 0.517. The
   deflated Sharpe — P(true SR > 0) — is **0.068** for the ensemble and **0.033** for
   the vanilla model. Nothing here clears a conventional bar.

The strategy is also **not market-neutral**, contrary to what a long/short framing
would imply: mean net exposure is **0.970** against gross exposure of 1.000. The
`tanh` head emits a predominantly positive cross-section (prediction mean 0.292), so
selecting the top-10 names by |signal| produces an essentially long-only book. The
beta and correlation above are the mechanical consequence.

---

## What This Repository Actually Establishes

**Negative findings (the substance):**

- A lightweight additive-attention path inside the LSTM filter buys ~0.18 of daily
  Sharpe over the vanilla filter (0.513 vs 0.328) — but from a base that is itself
  below the benchmark, and with a deflated Sharpe of 0.070 the gap is not
  distinguishable from search noise.
- **The regime-aware blend does not switch.** The weight network converged to a
  near-constant tilt toward the attention arm: mean attention weight **0.776**,
  standard deviation **0.028**, range **[0.649, 0.797]**, with **100%** of test
  observations above the 0.5 threshold. Zero observations favored the vanilla arm.
  The ensemble is therefore not a regime detector; it is a slightly damped copy of the
  attention model, and it performs like one (0.507 vs 0.513).
- **The edge lives entirely inside the bid-ask spread.** At 20 bps one-way the
  ensemble returns −2.82% with a 0.027 Sharpe. Interpolating between the 10 bps and
  20 bps scenarios puts break-even at roughly **19 bps for total return and 21 bps for
  Sharpe** — inside the plausible cost range for a weekly-rebalanced book in mid-cap
  names.
- **Walk-forward is a coin flip.** Across 22 quarterly out-of-sample windows,
  12 are positive, mean window Sharpe is 0.55, and the 95% CI on that mean is
  **[−0.80, 1.89]**.

**Positive findings (small, honestly sized):**

- The ensemble's pooled information coefficient is **0.0071** with a block-bootstrap
  95% CI of **[0.0035, 0.0109]** that excludes zero. Its cross-sectional IC t-statistic
  is 1.66 (p = 0.096), which is suggestive and not significant. There is a faint,
  correctly-signed signal; it is smaller than the cost of harvesting it.
- The attention arm generalizes better than the vanilla arm at equal capacity
  (157,444 vs 156,194 parameters — a 1,250-parameter difference, 0.8%). Whatever the
  attention path is doing, it is not buying performance with capacity.

**Methodological contributions (why the null is trustworthy):**

- Every feature is built from `close.shift(1)`; the target at bar *t* uses `close[t]`,
  a price no feature at row *t* has seen. This boundary is asserted by a perturbation
  test, and a deliberately leaky feature is included to prove the test can fail.
- Splits are back-adjusted, not deleted. Bad ticks are filtered against a *trailing*,
  shifted volatility estimate. Realized PnL is never winsorized.
- Windows spanning a calendar gap are rejected rather than silently treated as
  contiguous (41,939 rejections on the test split alone).
- Normalization statistics are fitted on the training split only and persisted.
- Model selection uses a validation score computed by *running the actual portfolio
  backtest* on the validation split — not a proxy objective the strategy never trades.
- Statistical claims are reported with honest standard errors: cross-sectional IC
  t-statistics, moving-block bootstrap CIs, and a deflated Sharpe that counts the
  search.

## Inspiration and Intellectual Provenance

Two papers are vendored in this repository, and they are not decoration — nearly every
non-obvious design decision in the codebase traces to one of them. They form a single
lineage out of the Oxford-Man Institute: the first establishes *what to optimize*, the
second establishes *what architecture to optimize it with*.

| Paper | File | Role in this project |
|---|---|---|
| Lim, Zohren & Roberts (2019), *Enhancing Time Series Momentum Strategies Using Deep Neural Networks* | `1904.04912v3.pdf` | The objective, the position, and the risk scaling |
| Wood, Giegerich, Roberts & Zohren (2022), *Trading with the Momentum Transformer* | `2112.08534v3.pdf` | The architecture, the interpretability, and the regime framing |

A third paper — Lim, Arık, Loeff & Pfister (2019), *Temporal Fusion Transformers*
(arXiv:1912.09363) — is cited in the source but not vendored. It reaches this project
second-hand: the Momentum Transformer's best architecture *is* a decoder-only TFT, so
the TFT's components arrive through Wood et al. rather than directly.

---

### Deep Momentum Networks — what to optimize

Lim, Zohren & Roberts start from a complaint about how machine learning is usually
applied to trading. Casting momentum as a regression on next-period returns, or a
classification of next-period direction, optimizes the wrong thing: it ignores risk
entirely, and — as they note, citing the trend-following literature — a strategy can
place more losing trades than winning ones and still be highly profitable, because it
sizes up only into large infrequent moves. High classification accuracy is therefore
not evidence of a good strategy. Their fix is to delete the intermediate prediction
target altogether and have the network emit the position directly, trained on a
risk-adjusted objective.

**What this project took, and why:**

| Taken | Where it lives | The intuition |
|---|---|---|
| **Sharpe ratio as the loss function**, $\mathcal{L} = -\sqrt{252}\,\mathbb{E}[R]/\sqrt{\text{Var}[R]}$ | `Utils/Losses.py::SharpeRatioLoss` | There is no ground-truth "correct position" to regress against, so supervision has to come from the portfolio-level outcome. Optimizing Sharpe directly makes the network trade off return against risk internally instead of leaving that to a post-hoc sizing rule. |
| **The network output *is* the position**, bounded by `tanh` into $[-1, 1]$ | `Momentum_transformer.py::prediction_head` | Their Equation (10) is $Z = (\tanh \circ f \circ g)(U)$ — trend estimation and position sizing collapse into one learned function. This repo's decision to end the head in `tanh` is a direct inheritance, and the README's [Architecture](#architecture) section explains why dropping it silently breaks the Sharpe objective. |
| **Volatility scaling to a 15% annualized target** | `Momentum_transformer.py::apply_volatility_target`, `TrainingConfig.target_volatility` | Both papers hold this fixed at 15% for comparability with the TSMOM literature. Its real job is approximate stationarity: without it, a single high-volatility asset dominates the portfolio's PnL and the network learns that asset's idiosyncrasies rather than momentum. |
| **Risk-adjusted momentum features**, $r_{t-w,t} / (\hat\sigma_t\sqrt{w})$ | `Feature_engineering.py`, the `momentum_*` block | Their normalized-returns construction. This is what makes `momentum_35` a genuinely different signal from `return_35` rather than a duplicate column — the repo has a test asserting exactly that. |
| **Multi-horizon normalized returns** | `FeatureConfig.return_days` | A trend strong at one horizon and weak at another is informative; handing the network several horizons lets it learn the blend rather than committing to one lookback. Their ladder is day / 1M / 3M / 6M / 1Y; this repo shifts it one notch shorter — hour / day / week / month / quarter / half-year — to suit hourly bars. |
| **MACD indicators** (Baz et al., reaching this project through DMN) | `StockAgnosticFeatureEngineer._macd` | Classical volatility-normalized trend estimators, included so the network has access to the signals the benchmarks use rather than having to rediscover them. |
| **Early stopping on a validation split, ≤100 epochs** | `TrainingConfig`, `Utils/Training.py` | Their calibration protocol. Patience is tightened from 25 to 15 here. |

**Taken but left unused:** DMN §VI-A proposes a **turnover regularizer** — folding the
cost term $c\,|X_t/\sigma_t - X_{t-1}/\sigma_{t-1}|$ directly into the Sharpe loss so
the network learns to avoid churn at training time. This repo implements it
(`Losses.py::SharpeWithTurnoverPenalty`) but never wires it to the CLI: every committed
run trains on the plain `SharpeRatioLoss`. Costs are therefore measured but never
optimized against — a gap that the [cost-sensitivity results](#transaction-cost-sensitivity)
make consequential.

---

### The Momentum Transformer — what to optimize it with

Wood et al. begin where DMN ends. Their observation is that the LSTM at the heart of a
DMN is *good at exactly the wrong thing*: it is built for local sequential processing,
and its recursive structure with a resetting forget gate makes it prone to discarding
information from before a regime change. That is a serious defect for momentum, whose
failure mode — as they put it, momentum strategies "work well until they don't" — is
precisely the turning point where a trend breaks down. The LSTM cannot draw on a past
regime that resembles the current one, because it has forgotten it.

Attention fixes the specific defect: it forms a direct connection to every previous
timestep, so a model can look back at an analogous historical regime instead of relying
on a compressed running summary. Their winning architecture is not pure attention,
though — it is an **attention-LSTM hybrid**: recurrent layers for local processing,
self-attention for long-term dependencies.

**What this project took, and why:**

| Taken | Where it lives | The intuition |
|---|---|---|
| **The hybrid itself** — LSTM for local structure, Transformer for global dependencies | `Momentum_transformer.py::MomentumTransformer` | This is the project's entire premise. The LSTM is not a competitor to attention; it is the front end that gives attention something structured to attend over. |
| **The LSTM as a *local filter*, made literal** | `LSTM.py`, `lstm_transformer_mode = "strided"` | The paper describes the LSTM as handling local processing; this repo takes that description at its word and runs the LSTM over consecutive 63-bar blocks, emitting one embedding per block for the Transformer to attend across. This is where the repository's name comes from, and it is a sharpening of the paper rather than a copy — the paper's own arrangement, one LSTM pass across the whole sequence, is implemented alongside it as the `full` mode. |
| **Causal masking** ($s_{t,\tau} = 0$ for $\tau > t$) | `Transformer_layers.py::build_causal_mask` | Their masked-MHA formulation, adopted so the architecture stays honest even before the portfolio layer enforces causality. |
| **Interpretable multi-head attention** — one value projection shared across heads, outputs averaged | `Transformer_layers.py::InterpretableMultiheadAttention` | The TFT construction the paper adopts (their MIMHA). Ordinary multi-head attention gives each head its own value space, so a head-averaged attention map compares incommensurable quantities. Sharing $W_v$ makes every head write into the same space, which is the only thing that makes the averaged attention matrix readable as "how much did position $i$ use position $j$." |
| **Multiple heads for concurrent regimes** | `num_attention_heads = 4` | Their argument that regimes operate concurrently at different timescales, and that separate heads can specialize to them. |
| **Regime change as *the* problem worth solving** | `Ensemble_model.py` (this repo's own answer) | The framing that motivates the whole ensemble. The paper's own solution is different — see below. |
| **Reporting a transaction-cost ladder rather than one number** | `BacktestConfig.cost_scenarios` | Their Exhibit 10 sweeps cost from 0 to 3 bps per architecture. The practice of showing the whole curve, not the most flattering point on it, is inherited directly. |

**Not taken — and the omissions matter.** Wood et al. improve regime handling with two
components this repository does not implement:

- **The Variable Selection Network.** A learned per-sample weighting over *input
  features*, which both filters low-signal covariates and yields the paper's variable-
  importance tables. This repo has no VSN; all 54 features enter the LSTM unweighted.
  The `DynamicWeightNetwork` is *not* an analogue — it weights two **models**, not
  features, and so provides none of the feature-attribution interpretability the paper
  gets for free.
- **The changepoint-detection module.** A Bayesian Gaussian-process online CPD
  preprocessing step, supplying changepoint severity and location as extra covariates.
  In the paper this is the single largest improvement in the hardest period: adding CPD
  raised the 2015–2020 portfolio Sharpe from 1.71 to 2.00, and roughly doubled it
  through the SARS-CoV-2 crash. This repo's `Utils/Regime_detector.py` is explicitly
  *not* that — its own docstring scopes it to descriptive PnL attribution, and it feeds
  no features to any model.

The regime ensemble is therefore this project's **substitute** for the paper's regime
machinery, not an implementation of it: rather than telling one model *when* the regime
changed, it trains two models and learns *which one to trust*. The
[blend-weight diagnostics](#ensemble-blend-weight-diagnostics) show that substitution
failing — and the failure is instructive. The paper's CPD module injects genuinely
exogenous information about regime boundaries. A blend network sees only the two arms'
shared feature window, and the arms are 99% identical architecture trained on the same
data with the same objective, so their errors are correlated and there is no regime in
which one reliably beats the other. The paper's approach adds information; the
ensemble only redistributes it.

---

### Where this project departs from both papers

These deviations are worth stating plainly, because a reader who knows the papers will
notice that they report Sharpe ratios of 1.7–2.6 where this repository reports 0.51,
and the gap is mostly explained here rather than by the architecture.

| Dimension | Both papers | This project | Consequence |
|---|---|---|---|
| **Asset class** | 50–88 ratio-adjusted continuous **futures** (Pinnacle CLC), spanning commodities, equity indices, fixed income and FX | 53 US **single-name equities and ETFs** | The largest departure. Wood et al. explicitly choose futures because they have "substantially less covariance structure than equities," which is what makes a diagonal-covariance, independently-sized portfolio valid. An equity cross-section is dominated by a single common factor — which is exactly why this project's book ends up with 0.63 beta and no alpha. |
| **Period** | 1990–2020 (30 years, many regimes) | 2018–2025, tested on 2022–2025 | Under four years of out-of-sample data against the papers' 20–25. |
| **Bar frequency** | Daily | Hourly | ~7× the observations at materially lower signal-to-noise per bar. |
| **Portfolio construction** | **Univariate TSMOM**: hold *every* asset, size each independently, equal-weight the sleeve — $R = \frac{1}{N}\sum_i z_i \frac{\sigma_{\text{tgt}}}{\sigma_i} r_i$ | **Cross-sectional**: rank by \|signal\|, hold the top 10, normalize gross exposure to 1.0 | A different strategy family, and one both papers explicitly distinguish themselves from. It is also the direct mechanical cause of the ~97% net-long book documented in the [Discussion](#discussion-reading-the-null). **A DMN-faithful construction — all 53 names, vol-scaled, equal-weighted, sized by the model's `tanh` output — is implemented nowhere in this repo and is the most obvious untried experiment.** |
| **Lookback** | Wood et al. find one **year** optimal for the Momentum Transformer (vs ~one quarter for a plain LSTM) | 252 hourly bars ≈ 36 trading days | Roughly 7× shorter in calendar time than the paper's tuned optimum, which plausibly forfeits the long-dependency advantage that motivates attention in the first place. |
| **Regime handling** | CPD covariates + Variable Selection Network | Two-arm blend over 21 regime descriptors | Substitution, not implementation — see above. |
| **Training loss** | Sharpe; DMN adds an optional turnover regularizer | Sharpe only | Costs never enter training. |
| **Validation protocol** | Expanding window, recalibrate every 5 years, 5-year out-of-sample blocks | Rolling **fixed** 2-year train / 1-quarter test / 1-quarter step | Shorter, more numerous, noisier windows — hence the ±3.2 standard deviation on window Sharpe. |
| **Overfitting control** | Wood et al. cite Bailey & López de Prado's deflated Sharpe paper *by name* (their reference [27]), note that inflated Sharpes "may need to be corrected" — and then report per-year out-of-sample Sharpe instead of computing the correction | Computes that exact statistic, against a counted 93 configurations | The one dimension on which this project is stricter than its sources. It is also what turns the headline from "underperformed the benchmark" into "consistent with no skill." |

One inheritance deserves emphasis, because it reframes this repository's central
negative finding as a confirmation rather than a surprise: **cost fragility is a
documented property of the DMN family, reported in both source papers.** Lim et al.
find their Sharpe-optimized LSTM outperforms benchmarks only "up to 2-3 basis points."
Wood et al.'s Exhibit 10 is starker — their best architecture's portfolio Sharpe falls
from 2.00 at zero cost to −0.35 at 3 bps, and the plain LSTM from 0.82 to −1.05. The
cost units are not directly comparable to this project's (they charge on
volatility-scaled position changes; this repo charges on portfolio-weight turnover), so
the thresholds should not be read against each other. But the qualitative finding is
the same one, arrived at independently: **the edge in this family of models lives
inside the spread.** What this project adds is the measurement on hourly US equities,
where realistic spreads are wide enough to consume it entirely.

---

## Repository Layout

```
Models/
  config.py              Calendar constants + all hyperparameter dataclasses
  LSTM.py                Shared LSTM filter; attention path is a toggle
  Transformer_layers.py  Positional encoding, interpretable MHA, encoder stack
  Momentum_transformer.py  LSTM filter -> Transformer -> tanh position head
  Ensemble_model.py      Regime feature extractor + blend network + ensemble
Utils/
  Market_data.py         Loading, split adjustment, trading calendar, bad ticks
  Feature_engineering.py Stock-agnostic features, normalizer, quality checks
  Losses.py              Sharpe / Sortino / Calmar / drawdown / turnover objectives
  Training.py            Training loop, early stopping, sequential sampler
  Portfolio.py           Weight construction, rebalancing, costs, baselines
  Backtesting.py         Walk-forward analyzer
  Metrics.py             Single source of truth for every ratio and statistic
  Regime_detector.py     Descriptive regime labelling for PnL attribution
data/Dataset.py          Contiguity-checked rolling-window datasets
Examples/run_backtest.py Canonical end-to-end entry point
tests/                   74 test cases across causality, features, metrics, model, portfolio
outputs/                 Committed run summaries, reports and resolved configs
```

---

## Data Pipeline

**Source.** Hourly OHLCV bars for 100 US equities and ETFs, 2018-05-01 → 2025-11-26
(`OHLCV-1HR/OHLCV.csv`, ~951k rows).

**Trading calendar.** Regular hours are *inferred from bar density* rather than
hardcoded: any UTC hour carrying at least half the busiest hour's bar count is treated
as regular hours. For this feed that resolves to **13:00–19:00 UTC, 7 bars per
session**, yielding **13,278 calendar bars over 1,906 sessions** (6.97 bars/session).
Every symbol is reindexed onto this grid, so `n_periods` means the same thing for
every series.

This matters for annualization. The repo's constant `PERIODS_PER_YEAR` is
$7 \times 252 = 1764$, but the *empirically derived* value from the actual timestamp
index is **1,753** bars/year, and that is what the reported metrics use. The headline Sharpe additionally sidesteps the question entirely
by compounding PnL to daily and annualizing with $\sqrt{252}$.

**Split adjustment before filtering.** An hourly bar that moves more than 2× in either
direction is far more likely a corporate action than a return. Detected ratios are
snapped to the nearest simple rational (a true 2:1 split on a bar that also moved 0.3%
reads as 1.994, not 2.000) and prices are back-adjusted:

$$\text{adjusted}[t] = \frac{\text{raw}[t]}{\prod_{i \,:\, t_i > t} f_i}$$

This keeps the return series continuous across the split instead of punching a hole in
it, so window-spanning features stay valid. **31 symbols** carried detected splits.

**Universe filters.** Applied *after* split adjustment, so a real split is never
mistaken for an implausible return:

| Stage | Rule | Rationale |
|---|---|---|
| Price | min close ≥ 2 USD | Tick-size artifacts dominate sub-2-dollar returns |
| Leverage | drop 44 known leveraged/inverse ETFs | Volatility decay and path dependence break the "accurate prediction ⇒ proportional profit" assumption |
| Returns | drop symbols with any \|r\| > 1000% | Residual data pathology |
| History | ≥ 3,000 bars | Enough history for the 882-bar feature burn-in plus a 252-bar model window |

**100 raw symbols → 53 tradeable.** The benchmark (SPY) is loaded **outside** these
filters by `load_market_series`, which raises if it is absent. This is deliberate: a
filter silently dropping SPY would remove eight market-context feature columns and
change `input_dim` with no error.

**Causal bad-tick filter.** Bars where $|r_t| > k \cdot \hat\sigma_{t-1}$ (with
$k = 12$ and $\hat\sigma$ a 147-bar trailing standard deviation, shifted by one bar)
are dropped at the *data* layer. **282 bars removed.** Realized PnL is never
winsorized — that would be marking the strategy's own losses to a rosier price.

**Data splits.** Chronological, inclusive upper bounds:

| Split | Range | Feature rows | Usable windows | Rejected (non-contiguous) |
|---|---|---|---|---|
| Train | ≤ 2020-12-31 | 142,573 | 114,694 | 17,086 |
| Validation | 2021 | 77,313 | 60,582 | 5,524 |
| Test | > 2021-12-31 | 354,497 | 299,255 | 41,939 |

Total feature rows: 574,383 across 54 columns. The first test prediction lands on
2022-02-28 rather than 2022-01-01 because a 252-bar contiguous window must close
before a prediction exists.

---

## Architecture

Three variants share one code path. `LSTMMomentumEncoder` is a single class with
`use_attention_path` toggled, and **both arms use an identical prediction head** — the
earlier design gave the attention arm two extra layers, which confounded "attention
helps" with "more layers help." The ablation below is therefore a clean one.

### Common backbone: local filter, global attention

The input is a $[B, 252, 54]$ window — 36 trading days of hourly bars. The LSTM filter
runs in **strided** mode: the sequence is split into four consecutive 63-bar blocks
(9 trading days each), every block is encoded independently by the shared LSTM, and
the final hidden state of each block becomes one embedding. The Transformer then
attends across those four local summaries.

$$
\underbrace{[B, 252, 54]}_{\text{raw window}}
\;\xrightarrow{\text{LSTM per 63-bar block}}\;
\underbrace{[B, 4, 64]}_{\text{local embeddings}}
\;\xrightarrow{\text{causal Transformer}}\;
\underbrace{[B, 4, 64]}_{\text{contextualized}}
\;\xrightarrow{\text{head}}\;
\underbrace{[B]}_{\text{position}}
$$

This is what makes the name literal — a lightweight local filter feeding global
attention. Two alternatives are implemented for comparison: `full` (LSTM across the
whole sequence, closest to Wood et al., ~4× the LSTM compute) and `truncate` (keep only
the last 63 bars — legacy behavior, retained so the regression can be measured rather
than asserted).

**The head ends in `tanh`, and the output *is* the position.** This is not cosmetic.
The Sharpe objective is scale-invariant — $\text{Sharpe}(c \cdot \text{pnl}) =
\text{Sharpe}(\text{pnl})$ — so without a bounded activation the position *magnitude*
is completely unidentified by training: the output scale becomes an accident of
initialization, any turnover penalty is meaningless, and downstream code ends up
patching it with signal thresholds and standard-deviation rescaling.

**Volatility targeting** then converts the bounded signal into a sized position:

$$w_t = \tanh(z_t) \cdot \min\!\left(\frac{\sigma_{\text{target}}}{\hat\sigma_t + \epsilon},\; L_{\max}\right)$$

with $\sigma_{\text{target}} = 15\%$ annualized, $L_{\max} = 3$, and $\hat\sigma_t$ an
ex-ante forecast built only from information strictly before bar $t$. The cap exists so
a near-zero volatility estimate cannot produce unbounded leverage.

### Transformer stage

Two encoder blocks, four heads, $d_{\text{model}} = 64$, feed-forward width 256,
dropout 0.2, sinusoidal positional encoding, **causal mask applied**.

The mask convention is unified on PyTorch's: `True` means *disallowed*. Attention
within the window was previously bidirectional — with only the final timestep read that
was not lookahead, but it deviated from the paper and would become real leakage the
moment per-timestep targets are introduced.

Standard scaled dot-product attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V,
\qquad M_{ij} = \begin{cases} -\infty & j > i \\ 0 & \text{otherwise} \end{cases}$$

The blocks use `InterpretableMultiheadAttention`, the TFT construction from Lim et al.
(1912.09363): queries and keys are projected per head, but **one** value projection of
width $d_k$ is shared across all heads and the per-head outputs are averaged:

$$\tilde{H} = \frac{1}{h}\sum_{m=1}^{h} \text{softmax}\!\left(\frac{Q_m K_m^\top}{\sqrt{d_k}} + M\right) V_{\text{shared}}$$

Because every head writes into the same value space, the head-averaged attention matrix
is a meaningful "how much did position $i$ use position $j$" map. That is the whole
point of the construction — an ordinary per-head value projection reshaped across heads
is just multi-head attention with an interpretability claim attached.

### Variant 1: Vanilla filter (`MomentumTransformerSimple`)

`use_attention_path = False`. The LSTM encodes each block; the final hidden state is the
block embedding. **156,194 parameters.**

### Variant 2: Attention-enhanced filter (`MomentumTransformerDualPath`)

Adds a lightweight additive-attention path over each local window, gated against the
LSTM's own output. For hidden states $h_1, \dots, h_W$ in a 63-bar block:

$$e_i = v^\top \tanh(W_a h_i), \qquad \alpha_i = \frac{\exp(e_i)}{\sum_{j=1}^{W}\exp(e_j)}, \qquad c = \sum_{i=1}^{W} \alpha_i h_i$$

$$g_i = \sigma(W_g h_i), \qquad \tilde{h}_i = \text{LayerNorm}\big((1 - g_i)\,h_i + g_i\, c\big)$$

The gate is the mechanism that matters. It lets the model choose, per timestep, how much
to trust the pooled context against the sequential summary: when $g \to 0$ the block
degenerates to the vanilla filter, and when $g \to 1$ it reads entirely from the
attention-weighted context. The intended behavior is that the gate opens during
volatility spikes and volume surges, where a specific recent event carries more
information than a running average, and closes during quiet trending markets.

The bottleneck is deliberately narrow — $W_a \in \mathbb{R}^{16 \times 64}$ — so the
whole path costs **1,250 parameters**: 1,040 for $W_a$, 17 for $v$, 65 for the gate,
128 for the extra LayerNorm. **157,444 parameters total, +0.8% over vanilla.** Any
performance difference between the two arms is therefore attributable to the mechanism,
not to capacity.

### Variant 3: Regime ensemble (`EnsembleMomentumTransformer`)

$$\hat{y}_t = \big(1 - w(r_t)\big)\cdot \hat{y}_t^{\text{vanilla}} + w(r_t)\cdot \hat{y}_t^{\text{attention}}$$

where $r_t \in \mathbb{R}^{21}$ is a regime descriptor extracted from the same input
window, and $w(\cdot)$ is a small MLP constrained to $[0.2, 0.8]$ so neither arm can be
switched off entirely.

**The 21 regime descriptors**, extracted batched and differentiably from the feature
window itself (no side channel, no second data source):

| Group | Features |
|---|---|
| Volatility state (5) | one-hot high/medium/low vol from **training-fitted terciles**, $\sigma_{21}/\sigma_{63}$ ratio, mean trailing vol |
| Distribution shape (5) | mean return, mean absolute return, range, skewness, 21-bar cumulative momentum |
| Trend (1) | signed lag-1 autocorrelation of returns |
| Named passthrough (2) | current RSI, current momentum feature |
| Cross-sectional (5) | return vs market, vol vs market, beta, relative strength, market vol level |
| Temporal (3) | vol persistence (std of sub-window vols), vol trend, regime-shift magnitude |

Two design details worth naming. First, the volatility terciles are **fitted on the
training split** (`fit_thresholds`) and registered as buffers, not hardcoded at
"3% is high" — hourly equity volatility does not live on the same scale as daily.
Second, the weight network opens with a `LayerNorm` on its input, because the
descriptors span wildly different scales (RSI near 50 sitting next to mean returns near
$10^{-4}$) and without it a single large value saturates the sigmoid and pins the blend
weight for that sample.

**Parameters:** 156,194 (vanilla arm) + 157,444 (attention arm) + 1,291 (weight network)
= **314,929 total**. The `run_config.json` records 1,291 because `count_parameters`
counts *trainable* parameters and the arms are frozen by the time it runs.

---

## Feature Engineering

54 features per bar, all dimensionless or explicitly normalized, all declared in
**trading days** and converted to bars via `bars(days) = round(days × 7)` so the
horizons stay auditable.

**The causality rule, stated once and enforced everywhere:** every raw input is shifted
before any feature is computed.

```python
close = df["close"].shift(1)   # at time t we may only use data through t-1
```

The target at row $t$ is $r_t = \text{close}[t]/\text{close}[t-1] - 1$, which depends on
`close[t]` — a price no feature at row $t$ has seen. `tests/test_causality.py` asserts
this by perturbing `close[t]` and requiring that no feature at any row $\le t$ moves,
and includes a deliberately leaky feature to prove the test can fail.

| Block | Count | Contents |
|---|---|---|
| Returns | 6 | 1, 7, 35, 147, 441, 882 bars (≈1 hour to 6 months) |
| Intrabar ratios | 3 | high-low range, open-close, close position in range |
| MA ratios | 8 | $(P - \text{MA}_w)/\text{MA}_w$ for 7 windows, plus fast-vs-mid crossover |
| Volatility | 5 | rolling std at 4 horizons, plus short/long ratio |
| Risk-adjusted momentum | 4 | $r_{t-w,t} \,/\, (\hat\sigma_{\text{ewm}}\sqrt{w})$ |
| Technical | 6 | Wilder RSI, MACD triplet (price-normalized), Bollinger position and width |
| Volume | 3 | ratio to own MA, momentum, z-score |
| Higher moments | 4 | skewness and kurtosis at 21 and 63 bars |
| Autocorrelation | 3 | lags 1, 5, 21 |
| Z-scores & percentiles | 4 | price/return z-score, price/return rolling percentile |
| ATR | 1 | true range as a fraction of price |
| Market context | 5 | SPY returns at 3 horizons, market vol, **beta** |
| Relative strength | 2 | stock minus market return at 2 horizons |

**Design decisions that survived audit:**

- **Momentum is risk-adjusted, not a duplicate of returns.** $r_{t-w,t}$ divided by
  $\hat\sigma\sqrt{w}$ — the Deep Momentum Networks normalization. Without it,
  `momentum_35` and `return_35` are literally the same column; a test asserts they are
  not.
- **Beta is $\text{cov}/\text{var}$, not correlation.** Different quantity, different
  economic meaning.
- **The MA crossover uses a mid window, not the longest.** Crossing the longest window
  duplicates `ma_ratio_882` at measured $\rho = 0.996$.
- **`outperformance_ratio` was dropped.** For bar-scale returns
  $(1+s)/(1+m) - 1$ is a first-order Taylor expansion of $s - m$; measured $\rho$
  against `relative_strength_35` was 0.9997.
- **RSI stays on its native 0–100 scale in the feature frame** and is put on equal
  footing by the normalizer. A separately z-scored copy alongside the raw value was
  removed as a duplicate.
- **Market features are reindexed explicitly, and non-overlapping timestamps become
  NaN and are dropped** with the burn-in rows. Filling them with zeros would read as a
  genuine observation of "market return exactly 0, beta exactly 0."
- **Burn-in rows are dropped, not imputed.** The first 882 rows per symbol (the longest
  lookback) are removed rather than filled with a semantically loaded zero.

**Normalization.** `FeatureNormalizer` fits mean and standard deviation on the
**training split only**, persists to `feature_normalizer.json`, and applies unchanged
to validation and test. Standardization happens *before* clipping at ±5, so the bound
really is in standard deviations. Bounded features (`bb_position`, `close_position`,
`price_percentile`, `return_percentile`) pass through untouched.

**Quality gates.** Every run checks for feature pairs with $|\rho| > 0.99$ and reports
the spread of per-feature standard deviations. All three committed runs report
**zero collinear pairs**.

---

## Training Methodology

**Objective.** Negative annualized Sharpe of the position-weighted PnL:

$$\mathcal{L} = -\frac{\mathbb{E}[w \odot r]}{\text{std}(w \odot r) + \epsilon}\sqrt{P}$$

Alternatives are implemented and registered (`Sortino`, `Calmar`, `MaximumDrawdown`,
`NormalizedDrawdown`, `SharpeWithTurnoverPenalty`, `SoftDirectional`,
`InformationRatio`, `Combined`). Path-dependent objectives inherit from
`_PathDependentLoss` and **refuse to run** unless constructed with `sequential=True`
and fed by `SequentialBlockSampler` — a `cummax` over a shuffled batch of unrelated
timestamps is not a drawdown, and silently computing one is worse than crashing.

**Sharpe is estimated at the epoch level, not per batch.** A per-minibatch Sharpe is a
ratio of two noisy estimates whose mean across batches is not the Sharpe of anything.
Predictions and returns are accumulated across 16 minibatches *keeping the autograd
graph*, and a single objective is backpropagated over the pooled sample. At
`batch_size = 128` that is 2,048 observations per estimate instead of 128, cutting the
ratio's standard error by ~4×. Whole-epoch pooling is the ideal and is supported
(`--accumulation-steps 0`), but on this dataset it does not fit in memory on CPU.

**Optimization.** AdamW, lr $10^{-3}$, weight decay $10^{-5}$, gradient clipping at
norm 1.0, `ReduceLROnPlateau` on validation Sharpe (patience 5, factor 0.5), early
stopping patience 15, up to 100 epochs.

**Model selection uses the backtest, not a proxy.** `validation_fn` builds the actual
10-name portfolio at every validation timestamp, applies the weekly rebalance schedule
and the cost model, and returns the Sharpe of the resulting net return series. Early
stopping and checkpoint selection use *that* number, and they use the same number as
each other. A proxy validation metric that scores a different strategy than the one you
run selects a checkpoint for a strategy that was never traded.

**Ensemble schedule — genuinely two-stage:**

1. **Stage 1a/1b.** Vanilla and attention arms trained independently to convergence on
   the training split, each with its own early stopping.
2. **Stage 2.** Both arms frozen (`freeze_submodels()` sets `requires_grad = False` and
   puts them in `eval()`), then the 1,291-parameter blend network is fitted for 10
   epochs with an optimizer constructed over `weight_network.parameters()` only.

Freezing is what makes Stage 2 interpretable: any change in performance is attributable
to the blend, because nothing else can move.

**Runtimes** (CPU, single run): vanilla 39 min, attention 44 min, ensemble 2 h 13 min,
walk-forward ensemble considerably longer (22 windows × full retrain).

---

## Evaluation Protocol

This section is the reason the null above is worth reading.

**Portfolio construction.** At each weekly rebalance, take the cross-section of
predictions available at that timestamp, select the top 10 by |signal|, normalize gross
exposure to 1.0, cap any single weight at 0.2, and hold between rebalances. Weights are
held rather than renormalized on bars where only some names print — renormalizing would
manufacture exposure out of a missing quote. A hard assertion fires if realized gross
exposure ever exceeds the cap.

**Cost model.** Charged on realized turnover, $\sum_i |w_{i,t} - w_{i,t-1}| \cdot c$,
with the first bar counted as entering from flat. Live rate $c = 6$ bps one-way
(5 bps half-spread + 1 bp fee). Scenarios at 5, 10 and 20 bps are reported alongside.
Market impact is modeled as zero — see [Limitations](#limitations-and-threats-to-validity).

**Metric conventions.** `Utils/Metrics.py` is the only place any ratio is defined.

- Annualized return is a **true CAGR** compounded over the realized horizon, never
  $\text{mean} \times P$.
- Sortino uses downside deviation about the target,
  $\sqrt{\mathbb{E}[\min(r - \tau, 0)^2]}$, over the **full** sample — not the standard
  deviation of the losing subset about its own mean, which is a different and flattering
  quantity.
- Max drawdown is **not floored**: a strategy that loses more than 100% reports that it
  did.
- The **headline Sharpe aggregates PnL to daily and annualizes with $\sqrt{252}$**,
  removing all dependence on the assumed bars-per-day. The per-bar figure is reported
  beside it, never instead of it.

**Statistical honesty.** Three separate corrections, because the naive versions are all
badly optimistic here:

1. **Cross-sectional IC.** Rank-correlate predictions against realized returns *within*
   each timestamp, then test whether that time series of ICs has a nonzero mean. This
   sidesteps the overlapping-window problem that makes the pooled p-value meaningless.
2. **Moving-block bootstrap** (block size 252, 500 resamples) for the pooled IC's
   standard error and CI, preserving the serial dependence the analytic Spearman
   p-value ignores.
3. **Deflated Sharpe Ratio** (Bailey & López de Prado, 2014), adjusting for both the
   number of configurations searched and the non-normality of the return distribution:

$$\widehat{\text{DSR}} = \Phi\!\left(\frac{\hat{\text{SR}} - \text{SR}_{\max}}{\hat\sigma_{\text{SR}}}\right), \qquad \hat\sigma_{\text{SR}} = \sqrt{\frac{1 - \gamma_3\hat{\text{SR}} + \frac{\gamma_4 - 1}{4}\hat{\text{SR}}^2}{n - 1}}$$

The reports state plainly, in the artifact itself, that the nominal pooled IC p-value
"assumes i.i.d observations. These are overlapping windows across correlated names, so
it is badly optimistic."

**Baselines.** SPY buy-and-hold (derived from the market frame, so it is always
available even when SPY is not in the tradeable universe), equal-weight long-only
basket of the 53 survivors, and a naive time-series momentum rule (long/short the sign
of trailing 21-day returns, same top-10 selection and weekly schedule). Everything is
additionally reported **vol-matched to SPY's realized 17.36%**, so no comparison turns
on leverage.

---

## Results

### Performance decomposition

| | Vanilla | Attention | Ensemble |
|---|---|---|---|
| Gross Sharpe (daily agg.) | 0.516 | 0.692 | 0.709 |
| Net Sharpe (daily agg.) | 0.328 | 0.513 | 0.507 |
| **Cost drag on Sharpe** | **−0.188** | **−0.179** | **−0.202** |
| Gross total return | 33.09% | 47.71% | 46.42% |
| Net total return | 17.31% | 31.63% | 29.49% |
| Total cost drag | 0.126 | 0.115 | 0.123 |
| Sortino | 0.482 | 0.758 | 0.735 |
| Calmar | 0.185 | 0.335 | 0.283 |
| Win rate | 51.33% | 51.99% | 51.88% |

Costs consume ~0.19 of Sharpe and 16–17 percentage points of cumulative return in
every variant. (The "total cost drag" row is the undiscounted sum of per-bar costs,
~0.12 of notional; the effect on cumulative return is larger because the drag compounds
inside the equity curve.) Note that even the **gross** ensemble Sharpe (0.709) sits
below SPY's net 0.783. The strategy does not lose to the benchmark because of trading frictions; it
loses before they are charged, and then loses by more.

### Transaction-cost sensitivity

| One-way cost | Vanilla Sharpe / return | Attention Sharpe / return | Ensemble Sharpe / return |
|---|---|---|---|
| 5 bps | 0.374 / 19.81% | 0.563 / 34.18% | 0.561 / 32.17% |
| 6 bps (live) | 0.340 / 17.31% | 0.532 / 31.63% | 0.526 / 29.49% |
| 10 bps | 0.204 / 7.85% | 0.405 / 21.88% | 0.383 / 19.30% |
| 20 bps | **−0.135 / −12.63%** | **0.090 / 0.55%** | **0.027 / −2.82%** |

*(per-bar annualized Sharpe, for comparability with the scenario block in the raw
summaries)*

Total turnover is 204.83 against a gross book of 1.0, spread over 205 weekly
rebalances — roughly **1.0 of gross turnover per rebalance**, or about half the book
replaced each week, ~55× gross notional per year. That is not a low-turnover strategy,
and at 6 bps it costs ~17 percentage points of cumulative return and ~0.19 of Sharpe.

But cost is not the binding constraint. Even at **zero** cost the ensemble's gross
Sharpe of 0.709 loses to SPY's net 0.783. Reducing turnover would narrow the gap; it
would not close it.

### Benchmark comparison, vol-matched to SPY (17.36%)

| Series | Sharpe | CAGR | Max DD |
|---|---|---|---|
| Ensemble | 0.526 | 7.92% | −28.11% |
| SPY | **0.783** | **12.85%** | **−23.81%** |
| Equal-weight | 0.769 | 12.57% | −25.73% |
| Naive TSMOM | −0.063 | −2.57% | −32.12% |

Matched on volatility, the ensemble delivers ~62% of SPY's return with a deeper
drawdown. The one comparison it wins is against naive time-series momentum, which is
strongly negative over this window — a genuine but limited finding: **the model learned
something better than raw trend-following, and worse than not trading at all.**

### Factor exposure

| | Vanilla | Attention | Ensemble |
|---|---|---|---|
| Annualized alpha vs SPY | −4.73% | −1.78% | **−0.45%** |
| Alpha t-statistic | −0.93 | −0.37 | **−0.08** |
| Beta | 0.760 | 0.765 | 0.625 |
| Correlation | 0.801 | 0.819 | 0.709 |
| $R^2$ | 0.642 | 0.671 | 0.503 |
| Mean net exposure | 0.998 | 1.000 | 0.970 |

Alpha is negative in all three variants and statistically indistinguishable from zero
in all three. Half the ensemble's variance is explained by SPY alone. The ensemble's
lower beta and $R^2$ are the one structural improvement over the single-arm models —
it is marginally less of a pure index proxy — but that buys it nothing in Sharpe.

### Predictive power

| | Vanilla | Attention | Ensemble |
|---|---|---|---|
| Observations | 299,255 | 299,255 | 299,255 |
| Pooled IC | 0.0023 | 0.0015 | **0.0071** |
| Block-bootstrap 95% CI | [−0.0014, 0.0065] | [−0.0023, 0.0054] | **[0.0035, 0.0109]** |
| Mean cross-sectional IC | 0.0016 | 0.0028 | 0.0043 |
| IC t-statistic | 0.56 | 0.93 | 1.66 |
| IC p-value | 0.574 | 0.352 | 0.096 |
| Timestamps | 6,614 | 6,614 | 6,614 |
| Directional accuracy (raw) | 46.79% | 46.75% | 46.79% |
| Directional accuracy (demeaned) | 51.18% | 51.30% | 51.11% |

Read carefully:

- **Only the ensemble's pooled IC has a bootstrap CI excluding zero.** That is the
  single strongest positive result in this repository, and it is an IC of 0.007.
- **The cross-sectional t-statistic — the more honest of the two — is 1.66, p = 0.096.**
  Not significant at any conventional threshold.
- **Raw directional accuracy below 50% is not a bug.** Predictions are almost always
  positive (mean 0.292, median 0.244) while the median bar return is exactly 0.000. The
  demeaned figure, 51.1%, is the interpretable one: the model has a slight edge on the
  *cross-sectional ranking*, essentially none on the raw sign.

An IC of 0.004–0.007 is not absurd for hourly horizons — published cross-sectional
equity work lives in this neighborhood. The problem is not that the IC is small; it is
that the portfolio built on top of it does not convert the IC into return net of the
spread.

### Multiple-testing adjustment

| | Vanilla | Attention | Ensemble |
|---|---|---|---|
| Configurations counted | 93 | 93 | 93 |
| Realized Sharpe (per-bar) | 0.340 | 0.532 | 0.526 |
| Expected max Sharpe under null | 1.291 | 1.290 | 1.295 |
| Sharpe standard error | 0.515 | 0.515 | 0.517 |
| Return skew | 0.162 | 0.250 | −0.338 |
| Excess kurtosis | 16.64 | 15.93 | 9.37 |
| **Deflated Sharpe, P(true SR > 0)** | **0.033** | **0.070** | **0.068** |

The expected maximum Sharpe under the null — what the *luckiest* of 93 skill-free
configurations would be expected to show — is roughly 2.5× the best Sharpe actually
observed, and 3.8× the vanilla model's.
Combined with heavy tails (excess kurtosis 9–17, which inflates the Sharpe estimator's
variance), the probability that any variant has positive true Sharpe is under 7%.

This is the number that closes the question. Everything else is a description of *how*
the strategy fails to beat its benchmark; this is the statement that the observed
performance is consistent with no skill at all.

### Ensemble blend-weight diagnostics

Recorded on the test split only, 100,000 sampled observations:

| Statistic | Value |
|---|---|
| Mean attention weight | 0.776 |
| Std attention weight | 0.028 |
| Median | 0.786 |
| Min / Max | 0.649 / 0.797 |
| Bars favoring attention (w > 0.5) | **100.0%** |
| Bars favoring vanilla (w ≤ 0.5) | **0.0%** |

**The regime-switching hypothesis is not supported.** The weight network converged to a
near-constant 0.78, pressed against its own 0.8 upper bound, with a standard deviation
of 0.028 across nearly four years spanning the 2022 bear market, the 2023
regional-banking crisis, the 2024 rate cycle and 2025. If regime information were driving the blend, this
distribution would be bimodal or at minimum wide. It is neither.

The honest reading is that Stage 2 learned a single global fact — "the attention arm
validates better" — and encoded it as a constant tilt. The $[0.2, 0.8]$ clamp, which
exists to preserve ensemble diversity, is the only thing keeping the vanilla arm in the
prediction at all. And the outcome is exactly what a constant blend predicts: the
ensemble (0.507) tracks the attention arm (0.513) almost exactly, marginally worse for
the 22% of vanilla it is forced to carry.

### Walk-forward analysis

Rolling-origin with a **fixed 2-year training window** (`start += step` each
iteration, so old data leaves the window rather than accumulating), 1-quarter test,
1-quarter step. 22 windows spanning **2020-05-14 → 2025-11-18**. Each window is fully
retrained, with its own normalizer fitted on that window's training data only, and the
last 15% of each training window held out for early stopping so nothing in the test
window can influence the fitted model.

| Statistic | Value |
|---|---|
| Windows | 22 |
| Positive-Sharpe windows | **12 / 22 (54.5%)** |
| Mean window Sharpe | 0.546 |
| Median window Sharpe | 0.511 |
| Std of window Sharpe | 3.225 |
| Min / Max | −4.955 / 6.101 |
| **95% CI on mean Sharpe** | **[−0.80, 1.89]** |
| Mean window return | 0.74% |
| Sum of window returns | 16.36% |
| Mean turnover | 0.045 |

Individual quarters swing from −4.96 to +6.10 Sharpe. Quarterly Sharpe estimates over
441 bars are inherently noisy, which is precisely why the confidence interval is the
statistic to read — and it straddles zero comfortably. A strategy with a real edge
should not need 22 windows to produce a mean of 0.55 with a standard deviation of 3.2.

> **Artifact caveat, stated plainly.** This walk-forward run completed all 22 windows
> but crashed while printing the final pooled summary, so `walk_forward.json` and the
> pooled out-of-sample return series were never written. The window-level metrics above
> were salvaged from terminal output and are stored in
> `outputs/run_2026-08-23_2302_ensemble_walk_forward/salvaged_walk_forward.json`, which
> records its own provenance and lists exactly what is missing (pooled Sharpe, pooled
> drawdown, pooled CAGR, pooled total return). Treat this as a **robustness check, not a
> headline** — the summary statistics are computed from values rounded to three decimals
> in the terminal, and the chained total return (13.86%) is an approximation.

---

## Discussion: Reading the Null

**Why does the attention path help while the ensemble does not?**

The attention arm beats the vanilla arm by 0.185 of daily Sharpe at a 0.8% capacity
premium. That is a real, if small, architectural result: pooled context gated against
the sequential summary generalizes better than the sequential summary alone. It should
still be held loosely: the deflated Sharpe puts the attention arm's *own* Sharpe well
inside the range 93 skill-free configurations would produce, so a 0.185 gap between two
such numbers carries little weight — and with one seed per architecture, the gap has not
been measured against its own sampling variance.

The ensemble then fails for a specific, diagnosable reason. Stage 2 optimizes the same
Sharpe objective on a validation split where the attention arm dominates uniformly. The
gradient signal available to the weight network is therefore "push $w$ up," and it does
— to the clamp. Learning a *conditional* policy requires the two arms to have
complementary failure modes that are (a) real and (b) predictable from the 21
descriptors. Neither condition holds here: the arms are 99% shared architecture trained
on the same data with the same objective, and their errors are correspondingly
correlated. There is no regime in which vanilla is meaningfully better, so there is
nothing for a regime detector to detect.

**Why is the book long-only when the design is long/short?**

`construct_weights` selects the top 10 names by |signal| and normalizes gross exposure
to 1.0. Nothing forces balance between the sides. Because the `tanh` head emits a
predominantly positive cross-section (mean 0.292 against a target mean of 0.0001), the
top 10 by magnitude are almost always all longs, giving mean net exposure of 0.970. The
resulting book is a concentrated 10-name long portfolio with 0.625 beta — which is why
it correlates 0.71 with SPY and posts negative alpha. **A dollar-neutral construction
would require ranking and taking both tails explicitly, not ranking on magnitude.** This
is the single most consequential design gap between the stated intent and the
implementation, and it is a concrete next experiment rather than a fatal flaw.

**Is the signal real at all?**

Marginally. The ensemble's bootstrap IC CI excludes zero, and demeaned directional
accuracy is 51.1% on 299,255 observations. That is a faint but non-zero cross-sectional
ranking ability. What the results establish is that this signal is **too weak to survive
the round trip**: 0.007 IC, converted through a top-10 equal-gross portfolio with weekly
rebalancing, yields 46.4% gross return over 3.75 years, of which ~17 percentage points
go to the spread and the remaining 29.5% underperforms buying the index.

**Is the negative result specific to this window?**

Partly, and it is worth being explicit about the direction of the bias. The test period
2022-02 → 2025-11 contains a severe bear market followed by a strong, narrow,
mega-cap-led recovery. That regime is unusually punishing for cross-sectional momentum
and unusually kind to cap-weighted buy-and-hold, which is exactly the comparison being
lost. The walk-forward extends coverage back to 2020-05 and finds 12/22 positive
windows with a CI straddling zero — so the window is not *creating* the null, but a
different regime would likely narrow the gap to SPY.

---

## Limitations and Threats to Validity

Stated in decreasing order of how much they should worry a reader.

1. **The DSR trial count is a stated assumption, not a log.** `--n-trials 93` documents
   "30 tuning trials × 3 architectures + 3 seeds." The true search was not
   instrumented. If the real count is higher, the deflated Sharpe falls further; if
   lower, it rises. The conclusion is not sensitive over any plausible range: even at
   $n = 10$ the expected maximum Sharpe under the null is **0.814** against a realized
   0.526, and at $n = 30$ it is 1.072.
2. **Market impact is modeled as zero.** `BacktestConfig.impact = 0.0` and the field is
   reserved but unimplemented. For a 10-name book this is defensible at small AUM and
   indefensible at scale. Any capacity claim would need this filled in.
3. **No borrow cost or short-availability constraint.** In practice moot — the book is
   ~97% net long — but it would bind the moment the dollar-neutral construction
   discussed above is implemented.
4. **The universe is defined by what is in the CSV.** 100 symbols reduced to 53 by the
   filters. There is no delisting reconstruction, so survivorship bias runs in the
   optimistic direction; the file also over-represents recent high-volatility listings.
   The `min_bars ≥ 3000` filter removes most late listings, which mitigates but does
   not eliminate this.
5. **Single seed per architecture.** Every committed run uses `seed = 123`. The
   variance across seeds is unmeasured, and for a Sharpe with a standard error of ~0.52
   that variance is likely comparable to the vanilla-vs-attention gap being discussed.
6. **The walk-forward artifact is incomplete.** See the caveat above. Pooled
   out-of-sample statistics were never written.
7. **Hourly bars carry microstructure noise the cost model only approximates.** A flat
   6 bps one-way applies the same spread to AAPL and to a small-cap at 15:00 UTC on a
   volatile day. The 20 bps scenario exists precisely to bracket this, and the strategy
   does not survive it.
8. **Sharpe accumulation is 16 minibatches, not a full epoch.** The whole-epoch
   objective is the correct one and is supported by the code, but does not fit in
   memory on CPU at this dataset size. The pooled estimate over 2,048 observations is a
   compromise.

---

## Reproducing These Results

**Requirements.** Python 3.9+, and:

```
numpy>=1.24.0    pandas>=2.0.0    scipy>=1.10.0
torch>=2.0.0     matplotlib>=3.7.0    seaborn>=0.12.0    pytest>=7.0.0
```

**Setup.**

```bash
git clone https://github.com/RayyanWaseem1/Lightweight-Attention-LSTM-Filter-in-Momentum-Transformer.git
cd Lightweight-Attention-LSTM-Filter-in-Momentum-Transformer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Run.** Everything routes through one entry point.

```bash
# The three committed runs, in order
python Examples/run_backtest.py --model vanilla
python Examples/run_backtest.py --model attention
python Examples/run_backtest.py --model ensemble

# Walk-forward robustness check (22 windows, full retrain each)
python Examples/run_backtest.py --model ensemble --walk-forward

# Fast smoke test on a small universe
python Examples/run_backtest.py --model vanilla --max-symbols 8 --epochs 2
```

Useful flags: `--seed`, `--rebalance {daily,weekly,monthly}`, `--n-trials` (the DSR
trial count), `--accumulation-steps 0` (whole-epoch Sharpe), `--start` / `--end`,
`--output-dir`.

**Artifacts.** Each run writes to `outputs/run_<timestamp>_<model>/`:

| File | Contents |
|---|---|
| `performance_summary.txt` | The full human-readable report reproduced above |
| `report.json` | Every metric, machine-readable |
| `run_config.json` | **Resolved** config, data diagnostics, feature list, parameter count, runtime |
| `feature_normalizer.json` | Fitted normalization statistics |
| `portfolio_returns.csv` | Per-bar net return, gross return, turnover |
| `weights.csv`, `predictions.csv` | Full position and prediction history |
| `equity_curves.png`, `underwater_chart.png`, `rolling_sharpe.png`, `returns_distribution.png`, `regime_performance.png` | Diagnostic plots |

Every run writes its resolved configuration next to its outputs, so any number in this
README can be traced to the parameters that produced it. `.gitignore` keeps the heavy
artifacts (CSVs, PNGs) out of the repository and commits the summaries, reports and
configs. The four committed run directories are:

| Directory | Contents |
|---|---|
| `outputs/run_2026-08-23_2007_vanilla/` | `vanilla_performance_summary.txt`, `vanilla_report.json`, `run_config.json` |
| `outputs/run_2026-08-23_2107_attention/` | `attention_performance_summary.txt`, `attention_report.json`, `run_config.json` |
| `outputs/run_2026-08-23_1741_ensemble/` | `ensemble_performance_summary.txt`, `ensemble_report.json`, `run_config.json` |
| `outputs/run_2026-08-23_2302_ensemble_walk_forward/` | `salvaged_walk_forward.json` |

(The committed summaries and reports carry a model-name prefix; a fresh run writes them
as `performance_summary.txt` and `report.json`.)

**Determinism.** `set_seed` fixes Python, NumPy and Torch RNGs and sets
`cudnn.deterministic = True`. Runs are CPU by default (`TrainingConfig.device = "cpu"`).

---

## Test Suite

74 collected test cases (68 test functions, two of them parametrized) across five
modules. They are written as **falsifiable claims about the pipeline**, not smoke
tests.

| Module | Tests | What it pins down |
|---|---|---|
| `test_causality.py` | 9 | Perturbing `close[t]` moves no feature at row ≤ t; the target uses a price no feature has seen; **a deliberately leaky feature must break the test**; shuffled targets and time-reversed returns must collapse a planted edge; one extra bar of lag must degrade it gracefully, not catastrophically |
| `test_features.py` | 15 | Momentum is not a duplicate of returns; no pair exceeds \|ρ\| = 0.99; normalizer statistics come from train only; clipping happens after standardizing; burn-in rows are dropped not filled; splits are back-adjusted not deleted; features are comparable across wildly different price scales; regular hours are inferred from density; the bad-tick filter uses only trailing information |
| `test_metrics.py` | 13 | Sortino is annualized exactly once and uses full-sample downside deviation; max drawdown is not floored; annualized return compounds over the real horizon; **daily Sharpe is independent of the assumed bars-per-day**; wrong annualization inflates Sharpe by a known factor; deflated Sharpe falls as trials rise; alpha/beta recovers planted coefficients |
| `test_model.py` | 27 | Encoder modes produce the expected sequence lengths; strided mode uses the whole window; positions are bounded by `tanh`; vol targeting scales and caps; the causal mask matches PyTorch's convention and future positions cannot affect earlier outputs; interpretable attention shares one value head; **the ablation arms differ only by the attention path**; no regime feature is constant; the weight network normalizes its input; freezing leaves only the weight network trainable |
| `test_portfolio.py` | 10 | Rebalance schedules are derived from the calendar, not hardcoded; gross exposure normalizes; static predictions incur no ongoing cost; weights are held between rebalances; a missing bar earns zero rather than a renormalized full weight; net = gross − costs; **returns are not winsorized** |

```bash
pytest tests/ -v
```

---

## References

1. Wood, K., Giegerich, S., Roberts, S., Zohren, S. (2022). *Trading with the Momentum
   Transformer: An Intelligent and Interpretable Architecture.*
   [arXiv:2112.08534](https://arxiv.org/abs/2112.08534) — included as
   `2112.08534v3.pdf`. Source of the LSTM-filter-plus-Transformer design.
2. Lim, B., Zohren, S., Roberts, S. (2019). *Enhancing Time Series Momentum Strategies
   Using Deep Neural Networks.* [arXiv:1904.04912](https://arxiv.org/abs/1904.04912) —
   included as `1904.04912v3.pdf`. Source of the Sharpe-ratio objective and the
   volatility-scaled momentum normalization.
3. Lim, B., Arık, S. Ö., Loeff, N., Pfister, T. (2019). *Temporal Fusion Transformers
   for Interpretable Multi-horizon Time Series Forecasting.*
   [arXiv:1912.09363](https://arxiv.org/abs/1912.09363). Source of the shared-value-head
   interpretable attention.
4. Bailey, D. H., López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for
   Selection Bias, Backtest Overfitting and Non-Normality.* Journal of Portfolio
   Management, 40(5).
5. Vaswani, A., et al. (2017). *Attention Is All You Need.*
   [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).

---

## A Note on Scope

This is a research codebase, not a trading system. It contains no broker integration,
no order management, no live data path, and no position reconciliation — and given the
results above, adding them would be the wrong next step. The useful next experiments are
the ones this analysis points at directly: a genuinely dollar-neutral construction that
ranks and takes both tails, a seed sweep to size the vanilla-vs-attention gap against
its own variance, an implemented market-impact term, and a blend network trained on
splits where the two arms actually disagree.