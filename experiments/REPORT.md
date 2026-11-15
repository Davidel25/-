# Strategy-adjustment simulations: lower delta, 30% profit target, wider universe

**Read this first.** Everything below was produced on **synthetic** option chains (Black-Scholes prices on a simulated
underlying with an implied-vol premium *I chose*). It tells you how each rule behaves *mechanically* and how big the
effects are *if* a given premium exists. It does **not** tell you whether the premium exists in real markets, so none of
it is evidence of a real edge. Before changing live configuration, repeat the comparison on purchased option history
(`python -m src.backtest` / `src.backtest.compare.run_variants`).

## Method

* Harness: the real scanner, `RiskManager` limits and live exit rules (50% target, stop at a 2x-credit loss, close at
  DTE <= 14), $0.65 per contract per leg commission, **natural fills** (cross the spread on every leg, entry and exit).
* 900 trading days (about 3.6 years), **8 seeds**, one contract per signal, at most one position per symbol and direction.
* Every rule variant is run on the **identical price path** (same seed), so differences are *paired*; tables give
  mean +/- standard error over the 8 seeds and, where useful, in how many seeds the variant beat the baseline.
* "Premium" (VRP) = implied vol divided by realized vol minus 1. +10% on a 16-vol index is 1.6 vol points; +20% is 3.2
  points, roughly the long-run index average as I recall it (unverified here).
* Sharpe = annualised mean/std of the daily change in realized + open mark-to-market P&L, no risk-free subtraction.
  Short-premium strategies flatter Sharpe until a large loss arrives: always read it next to drawdown and the worst trade.
* Calibration: with **no friction and no premium** the simulator earns about zero (-0.43% +/- 0.37% of risk per trade),
  and the effect of the premium is monotone and roughly linear (+0.9% at +10%, +2.0% at +20%).

## Baseline ("current": short delta 0.20-0.25, 50% target), SPY-like, per trade as % of capital at risk

| Premium | Win rate | Stop-hit | Net / trade | Sharpe | Annual $ per 1-lot slot | Max drawdown |
|---|---|---|---|---|---|---|
| 0% | 71.8% | 17.5% | -2.54% | -1.40 | -$505 | $1,957 |
| +10% | 72.4% | 15.2% | -1.47% | -0.80 | -$275 | $1,245 |
| +20% | 72.1% | 13.2% | -0.61% | -0.35 | -$115 | $864 |

With natural fills and commissions the current configuration loses money in this world unless the premium is well above
+20%. Friction, not the choice of delta or exit, is what the next section shows to be the dominant effect.

## 1. Lower short delta (0.12-0.15 instead of 0.20-0.25)

| Premium | Variant | Win rate | Stop-hit | Avg win / avg loss | 5th pct trade | Net / trade | Sharpe | Max DD | Annual $/slot |
|---|---|---|---|---|---|---|---|---|---|
| 0% | 0.20-0.25 | 71.8% | 17.5% | $54 / -$173 | -$246 | -2.54% | -1.40 | $1,957 | -$505 |
| 0% | 0.12-0.15 | 76.1% | 19.1% | $31 / -$128 | -$158 | -1.51% | -1.12 | $1,545 | -$386 |
| +10% | 0.20-0.25 | 72.4% | 15.2% | $54 / -$164 | -$239 | -1.47% | -0.80 | $1,245 | -$275 |
| +10% | 0.12-0.15 | 77.6% | 16.3% | $31 / -$121 | -$157 | -0.63% | -0.41 | $959 | -$152 |
| +20% | 0.20-0.25 | 72.1% | 13.2% | $54 / -$149 | -$237 | -0.61% | -0.35 | $864 | -$115 |
| +20% | 0.12-0.15 | 78.9% | 13.5% | $32 / -$110 | -$146 | +0.36% | +0.29 | $527 | +$74 |

Paired difference, 0.12-0.15 minus 0.20-0.25: Sharpe **+0.29 +/- 0.10 (6/8 seeds), +0.39 +/- 0.07 (8/8), +0.64 +/- 0.16 (7/8)**;
net per trade +1.0, +0.8, +1.0 points of risk; win rate +4.3, +5.2, +6.9 points; stop-hit **+1.6, +1.1, +0.4 points (not lower)**.
The intermediate 0.15-0.20 band captured about a third of the benefit (Sharpe +0.09, +0.10, +0.34; not significant at low premium).

What changes, in order of confidence:
* **Smaller losses, not fewer stop-outs.** Average credit falls from 19.6% to 11.3% of width, wins shrink 43%, losses 26%, the
  5th-percentile trade improves 36% and drawdown 21-39%. The *frequency* of stops does not fall: the stop is a multiple of
  the credit (loss >= 2x credit), so it tightens as the credit shrinks. (My explanation; plausible, not isolated.)
* **Realised win rate is 76-79%, nowhere near the 85-88% implied by delta**, because "win" is net of costs and after stops
  and early exits. The break-even win rate also rises (76% to 80%) because the payoff ratio worsens.
* Part of the gain is **less friction per trade** (cheaper options cost less to cross), not extra edge: it appears even at 0% premium.
* **The current "disciplined" floors (credit >= 0.60, ROC >= 20%) produce zero trades at 0.12-0.15 delta.** Adopting this
  band requires lowering them together (about credit >= 0.45 and ROC >= 12% on 5-wide SPY spreads).

## 2. 30% profit target instead of 50%

| Premium | Target | Win rate | Stop-hit | Avg hold (days) | Trades / yr | Avg win | Net / trade | Sharpe | Max DD |
|---|---|---|---|---|---|---|---|---|---|
| +10% | 30% | 78.3% | 13.0% | 12.1 | 59.7 | $39 | -1.59% | -1.23 | $1,662 |
| +10% | 40% | 75.1% | 14.0% | 14.0 | 51.6 | $47 | -1.49% | -0.99 | $1,410 |
| +10% | **50%** | 72.4% | 15.2% | 16.0 | 45.1 | $54 | -1.47% | -0.80 | $1,245 |
| +10% | 65% | 68.7% | 15.7% | 18.4 | 39.0 | $63 | -1.29% | -0.55 | $1,138 |
| +10% | 80% | 66.5% | 16.4% | 20.7 | 34.7 | $69 | -1.11% | -0.39 | $1,048 |

Paired, 30% minus 50% (0.20-0.25 delta): Sharpe **-0.42 +/- 0.12 (1/8 seeds better), -0.42 +/- 0.15 (1/8), -0.51 +/- 0.09 (0/8)** at 0/10/20% premium;
win rate +6 points; stop-hit **-2 to -3 points**; trades per year +29%; average win -28%; drawdown 18-41% *worse*.
With 0.12-0.15 delta: Sharpe -0.39, -0.37, -0.43 versus the 50% target. Later targets (65%) at 0.12-0.15 delta: Sharpe **+0.33, +0.60, +0.73** versus the current configuration (8/8 seeds at +10% and +20%), the best configuration tested.

* The 30% target does what it promises on **win rate and stop frequency**, and turns capital over 29% faster.
* It does **not** reduce tail exposure: the worst trade (-$348 vs -$351) and 5th-percentile trade are unchanged, because the
  DTE <= 14 exit already removes the final-two-weeks gamma that an early exit is meant to avoid.
* It makes the payoff ratio worse (smaller wins, unchanged losses) and adds friction per dollar earned. In this model the
  extra turnover does not pay for that: net per trade is no better and Sharpe and drawdown are worse.
* Not modelled: volatility shocks that arrive mid-trade and punish positions held for the last 20 points of profit. That is the
  one mechanism that could favour an early exit in real data, so it is worth testing on real history, not assuming away.

## 3. Wider universe (higher-IV indices, liquid single stocks)

| Profile (delta 0.15-0.25, 50% target) | Trades/yr | Win rate | Stop-hit | Net / trade | Sharpe | 5th pct | Worst |
|---|---|---|---|---|---|---|---|
| SPY-like (vol 16%, premium +10%) | 45 | 72.3% | 15.3% | -1.47% | -0.79 | -$239 | -$335 |
| QQQ-like (vol 22%, +10%) | 44 | 70.8% | 15.9% | -2.35% | -1.04 | -$241 | -$334 |
| High-vol index (vol 24%, **same** +10%) | 37 | 70.1% | 14.8% | -2.79% | -0.76 | -$177 | -$251 |
| High-vol index (vol 24%, **higher** +14%) | 36 | 68.6% | 13.7% | -3.03% | -0.77 | -$175 | -$282 |
| Stock, 0.5% half-spread, +10%, earnings gate on | 20 | 60.1% | 19.7% | -6.63% | -1.02 | -$143 | -$186 |
| Stock, 1.5% half-spread, +10%, gate on | 21 | 59.0% | 23.2% | -6.45% | -0.92 | -$199 | -$280 |
| Stock, 3.0% half-spread, +10%, gate on | 22 | 53.1% | 30.5% | -8.96% | -1.34 | -$228 | -$320 |
| Stock, 1.5%, +10%, news gaps **not** in implied vol | 22 | 60.0% | 26.0% | -7.50% | -1.09 | -$209 | -$284 |
| Stock, 1.5%, +10%, **no** earnings gate | 28 | 58.9% | 21.9% | -5.76% | -0.95 | -$210 | -$291 |
| Stock, 1.5%, premium 0%, gate on | 24 | 59.2% | 27.8% | -8.51% | -1.33 | -$190 | -$290 |
| Stock, 0.5% half-spread, premium **+20%**, gate on | 20 | 60.7% | 17.4% | -4.76% | -0.70 | -$138 | -$202 |

(Stock: price $150, vol 30%, four news gaps a year of 4%, quarterly earnings gap of 6%, widths $1/$2/$5. "Gate on" = the
live earnings exclusion with correct dates.)

* **A higher volatility level is not a higher premium.** At the *same* premium the 22- and 24-vol assets are no better than
  SPY-like (Sharpe -0.25 +/- 0.10 worse for QQQ-like, 7/8 seeds; +0.04 +/- 0.24 for the 24-vol index). Delta-targeting scales
  strikes with volatility, so extra vol mostly means larger moves, not more reward per unit risk. Raising the premium from 10% to
  14% on the high-vol index was within noise (-0.24 +/- 0.28 points).
* **Single stocks are dominated by friction and gaps in this model.** Friction decomposition (below) accounts for roughly four
  points of risk per trade; five-wide-only spreads recover about three of them (-3.6% vs -6.5%) because the fixed $2.60
  commission is a large share of a $1-wide spread's maximum profit (about $19). Even *frictionless* with a +10% effective
  premium, the stock profile is about -1.3% +/- 0.9%: daily stops gap through on news and earnings moves.
* **The earnings gate did not improve expected value here** (Sharpe +0.03 +/- 0.05, 3/8 seeds): the model prices the earnings
  move with the same premium, so trades spanning earnings are not penalised on average. It slightly trims the worst trade
  (+$10, noise). Keep the gate as a gap-risk control, not as a source of edge.
* The premise "single stocks consistently carry a larger premium" is, as far as I recall the literature (not re-checked here),
  not supported for single names relative to indices, and a bigger *stated* premium only helps if it survives the costs and the
  gaps above. Even +20% with tight quotes stayed negative (-4.8%).

## Where the real lever is: friction and fills (SPY-like, +10% premium, 8 seeds)

| Variant | Net / trade (% of risk) | Sharpe |
|---|---|---|
| Frictionless, premium 0% (calibration) | -0.43% +/- 0.37 | -0.27 |
| Frictionless, premium +10% | **+0.90%** +/- 0.40 | +0.55 |
| Frictionless, premium +20% | +1.97% +/- 0.50 | +1.25 |
| Commission only ($0.65 per contract per leg) | +0.24% | +0.15 |
| Bid/ask crossing only (natural fills) | -0.82% | -0.45 |
| Both (current assumption) | -1.47% +/- 0.47 | -0.79 |
| **Fills at the mid, commission kept** | **+0.18%** +/- 0.43 | **+0.12** |

At +10% premium the system keeps a +0.9% gross edge per trade and gives away 2.4 points to costs: about 0.7 to commissions and
1.7 to crossing the spread. Filling at the mid recovers about 1.7 points, **more than any of the three adjustments above
(+0.8 to +1.0 for lower delta, none for the 30% target, negative for stocks)**. Mid fills are an optimistic bound (real working orders get
filled part of the time and part of the way), but it shows execution quality is worth measuring on paper before tuning strategy
parameters. Track the natural-to-mid distance and your actual fill versus mid in the trade log.

## Recommendations

| Adjustment | Verdict | Why |
|---|---|---|
| Lower delta to 0.12-0.15 | **Adopt for paper trading, verify on real data** | Consistent Sharpe (+0.3 to +0.6, 6-8 of 8 seeds), smaller losses and drawdowns; needs lower credit/ROC floors; gains are partly friction, not edge |
| 30% profit target | **Do not adopt** | Better win rate and fewer stops, but worse Sharpe in 0-1 of 8 seeds and 18-41% larger drawdown; no tail reduction. Test 60-65% targets instead |
| Higher-IV indices / single stocks | **Do not expand** | Volatility level alone adds nothing; stocks lose to friction and gaps in this model. If testing, 5-wide-only on the tightest markets, on paper |
| Execution quality | **Measure first** | Largest effect found: about 1.7 points of risk per trade between natural and mid fills |

All of these are conditional on the premium. At +10% (about 1.6 vol points) no tested variant is profitable after costs with natural
fills; at +20% the best (lower delta, 50-65% target) reaches Sharpe 0.3-0.4 and about +$75-90 a year per one-lot slot, with a
standard error near $50-70. That is the realistic scale of the opportunity on a $20k account, before taxes.

## Verification of the adopted configuration (SPY + QQQ synthetic, full disciplined filter set)

Adopted: short delta 0.12-0.15, credit >= 0.45, ROC >= 12%, 65% profit target, SPY/QQQ only. Unlike the grids above,
this run keeps the *other* live filters on (IV/HV >= 1.1, trend alignment, term slope, put skew, 5% / $0.05 leg spreads).
8 seeds, 900 days, both symbols on the same paths, paired against the previous configuration (0.10-0.25 delta,
0.60 / 20% floors, 50% target). Reproduce with `python experiments/verify_adopted.py verify.json`.

| Premium | Config | Trades/yr | Win rate | Net / trade | Sharpe | Max DD | 5th pct trade | Annual $ per 1-lot slot |
|---|---|---|---|---|---|---|---|---|
| +10% | OLD | 35.6 | 71.3% | -1.05% | -0.30 | $1,440 | -$235 | -$148 +/- 123 |
| +10% | **NEW** | 31.1 | 75.1% | +0.25% | +0.09 | $825 | -$162 | +$25 +/- 65 |
| +20% | OLD | 38.7 | 71.6% | -0.25% | -0.09 | $1,173 | -$233 | -$45 +/- 136 |
| +20% | **NEW** | 34.1 | 73.5% | +0.89% | +0.39 | $483 | -$146 | +$131 +/- 51 |

Paired NEW minus OLD: Sharpe **+0.39 +/- 0.16 (6/8 seeds)** at +10% and **+0.48 +/- 0.22 (7/8)** at +20%; net per trade +1.3 and
+1.1 points of risk; drawdown -43% and -59%; stop-hit **+2.0 / +1.7 points (not lower)**; trades per year -13%.

* **The new band finds trades with the new floors (31-34 a year) and none with the old ones** (0 trades, both premium levels).
* The profit target's separate contribution is smaller than in the permissive-filter grid: moving the old rules to 65% changed Sharpe
  by +0.02 +/- 0.14 (3/8) at +10% and +0.28 +/- 0.09 (7/8) at +20%.
* **At +10% premium the new configuration is not distinguishable from zero** (Sharpe +0.09 +/- 0.17, about +$25 +/- $65 a year); it
  is better than the old one, not proven profitable.
* **Resolved: the configured per-trade cap is now 10%.** A $20k backtest on the configured defaults trades 34.1 times a year at about
  $440 risk per trade (max $446), with no risk-gate rejections; the previous 2% cap traded zero times (about 2,700 rejections per seed).
  10% is far looser than needed; the limits that actually bind are the scanner's $1,000 per-lot cap and the 20% total-open-risk cap.
* *Original finding:* **a $20k account with a 2% per-trade risk cap trades nothing:** a 5-wide spread at these credits risks about $443 (cap $400);
  every signal is rejected by the risk gate (2,700 rejections per seed). At 2.5% ($500) the results equal the unconstrained run above.
  Real SPY/QQQ strikes are $1 apart, so narrower widths exist, but at 0.12-0.15 delta a 3-wide spread collects about 0.30, below the 0.45
  credit floor and the 12% ROC floor; the synthetic chains here only have $5 strikes, so that path is untested.

## Exhaustive parameter sweep (`experiments/optimize.py`)

1,440 configs (delta 0.10/0.12/0.15/0.18/0.20 x floors x target 50/60/65/70 x stop 1.5/2/2.5 x DTE window 30-45/45-60 x
early exit 7/10/14/21) on 4 training paths x 2 premia, then 18 finalists re-run on 12 FRESH paths x 2 premia (paired vs CURRENT).
Synthetic data only; friction (commission + natural fills) included.

Validation (net, per 1 lot, $20k): best `d0.15|low|t70|s2.5|45-60|x7` Sharpe 0.20 (CURRENT 0.02), +$3.5/trade vs CURRENT,
max DD $684 (CURRENT $731), ~23 trades/yr; paired Sharpe gain +0.18 +/- 0.07, better in 16/24 datasets.
Runners-up: `d0.18|low|t70|s2|45-60|x7` 0.18; `CURRENT+window45-60` 0.17 (one-parameter change).
Theme: let winners run (target 65-70%, stop 2.5x, exit 7 DTE, 45-60 DTE entry); delta 0.12-0.20 roughly interchangeable once
floors are lowered; delta 0.10 trades too rarely. Caveats: winner's curse (winners' TRAINING Sharpe was negative), gains are ~2 SE,
premium-dependent (+10% VRP: still ~0; +20%: ~0.4), all configs marginal. No defaults changed; re-test on real option history.

## Time-stops and limit-order execution (`experiments/exec_timestop.py`)

Paired grid, 28 variants on 24 UNSEEN datasets (seeds 301-312 x implied-vol premia +10%/+20%, SPY+QQQ, 800 days), on top of
the sweep's recommended set (BASE = `d0.15|low|t70|s2.5|45-60|x7`; CURRENT = the adopted config). Both features are OFF by
default (`RiskLimits.time_stop_fraction`, `BacktestConfig.*_fill_improvement`). Synthetic data; the fill probability and
adverse-selection inputs are ASSUMPTIONS (daily bars cannot calibrate them).

Full table (Sharpe differences are paired vs BASE; max DD in $ per 1 lot):

| variant | Sharpe | dSharpe vs BASE (+/-SE, better/n) | d Sharpe @+10% | d Sharpe @+20% | net $/trade | d annual $ | max DD $ | trades/yr | avg hold d |
|---|---|---|---|---|---|---|---|---|---|
| CURRENT | -0.10 | -0.32 ± 0.11 (8/24) | -0.35 | -0.28 | -1.4 | -117 | 727 | 32.5 | 19.5 |
| BASE | 0.22 | - | 0.00 | 0.00 | 3.3 | 0 | 597 | 22.6 | 33.9 |
| TS f0.4 p0 | -0.17 | -0.39 ± 0.09 (6/24) | -0.45 | -0.34 | -2.5 | -138 | 615 | 24.8 | 27.6 |
| TS f0.4 p10 | -0.15 | -0.37 ± 0.09 (6/24) | -0.41 | -0.34 | -2.2 | -131 | 598 | 25.1 | 27.2 |
| TS f0.4 p20 | -0.13 | -0.36 ± 0.09 (6/24) | -0.37 | -0.34 | -2.0 | -126 | 587 | 25.4 | 26.9 |
| TS f0.4 p30 | -0.14 | -0.37 ± 0.09 (5/24) | -0.35 | -0.38 | -2.1 | -129 | 591 | 25.8 | 26.5 |
| TS f0.5 p0 | -0.02 | -0.24 ± 0.08 (8/24) | -0.30 | -0.19 | -0.7 | -91 | 581 | 23.8 | 29.7 |
| TS f0.5 p10 | -0.05 | -0.27 ± 0.08 (6/24) | -0.33 | -0.21 | -1.1 | -99 | 592 | 23.9 | 29.6 |
| TS f0.5 p20 | -0.07 | -0.29 ± 0.08 (6/24) | -0.35 | -0.23 | -1.3 | -106 | 590 | 24.2 | 29.2 |
| TS f0.5 p30 | -0.08 | -0.30 ± 0.08 (5/24) | -0.37 | -0.24 | -1.5 | -109 | 583 | 24.4 | 28.9 |
| TS f0.6 p0 | 0.18 | -0.05 ± 0.08 (12/24) | -0.10 | 0.00 | 2.2 | -24 | 560 | 22.9 | 31.8 |
| TS f0.6 p10 | 0.17 | -0.05 ± 0.07 (10/24) | -0.10 | -0.00 | 2.1 | -27 | 566 | 23.0 | 31.7 |
| TS f0.6 p20 | 0.14 | -0.08 ± 0.08 (12/24) | -0.11 | -0.04 | 1.7 | -35 | 566 | 23.1 | 31.6 |
| TS f0.6 p30 | 0.13 | -0.09 ± 0.08 (11/24) | -0.14 | -0.04 | 1.5 | -40 | 571 | 23.2 | 31.4 |
| FILL entry.5 | 0.35 | +0.13 ± 0.02 (23/24) | 0.09 | 0.16 | 5.4 | 48 | 581 | 22.6 | 33.9 |
| FILL exit.5 | 0.26 | +0.04 ± 0.00 (24/24) | 0.04 | 0.04 | 4.0 | 16 | 586 | 22.6 | 33.9 |
| FILL f0.25 prob1 adv0 | 0.31 | +0.09 ± 0.02 (22/24) | 0.06 | 0.12 | 4.8 | 34 | 586 | 22.6 | 33.9 |
| FILL f0.25 prob1 adv0.5 | 0.29 | +0.07 ± 0.02 (21/24) | 0.04 | 0.09 | 4.5 | 26 | 590 | 22.6 | 33.9 |
| FILL f0.25 prob0.6 adv0 | 0.31 | +0.09 ± 0.04 (17/24) | -0.02 | 0.20 | 5.0 | 37 | 586 | 22.5 | 34.1 |
| FILL f0.25 prob0.6 adv0.5 | 0.29 | +0.07 ± 0.05 (17/24) | -0.04 | 0.19 | 4.7 | 30 | 588 | 22.4 | 34.1 |
| FILL f0.5 prob1 adv0 | 0.39 | +0.17 ± 0.02 (24/24) | 0.13 | 0.20 | 6.2 | 65 | 571 | 22.6 | 33.9 |
| FILL f0.5 prob1 adv0.5 | 0.31 | +0.09 ± 0.02 (22/24) | 0.06 | 0.12 | 4.8 | 34 | 586 | 22.6 | 33.9 |
| FILL f0.5 prob0.6 adv0 | 0.37 | +0.15 ± 0.06 (18/24) | 0.13 | 0.18 | 6.3 | 67 | 585 | 22.3 | 34.4 |
| FILL f0.5 prob0.6 adv0.5 | 0.31 | +0.09 ± 0.05 (15/24) | 0.06 | 0.11 | 5.2 | 40 | 603 | 22.3 | 34.5 |
| FILL f1 prob1 adv0 | 0.50 | +0.28 ± 0.03 (23/24) | 0.24 | 0.33 | 8.0 | 108 | 567 | 22.7 | 33.7 |
| FILL f1 prob1 adv0.5 | 0.39 | +0.17 ± 0.02 (24/24) | 0.13 | 0.20 | 6.2 | 65 | 571 | 22.6 | 33.9 |
| FILL f1 prob0.6 adv0 | 0.41 | +0.19 ± 0.06 (17/24) | 0.05 | 0.33 | 6.9 | 78 | 614 | 21.9 | 34.6 |
| FILL f1 prob0.6 adv0.5 | 0.32 | +0.10 ± 0.09 (15/24) | -0.01 | 0.21 | 5.4 | 42 | 601 | 21.6 | 35.0 |

Friction-recovery bound (entry+exit at the mid, always filled): +108 $/yr vs BASE. Share of it captured:
  FILL entry.5: 45%
  FILL exit.5: 15%
  FILL f0.25 prob1 adv0: 32%
  FILL f0.25 prob1 adv0.5: 24%
  FILL f0.25 prob0.6 adv0: 34%
  FILL f0.25 prob0.6 adv0.5: 28%
  FILL f0.5 prob1 adv0: 60%
  FILL f0.5 prob1 adv0.5: 32%
  FILL f0.5 prob0.6 adv0: 63%
  FILL f0.5 prob0.6 adv0.5: 37%
  FILL f1 prob1 adv0: 100%
  FILL f1 prob1 adv0.5: 60%
  FILL f1 prob0.6 adv0: 72%
  FILL f1 prob0.6 adv0.5: 39%

Findings
* Time-stops HURT. Every variant has lower Sharpe than BASE; fractions 0.4-0.5 cost 0.24-0.39 Sharpe (about 3-4 SE). The mildest
  (close past 60% of life, only if not profitable) is statistically a wash (-0.05 +/- 0.08) and trims max DD 597 -> 560.
  The "frees buying power" mechanism is not binding here: trades/yr barely moves (22.6 -> 22.9) because the limit is one
  open spread per (symbol, side), not capital. Closing stale trades early locks in losses on positions that mostly
  recover or decay to the target. Keep OFF.
* Limit-order execution is the one lever with a clear, large effect, but it is ASSUMPTION-driven. Resting at the mid with
  guaranteed fills (the upper bound) adds +0.28 Sharpe and ~$108/yr per lot, more than BASE's whole net P&L (~$75/yr):
  friction is bigger than the edge. Halfway-to-mid with a 60% fill rate and half the gain lost to adverse selection still
  gives +0.09 +/- 0.05 (15/24 datasets, ~$40/yr). Entry improvement carries most of the value (45% of the bound for f=0.5
  at entry only vs 15% at exit only); stop-losses always cross.
* Scanner floors are tested at the natural credit (conservative); a mid-aware scanner would also admit more trades.
