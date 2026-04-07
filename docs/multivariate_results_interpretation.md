# Multivariate VAR/VECM Results: Interpretation

This note summarizes the econometric output from `src/multivariate_analysis.py` as run in your environment (monthly data, four variables in logs: commodity/import price index CTOT, EUR/RON, Food HICP, industrial production PI_SA). Figures are saved under `outputs/figures/multivariate/`.

---

## 1. Purpose and specification

The analysis studies how shocks to import/commodity prices and related macro variables relate to **food consumer prices** and industrial production in Romania, in the spirit of transmission from external price pressures. The estimated system uses **natural logs**, a **VECM** with **cointegration rank r = 2**, and **one lag in differences** (`k_ar_diff = 1`). Impulse responses and variance decomposition use **Cholesky ordering**: CTOT → EUR_RON → Food_HICP → PI_SA, which treats CTOT as the most “exogenous” block in the short run and is a standard but **assumption-dependent** choice.

---

## 2. Integration order (unit roots)

For each log series, **ADF** (null: unit root) and **KPSS** (null: stationarity) were applied to **levels** and **first differences**.

- **log_Food_HICP** and **log_EUR_RON**: Clear **I(1)** pattern—levels behave as non-stationary (ADF does not reject unit root; KPSS rejects stationarity at 5%), while first differences are stationary.
- **log_CTOT** and **log_PI_SA**: Classified as **I(1)** with a **mixed-signal warning** on levels (e.g. ADF p-values sometimes suggest stationarity for PI_SA in levels, or borderline KPSS for CTOT). That tension is common when tests have different null hypotheses and when series are persistent. The **joint conclusion** used in the code still treats them as **I(1)** for multivariate modelling, which is acceptable for applied work if differences are clearly stationary—here, first differences pass the usual checks.

**Takeaway:** The panel is treated as **integrated of order one**, which justifies **Johansen** on log levels and a **VECM** rather than a VAR only in differences without long-run structure.

---

## 3. Cointegration (Johansen)

- **Trace test:** Strong rejection of no cointegration (`r = 0`) and of at most one cointegrating vector (`r ≤ 1`). **No rejection** for the hypothesis that rank is at most 2 at the 5% critical value, so the **selected rank is r = 2**.
- **Max eigenvalue test:** Consistent with **at least two** cointegrating relationships (reject for `r = 0` and `r = 1`; do not reject for the step corresponding to `r = 2` in the same way as the trace step).

**Interpretation:** There are **two long-run equilibrium relationships** tying the four variables together. Food prices do not drift arbitrarily relative to CTOT, the exchange rate, and industrial production; deviations are partly corrected over time through the error-correction mechanism (see loading coefficients `alpha` on the estimated error terms `ec1`, `ec2` in the output).

---

## 4. VECM: short-run dynamics and error correction (selected points)

The printed coefficients show, among other things:

- **Food equation:** Strong **persistence in own lags** (coefficient on L1.log_Food_HICP ≈ 0.69, highly significant)—typical for CPI-type series.
- **Error correction:** For **log_Food_HICP**, the second cointegrating vector enters with a **positive and significant** loading on `ec2` (≈ 0.065, p ≈ 0.013), meaning food prices **adjust** toward that long-run relationship when the system is away from equilibrium. For **log_PI_SA**, loadings on `ec2` are large and negative (≈ −0.89), indicating industrial production responds strongly to that equilibrium error.
- **EUR equation:** Lagged food (L1.log_Food_HICP) is **positively** associated with EUR/RON in the short-run block (significant), which must be read as **conditional correlation within the system**, not necessarily a structural causal claim.

These results describe **conditional correlations and adjustment speeds** within the VECM; they do not replace economic identification of shocks.

---

## 5. Granger causality

Two layers were reported: **bivariate Granger on first differences of logs** (simple predictive content) and **multivariate Wald tests inside the VECM** (consistent with the estimated system).

### 5.1 Bivariate tests (differences, lag = 1)

For all listed pairs—CTOT ↔ Food, Food ↔ PI_SA, CTOT → EUR, EUR → Food—the **F-tests do not reject** the null of no Granger causality at 5%. So in this **minimal bivariate setup on Δlogs**, there is **no evidence** of incremental predictability at lag 1.

### 5.2 VECM Wald tests

- **CTOT → EUR:** **Rejected** (p ≈ 0.028): lagged CTOT terms appear to help predict EUR/RON **within the full system** that includes cointegration restrictions—**not** the same as the bivariate result for CTOT → EUR on differences alone.
- **Other directions** (CTOT → Food, Food ↔ CTOT, Food ↔ PI, EUR → Food): **Fail to reject** no Granger causality at usual levels in the Wald framework used.

**How to reconcile bivariate vs VECM:** Bivariate tests ignore **long-run relationships** and **omitted variables**. The VECM bundles all four series and error-correction terms; significance can appear or disappear relative to simple Δlog regressions. The **CTOT → EUR** contrast is a good example: multivariate evidence is stronger than the simple bivariate test at the same nominal lag length.

**Caution:** “Granger causality” is **predictive**, not structural. It does not prove economic causation in the policy sense.

---

## 6. Impulse responses (IRF)

IRFs are **orthogonalized** via Cholesky with order **CTOT, EUR_RON, Food_HICP, PI_SA**. A shock to CTOT is assumed to affect all variables **contemporaneously** through the recursive structure; variables later in the order do not affect earlier ones **within the same period**.

**Interpretation:** The IRF graphs in `irf_vecm.png` (or split panels if saved as multiple files) show **dynamic responses** to one-standard-deviation shocks under that ordering. Any narrative about “commodity shock → food prices” must explicitly acknowledge **ordering sensitivity**: putting EUR before Food means exchange-rate shocks are positioned before food in the contemporaneous matrix; different orderings can change short-run attribution.

---

## 7. Forecast error variance decomposition (FEVD) for Food HICP

For the **log_Food_HICP** equation, **orthogonalized** shocks contribute to the forecast error variance as follows (approximate shares):

| Horizon (months) | Own (Food) | CTOT   | EUR_RON | PI_SA  |
|-----------------|------------|--------|---------|--------|
| 12              | ~80.3%     | ~4.9%  | ~9.6%   | ~5.1%  |
| 24              | ~75.9%     | ~9.0%  | ~10.0%  | ~5.1%  |

**Reading:**

- **Own shocks dominate** food inflation forecast errors at both horizons—expected for a highly persistent price series.
- **EUR_RON** explains roughly **10%** of the variance by 24 months, **CTOT** rises from ~5% to ~9% as the horizon lengthens—suggesting that **external price and exchange-rate shocks** matter, but **not** as much as domestic persistence in this decomposition **under the chosen Cholesky order**.
- **PI_SA** stays around **5%**, a modest role in this specification.

Again, shares are **not structural**; they depend on **ordering** and the **VECM** parameterization.

---

## 8. Overall conclusion for a report

1. **Data support modelling the four log series as I(1)** with **two cointegrating vectors**, motivating a **VECM(1)** with **r = 2** rather than a plain VAR in differences.
2. **Long-run ties** exist between commodity/import prices, the exchange rate, food prices, and industrial production; **error-correction** is visible in the loadings (notably food and PI).
3. **Granger results are mixed and framework-dependent:** simple bivariate tests on differences find little; the **system-based Wald test** suggests **CTOT helps predict EUR** within the VECM. Other pairwise Granger hypotheses are not supported at 5% in that Wald setup.
4. **FEVD** indicates food inflation dynamics are **largely driven by own shocks**, with **non-negligible** roles for **EUR** and **CTOT** over 12–24 months under Cholesky ordering—consistent with a story of **partial transmission** of external and FX conditions, without claiming precise magnitudes for policy counterfactuals.

---

## 9. Limitations (worth stating in the thesis)

- **Sample and regime:** One structural break (e.g. 2021–2023 energy shock) may not be fully captured by a linear VECM over the full sample.
- **Cholesky ordering** identifies shocks only **recursively**; results are **not invariant** to reordering.
- **Lag length** fixed by information criteria at **one lag in differences**; sensitivity analysis with **r** or **k** varied slightly would strengthen robustness claims.
- **Seasonality** is not modelled explicitly in the VECM output shown; if monthly seasonality remains in residuals, seasonal extensions could be discussed.

---

*Generated to accompany the empirical output from the project’s multivariate pipeline. Adjust wording to match your faculty’s style guide.*
