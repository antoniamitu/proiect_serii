# Multivariate VAR/VECM Results — Interpretation

This note summarizes the econometric output from `src/multivariate_analysis.py` as run on the monthly Romanian dataset with four log-transformed variables: commodity/import prices (CTOT), EUR/RON, Food HICP, and industrial production (PI_SA). Figures are saved under `outputs/figures/multivariate/`.

---

## 1. Purpose and specification

The multivariate analysis examines whether external price pressures are transmitted into Romanian food prices and real activity through a system including:

- `log_CTOT`
- `log_EUR_RON`
- `log_Food_HICP`
- `log_PI_SA`

The estimated specification is a **VECM** with:

- **cointegration rank** `r = 2`
- **one lag in differences** (`k_ar_diff = 1`)
- **deterministic specification** `ci`

Impulse responses and FEVD are based on **Cholesky ordering**:

**CTOT → EUR_RON → Food_HICP → PI_SA**

This ordering treats CTOT as the most exogenous variable in the short run, but the interpretation remains **ordering-dependent**.

---

## 2. Integration order (unit roots)

ADF and KPSS were applied to each log series in levels and first differences. For level series, both **KPSS_c** and **KPSS_ct** were considered.

### Main conclusions

- **log_Food_HICP** and **log_EUR_RON** show **strong evidence of I(1)** behavior:
  - ADF does not reject a unit root in levels
  - KPSS rejects stationarity in levels
  - first differences appear stationary

- **log_CTOT** and **log_PI_SA** show **mixed evidence in levels**, but first differences are clearly stationary.
  - For these variables, the code classifies them as **likely I(1)** rather than perfectly clean I(1) cases.

### Interpretation

Taken together, the system is treated as **I(1) in an applied project sense**, which is sufficient to proceed with Johansen cointegration testing, but with some caution because not all level tests are equally clean across variables.

---

## 3. Cointegration (Johansen)

Johansen tests were run on the log-level system.

### Results

- **Trace test:** selected **rank r = 2**
- **Max-eigenvalue test:** also selected **rank r = 2**

The two Johansen criteria therefore **agree** on the selected rank in this specification.

### Interpretation

Under the chosen lag structure and deterministic specification, the results support the existence of **two cointegrating relationships** among the four variables. This suggests that the variables share a common long-run structure and do not drift independently over time.

Because the system is modeled in logs and classified as I(1) / likely I(1), a **VECM** is more appropriate than a VAR estimated only on first differences.

---

## 4. VECM interpretation

The estimated VECM shows both short-run dynamics and error-correction effects.

### Short-run block

- The **Food HICP equation** shows strong persistence through its own lagged term.
- Most cross-variable short-run coefficients are not statistically significant at conventional levels.
- The **EUR/RON equation** includes a positive and significant lagged coefficient on Food HICP, but this should be interpreted as a **conditional system relationship**, not as structural causality.

### Error-correction terms

- In the **Food HICP equation**, the loading on the second error-correction term (`ec2`) is positive and statistically significant.
- In the **PI_SA equation**, the loading on `ec2` is large, negative, and strongly significant.
- In the **EUR/RON equation**, the loading on `ec1` is negative and significant.

### Interpretation

These results suggest that some variables respond to deviations from the estimated long-run relations, but the adjustment pattern is **not symmetric across equations**. In particular, Food HICP and PI_SA appear to participate in the correction of the second cointegrating relation, while EUR/RON adjusts more clearly to the first.

---

## 5. Granger causality

Two sets of Granger-type results were reported:

1. **Bivariate Granger tests** on first differences of logs
2. **Wald Granger tests inside the VECM**

### Bivariate results

For the reported pairs:

- CTOT ↔ Food HICP
- Food HICP ↔ PI_SA
- CTOT → EUR/RON
- EUR/RON → Food HICP

the null of no Granger causality is **not rejected** at the 5% level.

### VECM Wald results

Within the full system:

- **CTOT → EUR/RON** is **statistically significant**
- all other tested directions **fail to reject** at 5%

### Interpretation

The only robust predictive relationship in the VECM Wald framework is:

**CTOT → EUR/RON**

For Food HICP, the reported Granger tests do **not** provide evidence of strong short-run predictive causality from CTOT, EUR/RON, or PI_SA at conventional significance levels. This does not contradict the cointegration result: long-run comovement and short-run predictive causality are different concepts.

---

## 6. Impulse response functions (IRF)

IRFs are orthogonalized through the Cholesky decomposition using the ordering:

**CTOT → EUR_RON → Food_HICP → PI_SA**

### Interpretation

This means that:
- CTOT shocks can affect all later variables contemporaneously
- EUR/RON can affect Food HICP and PI_SA contemporaneously
- Food HICP can affect PI_SA contemporaneously
- reverse contemporaneous effects are restricted by construction

The IRFs therefore provide a useful dynamic illustration of transmission mechanisms, but they are **not structurally invariant**. Different orderings could change the short-run responses.

---

## 7. Forecast error variance decomposition (FEVD) for Food HICP

For the **Food HICP equation**, the FEVD shows the following approximate shares:

| Horizon | CTOT | EUR/RON | Food HICP | PI_SA |
|--------|------|---------|-----------|-------|
| 12 months | 4.9% | 9.6% | 80.3% | 5.1% |
| 24 months | 9.0% | 10.0% | 75.9% | 5.1% |

### Interpretation

- **Own shocks dominate** the forecast error variance of Food HICP at both horizons.
- **EUR/RON** and **CTOT** play a visible but secondary role.
- **PI_SA** contributes relatively little in this specification.

This pattern is consistent with a story of **strong domestic persistence in food prices**, combined with **partial transmission from external prices and exchange-rate conditions**.

---

## 8. Overall conclusion

The multivariate evidence supports the following picture:

1. The four log series are treated as **I(1) / likely I(1)** in a way that is sufficient for applied Johansen analysis.
2. Johansen tests support **cointegration rank 2**, which justifies estimation of a **VECM**.
3. The VECM suggests meaningful long-run relationships, with significant adjustment visible especially in **Food HICP**, **EUR/RON**, and **PI_SA** through the error-correction terms.
4. Short-run Granger evidence is limited: the only clear result in the system-based Wald tests is **CTOT → EUR/RON**.
5. FEVD indicates that Food HICP dynamics are driven mainly by **own persistence**, while **CTOT** and **EUR/RON** have non-negligible but smaller contributions over medium horizons.

Overall, the results are consistent with a **partial transmission mechanism** from external commodity and exchange-rate conditions into Romanian food prices, rather than with a dominant short-run causal chain visible in every test.

---

## 9. Limitations

The main limitations worth stating in the written report are:

- **Mixed integration evidence** for some variables in levels, especially CTOT and PI_SA
- **Sensitivity to the chosen system**, especially the inclusion of EUR/RON as a fourth variable
- **Sensitivity to deterministic specification and lag structure**
- **Recursive identification via Cholesky ordering**, which is assumption-dependent
- A relatively short monthly sample for a four-variable cointegrated system

---

*This interpretation is designed to accompany the empirical output of the multivariate pipeline and should be kept consistent with the final reported specification.*