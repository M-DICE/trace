### **Changepoint Detection**
**What it is:** A statistical method to identify when a time series changes behavior.

**Why it matters:** In rewilding, we want to know if an intervention actually caused a shift in the ecological indicator.

**Plain English:** Imagine measuring water temperature before and after dam removal. The method asks: "Did the average temperature shift at the moment of removal?"

**Resources:**
- [Wikipedia: Changepoint Detection](https://en.wikipedia.org/wiki/Change_detection)
- [Killick et al. (2012) "Optimal Detection of Changepoints"](https://arxiv.org/abs/1101.1438)

---

### **AMOC (At Most One Change)**
**What it is:** An offline changepoint detection method that assumes there is at most one change point in the entire time series.

**Why it matters:** In a rewilding intervention study, we expect one clear shift when the intervention starts (plus any response lag). AMOC tests this efficiently.

**Plain English:** You look at the entire time series at once and ask: "Where is the most likely point where the trend changed?" If no point stands out clearly, the answer is "nowhere."

**Resources:**
- **TBD**

---

### **T_max (Test Statistic)**
**What it is:** The maximum value of a statistical test computed across all possible changepoint locations.

**Why it matters:** AMOC scans every month as a potential changepoint, calculates a test score for each, and uses the highest score to decide if a change occurred.

**Plain English:** Imagine scoring how "suspicious" each month looks as a changepoint. T_max is the most suspicious month's score.

**Resources:**
- **TBD**

---

## Noise Models

### **i.i.d. (Independent and Identically Distributed) Noise**
**What it is:** Random variation in measurements where each observation is independent of the previous one, drawn from the same distribution.

**Why it matters:** The simplest noise assumption. If you use methods designed for i.i.d. but real data has autocorrelation, your statistical tests become unreliable.

**Plain English:** Think of measuring a stream's nitrogen level each month. With i.i.d. noise, a high reading in January doesn't tell you anything about February's reading. That's the i.i.d. assumption.

**Resources:**
- [Wikipedia: Independent and Identically Distributed Random Variables](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)

---

### **AR(1) (Autoregressive Model of Order 1)**
**What it is:** A statistical model where each observation depends on the previous observation plus random noise. Captures temporal autocorrelation (memory) in data.

**Why it matters:** Ecological data often have "sticky" behavior — if a population is high this month, it tends to stay high next month. Ignoring this correlation inflates false positive rates and reduces the sensitivity of statistical tests.

**Plain English:** Bird population size follows an AR(1) if: "Next month's population ≈ 0.7 × (this month's) + random noise." The 0.7 is the "memory" or autocorrelation coefficient (φ, "phi"). Higher φ means stronger memory; φ = 0 means no memory (same as i.i.d. noise).

**Resources:**
- [Wikipedia: Autoregressive Model](https://en.wikipedia.org/wiki/Autoregressive_model)
- [Box & Jenkins (1970) "Time Series Analysis: Forecasting and Control"](https://www.wiley.com/en-us/Time+Series+Analysis:+Forecasting+and+Control,+5th+Edition-p-9781118675778)

---

## Statistical Power & Detection

### **Power (Statistical Power)**
**What it is:** The probability of correctly detecting a real change when one exists.

**Why it matters:** Low power means you might miss real rewilding effects. High power (80%+) means you're likely to catch them if they're there.

**Plain English:** If the rewilding *really* works (has a real effect), what's the chance your test will spot it? That's power. We typically aim for 80% or higher.

**Resources:**
- [Cohen (1988) "Statistical Power Analysis for the Behavioral Sciences"](https://www.routledge.com/Statistical-Power-Analysis-for-the-Behavioral-Sciences/Cohen/p/book/9780805802832)

---

### **Effect Size**
**What it is:** The magnitude of the change you're trying to detect. Expressed as a percentage of the baseline in this analysis.

**Why it matters:** A 5% trend change is harder to detect than a 50% change. The analysis tests across effect sizes to show "how big does the change need to be to catch it reliably?"

**Plain English:** If a fish population is at 1000 individuals with a 5% effect, the post-intervention trend adds 50 fish/year. With 50%, it adds 500 fish/year. Larger effects are easier to spot.

**Resources:**
- [Wikipedia: Cohen's Effect Size Conventions](https://en.wikipedia.org/wiki/Effect_size)

---

## Data Design Terms

### **CIBA (Control-Intervention Before-After)**
**What it is:** A study design with two parallel time series — one control (no intervention) and one intervention — observed both before and after a change point.

**Why it matters:** The control series absorbs shared environmental variation (e.g., drought affecting both sites). Subtracting control from intervention isolates the intervention's true effect.

**Plain English:** Monitor two streams — one untouched (control), one restored (intervention). Any difference between them is more likely the restoration's doing.

**Resources:**
- [Campbell & Cook (1979) "Quasi-Experimentation"](https://www.degruyter.com/document/doi/10.4159/9780674037076/html)

---

### **BA (Before-After)**
**What it is:** A simpler design using only the intervention time series, with separate statistical models for before and after the changepoint.

**Why it matters:** When autocorrelation is high (AR(1)), or when there's no control site, BA is used. It compares the trend *before* the change to the trend *after*.

**Plain English:** Watch one site before the intervention starts, then after. Did the slope change? That's your answer.

---

### **Monitoring Window (`npost`)**
**What it is:** The length of time you observe after the intervention begins, measured in months.

**Why it matters:** Longer windows give more post-intervention data, strengthening evidence. Our analysis tests 2–10 years to show how window length affects detection.

**Plain English:** If you monitor for only 1 year post-intervention, you might not accumulate enough evidence to declare a change. 5 years gives you more confidence.

---

### **Pre-intervention Period (`npre`)**
**What it is:** The length of time you observe before the intervention, used to establish the baseline trend.

**Why it matters:** A longer pre-period estimates the "no-change" trend more precisely, which sharpens the test.

---

### **Intervention Delay**
**What it is:** The lag between the formal intervention date and when the ecological response actually begins.

**Why it matters:** Ecosystems rarely respond instantly. Seeds need to germinate, predators need to breed, vegetation needs to grow. This lag reduces your effective post-intervention signal duration.

**Plain English:** You reintroduce wolves on Jan 1, but deer behavior doesn't change noticeably until June. That 5-month delay reduces your detection power.

