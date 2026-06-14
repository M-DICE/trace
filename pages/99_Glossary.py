"""
Glossary of Statistical Terms
Educational reference page explaining key terms for non-statisticians.
"""

import streamlit as st

st.set_page_config(
    page_title="Glossary",
    page_icon="📚",
    layout="wide",
)

# Page registry
PAGES = {
    "01": ("pages/01_Trend_Change_AMOC.py", "Trend Change (AMOC)"),
    "02": ("pages/02_Distribution_Change_AMOC.py", "Distribution Change (AMOC)"),
    "03": ("pages/03_Trend_Change_Forecast.py", "Trend Change (Forecast)"),
    "04": ("pages/04_Distribution_Change_Forecast.py", "Distribution Change (Forecast)"),
    "05": ("pages/05_Trend_Change_BOCPD.py", "Trend Change (BOCPD)"),
    "06": ("pages/06_Distribution_Change_BOCPD.py", "Distribution Change (BOCPD)"),
}

# Glossary entries
GLOSSARY = [
    {
        "term": "Changepoint Detection",
        "what": "A statistical method to identify when a time series changes behavior.",
        "why": (
            "In rewilding, we want to know if an intervention actually caused a shift "
            "in the ecological indicator."
        ),
        "plain_english": (
            "Imagine measuring water temperature before and after dam removal. "
            'The method asks: "Did the average temperature shift at the moment of removal?"'
        ),
        "resources": [
            ("[Wikipedia: Changepoint Detection](https://en.wikipedia.org/wiki/Change_detection)"),
            (
                '[Killick et al. (2012) "Optimal Detection of Changepoints"](https://arxiv.org/abs/1101.1438)'
            ),
        ],
        "relevant_for": ["01", "02", "03", "04", "05", "06"],
    },
    {
        "term": "AMOC (At Most One Change)",
        "what": (
            "An offline changepoint detection method that assumes there is at most one "
            "change point in the entire time series."
        ),
        "why": (
            "In a rewilding intervention study, we expect one clear shift when the "
            "intervention starts (plus any response lag). AMOC tests this efficiently."
        ),
        "plain_english": (
            'You look at the entire time series at once and ask: "Where is the most likely '
            'point where the trend changed?" If no point stands out clearly, '
            'the answer is "nowhere."'
        ),
        "resources": [],
        "relevant_for": ["01", "02"],
    },
    {
        "term": "T_max (Test Statistic)",
        "what": (
            "The maximum value of a statistical test computed across all possible "
            "changepoint locations."
        ),
        "why": (
            "AMOC scans every month as a potential changepoint, calculates a test score "
            "for each, and uses the highest score to decide if a change occurred."
        ),
        "plain_english": (
            'Imagine scoring how "suspicious" each month looks as a changepoint. '
            "T_max is the most suspicious month's score."
        ),
        "resources": [],
        "relevant_for": ["01", "02"],
    },
    {
        "term": "i.i.d. (Independent and Identically Distributed) Noise",
        "what": (
            "Random variation in measurements where each observation is independent of the "
            "previous one, drawn from the same distribution."
        ),
        "why": (
            "The simplest noise assumption. If you use methods designed for i.i.d. but real "
            "data has autocorrelation, your statistical tests become unreliable."
        ),
        "plain_english": (
            "Think of counting deer at a grazing site each month. With i.i.d. noise, a "
            "higher-than-usual count in January tells you nothing about February's count and "
            "each observation is a fresh draw from the same distribution. "
            "That's the i.i.d. assumption."
        ),
        "resources": [
            "[Wikipedia: Independent and Identically Distributed Random Variables](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)",
        ],
        "relevant_for": ["01", "02", "03", "04", "05"],
    },
    {
        "term": "AR(1) (Autoregressive Model of Order 1)",
        "what": (
            "A statistical model where each observation depends on the previous observation "
            "plus random noise. Captures temporal autocorrelation (memory) in data."
        ),
        "why": (
            'Ecological data often have "sticky" behavior: if a population is high this '
            "month, it tends to stay high next month. Ignoring this correlation inflates "
            "false positive rates and reduces the sensitivity of statistical tests."
        ),
        "plain_english": (
            "Bird population size follows an AR(1) if: \"Next month's population ≈ 0.7 × "
            '(this month\'s) + random noise." The 0.7 is the "memory" or autocorrelation '
            'coefficient (φ, "phi"). Higher φ means stronger memory; '
            "φ = 0 means no memory (same as i.i.d. noise)."
        ),
        "resources": [
            "[Wikipedia: Autoregressive Model](https://en.wikipedia.org/wiki/Autoregressive_model)",
            '[Box & Jenkins (1970) "Time Series Analysis: Forecasting and Control"](https://www.wiley.com/en-us/Time+Series+Analysis:+Forecasting+and+Control,+5th+Edition-p-9781118675778)',
        ],
        "relevant_for": ["01", "02", "03", "04"],
    },
    {
        "term": "Power (Statistical Power)",
        "what": "The probability of correctly detecting a real change when one exists.",
        "why": (
            "Low power means you might miss real rewilding effects. "
            "High power (80%+) means you're likely to catch them if they're there."
        ),
        "plain_english": (
            "If the rewilding *really* works (has a real effect), what's the chance your "
            "test will spot it? That's power. We typically aim for 80% or higher."
        ),
        "resources": [
            '[Cohen (1988) "Statistical Power Analysis for the Behavioral Sciences"](https://www.routledge.com/Statistical-Power-Analysis-for-the-Behavioral-Sciences/Cohen/p/book/9780805802832)',
        ],
        "relevant_for": ["01", "02", "03", "04", "05", "06"],
    },
    {
        "term": "Effect Size",
        "what": (
            "The magnitude of the change you're trying to detect. Expressed as a percentage "
            "of the baseline in this analysis."
        ),
        "why": (
            "A 5% trend change is harder to detect than a 50% change. The analysis tests "
            'across effect sizes to show "how big does the change need to be to catch it '
            'reliably?"'
        ),
        "plain_english": (
            "If a fish population is at 1000 individuals with a 5% effect, the "
            "post-intervention trend adds 50 fish/year. With 50%, it adds 500 fish/year. "
            "Larger effects are easier to spot."
        ),
        "resources": [
            "[Wikipedia: Cohen's Effect Size Conventions](https://en.wikipedia.org/wiki/Effect_size)",
        ],
        "relevant_for": ["01", "02", "03", "04", "05", "06"],
    },
    {
        "term": "BACI (Before-After Control-Intervention)",
        "what": (
            "A study design with two parallel time series: one control (no intervention) "
            "and one intervention, observed both before and after a change point."
        ),
        "why": (
            "The control series absorbs shared environmental variation (e.g., drought "
            "affecting both sites). Subtracting control from intervention isolates the "
            "intervention's true effect."
        ),
        "plain_english": (
            "Monitor two streams: one untouched (control), one restored (intervention). "
            "Any difference between them is more likely the restoration's doing, because "
            "both streams experience the same weather, seasonal cycles, "
            "and other background changes."
        ),
        "resources": [
            '[Campbell & Cook (1979) "Quasi-Experimentation"](https://www.degruyter.com/document/doi/10.4159/9780674037076/html)',
        ],
        "relevant_for": ["01", "02", "03", "04", "05", "06"],
    },
    {
        "term": "BA (Before-After)",
        "what": (
            "A simpler design using only the intervention time series, with separate "
            "statistical models for before and after the changepoint."
        ),
        "why": (
            "Used when there is no suitable control site, or when autocorrelation is strong "
            "enough that the control series adds more noise than it removes. "
            "It compares the trend *before* the change to the trend *after*."
        ),
        "plain_english": (
            "Watch one site before the intervention starts, then after. Did the slope or the "
            "distribution change? That's your answer. The downside is that you cannot "
            "distinguish the intervention's effect from background environmental trends."
        ),
        "resources": [],
        "relevant_for": ["01", "02", "03", "04"],
    },
    {
        "term": "Wasserstein Distance",
        "what": (
            "A measure of how different two probability distributions are, calculated as the "
            'minimum "work" needed to transform one distribution into the other. '
            "Also called the Earth Mover's Distance (EMD)."
        ),
        "why": (
            "When a rewilding intervention changes the *shape* of a population's distribution, "
            "not just its average standard mean-based tests, miss it. Wasserstein distance "
            "captures changes in mean, variance, skewness, and any other aspect of the "
            "distribution simultaneously."
        ),
        "plain_english": (
            "Imagine two piles of earth (the two distributions). Wasserstein distance is the "
            "amount of earth you'd have to shovel, multiplied by the distance you'd move each "
            "shovelful, to turn one pile into the other. If the piles look alike, the distance "
            "is small; if they look very different, the distance is large."
        ),
        "resources": [
            "[Wikipedia: Wasserstein Distance](https://en.wikipedia.org/wiki/Wasserstein_metric)",
            (
                '[Schuhmacher et al. (2024) "transport: Computation of Optimal Transport Plans '
                'and Wasserstein Distances" (R package)](https://cran.r-project.org/package=transport)'
            ),
        ],
        "relevant_for": ["02", "04", "06"],
    },
    {
        "term": "Monitoring Window (`npost`)",
        "what": "The length of time you observe after the intervention begins, measured in months.",
        "why": (
            "Longer windows give more post-intervention data, strengthening evidence. "
            "Our analysis tests 2–10 years to show how window length affects detection."
        ),
        "plain_english": (
            "If you monitor for only 1 year post-intervention, you might not accumulate "
            "enough evidence to declare a change. 5 years gives you more confidence."
        ),
        "resources": [],
        "relevant_for": ["01", "02", "03", "04"],
    },
    {
        "term": "Pre-intervention Period (`npre`)",
        "what": (
            "The length of time you observe before the intervention, "
            "used to establish the baseline trend."
        ),
        "why": (
            'A longer pre-period estimates the "no-change" trend more precisely, '
            "which sharpens the test."
        ),
        "plain_english": None,
        "resources": [],
        "relevant_for": ["01", "02", "03", "04", "05", "06"],
    },
    {
        "term": "Intervention Delay",
        "what": (
            "The lag between the formal intervention date and when the ecological response "
            "actually begins."
        ),
        "why": (
            "Ecosystems rarely respond instantly. Seeds need to germinate, predators need to "
            "breed, vegetation needs to grow. This lag reduces your effective post-intervention "
            "signal duration."
        ),
        "plain_english": (
            "You reintroduce wolves in January, but deer behavior doesn't change noticeably "
            "until June. That 5-month delay reduces your detection power."
        ),
        "resources": [],
        "relevant_for": ["01", "02", "03", "04", "05", "06"],
    },
    {
        "term": "Forecast-based Detection (Page-CUSUM)",
        "what": (
            "A sequential changepoint detection method that monitors cumulative deviations "
            "from a forecast model in real time, raising an alarm the first time the data "
            "diverges significantly from its predicted path."
        ),
        "why": (
            'AMOC looks at the whole time series after the fact and asks "where did it change?" '
            'Page-CUSUM watches the series unfold month by month and asks "has it broken yet?" '
            "This makes it suitable for prospective monitoring, not just retrospective analysis."
        ),
        "plain_english": (
            "You monitor bird counts at a restored wetland each month. AMOC waits until the "
            "study ends and then looks back across the whole series to find where the trend "
            "broke. Page-CUSUM watches each month's count as it arrives and if the moment "
            "cumulative deviations from the forecast grow large enough, it raises an alarm. "
            "The earlier that alarm sounds, the less time passes before you know the ecosystem "
            "has shifted."
        ),
        "resources": [
            '[Page (1954) "Continuous Inspection Schemes"](https://www.jstor.org/stable/2333009)',
        ],
        "relevant_for": ["03", "04"],
    },
    {
        "term": "Critical Value",
        "what": (
            "A pre-computed threshold for the test statistic. If `T_max` (AMOC) or the "
            "cumulative sum (Page-CUSUM) exceeds this value, the method declares a change "
            "detected."
        ),
        "why": (
            'The app reports "Change Detected" or "Not Detected" based on whether the test '
            "statistic crosses this boundary. The threshold is calibrated to a target "
            "false-positive rate."
        ),
        "plain_english": (
            "Camera traps at a rewilding site occasionally misfire on swaying branches. "
            "Set the threshold too low and the system declares a wolf sighting every time "
            "the wind picks up; set it too high and it misses a real sighting. The critical "
            "value sets that sensitivity for the statistical test."
        ),
        "resources": [],
        "relevant_for": ["01", "02", "03", "04"],
    },
    {
        "term": "Drift in Mean vs. Variance",
        "what": (
            "Two distinct ways an ecological distribution can change over time. "
            "A *mean drift* shifts the typical population level up or down; "
            "a *variance drift* makes the population more or less variable."
        ),
        "why": (
            "Rewilding might not just increase average abundance, it might stabilise a "
            "previously volatile population, or vice versa."
        ),
        "plain_english": (
            "Imagine a fish population that hovers around 500 fish but swings wildly between "
            "200 and 800 each year. After a restoration, it still averages 500 but now stays "
            "between 400 and 600. The mean didn't change, the variance did. "
            "The distribution pages test for both."
        ),
        "resources": [],
        "relevant_for": ["02", "04", "06"],
    },
    {
        "term": "Two-Stage Detection (Detection Time vs. Changepoint Estimate)",
        "what": (
            "A two-step process used by the online methods (Forecast and BOCPD). First, it "
            "records the month the alarm is raised (detection time). Then, it goes back and "
            "estimates when the underlying change actually began (changepoint estimate)."
        ),
        "why": (
            "The alarm always rings *after* there is enough cumulative evidence, so it "
            "necessarily lags behind the true changepoint. Separating the two gives you both "
            'an operational answer ("when did we know?") and a scientific one '
            '("when did the ecosystem actually shift?").'
        ),
        "plain_english": (
            "Wolves are reintroduced in January, and the elk population begins shifting in "
            "March, but the monitoring system only accumulates enough evidence to confirm a "
            "change by August. August is the detection time; March is the changepoint estimate. "
            "The gap between them is the detection delay."
        ),
        "resources": [],
        "relevant_for": ["03", "04", "05", "06"],
    },
    {
        "term": "BOCPD (Bayesian Online Changepoint Detection)",
        "what": (
            "An *online* changepoint detection method that processes a time series one "
            "observation at a time, maintaining a posterior over the location of the most "
            "recent changepoint and declaring a changepoint as soon as the maximum a "
            "posteriori (MAP) estimate places a break within the series (τ>0)."
        ),
        "why": (
            "Unlike AMOC (which waits for the whole series) BOCPD needs no pre-computed "
            "critical value and can raise an alarm during monitoring. It runs in a single "
            "pass over the difference series and stops the moment the first τ>0 break "
            "enters the MAP path."
        ),
        "plain_english": (
            "Imagine reading a stream gauge every month and, after each reading, updating "
            'your best guess of "when did the river most recently change behaviour?" '
            "While that guess sits at the very start of the record, nothing has changed; "
            "the moment it jumps to a recent month, BOCPD declares a changepoint. It "
            "decides on the fly rather than looking back at the end of the study."
        ),
        "resources": [
            '[Fearnhead & Liu (2007) "On-line inference for multiple changepoint problems"](https://doi.org/10.1111/j.1467-9868.2007.00601.x)',
        ],
        "relevant_for": ["05", "06"],
    },
    {
        "term": "Run Length",
        "what": (
            "The number of time steps since the most recent changepoint — equivalently, "
            "t minus the location of that changepoint. BOCPD maintains a posterior over "
            "the most recent changepoint location (and hence the run length) at every step."
        ),
        "why": (
            "A changepoint is declared when the maximum a posteriori (MAP) estimate first "
            "places a break inside the series (τ>0) rather than at its very start — i.e. "
            "the MAP run length stops spanning the whole history and a fresh segment opens."
        ),
        "plain_english": (
            'If the MAP run length is 30, the method believes "nothing has changed for 30 '
            'months." When the MAP instead points to a recent break — a short run length — '
            "the method declares that a change has happened; that is the alarm."
        ),
        "resources": [],
        "relevant_for": ["05", "06"],
    },
    {
        "term": "Particle Filter",
        "what": (
            "A computational technique that approximates a complicated probability "
            'distribution with a set of weighted samples ("particles"). BOCPD uses one to '
            "track the run-length posterior efficiently as new data arrives."
        ),
        "why": (
            "Tracking every possible changepoint history exactly becomes too expensive as "
            "the series grows. The particle filter keeps only a fixed number of the most "
            "plausible hypotheses, resampling them so computation stays bounded."
        ),
        "plain_english": (
            "Instead of considering every possible story of when the change happened, the "
            'method keeps a manageable crowd of "candidate stories", giving each a weight '
            "for how well it fits the data, and periodically drops the least likely ones."
        ),
        "resources": [
            "[Wikipedia: Particle Filter](https://en.wikipedia.org/wiki/Particle_filter)",
        ],
        "relevant_for": ["05", "06"],
    },
    {
        "term": "Minimum Segment Length (`msl`)",
        "what": (
            "The smallest number of consecutive observations BOCPD requires between two "
            "changepoints."
        ),
        "why": (
            "Without a minimum, the detector could place changepoints almost every step "
            "and chase noise. Enforcing a minimum segment length stabilises the estimates "
            "and prevents spurious, rapid-fire detections."
        ),
        "plain_english": (
            'It tells the method: "don\'t declare two changes closer together than `msl` months."'
        ),
        "resources": [],
        "relevant_for": ["05", "06"],
    },
    {
        "term": "Run-Length Hazard Rate (`ptr`)",
        "what": (
            "The prior probability, at each step, that the current segment ends and a new "
            "changepoint begins."
        ),
        "why": (
            "It encodes how often we expect changes *before* seeing the data. A higher "
            "hazard makes BOCPD quicker to declare changes (more sensitive, more false "
            "alarms); a lower hazard makes it more conservative."
        ),
        "plain_english": (
            'It is the method\'s built-in expectation of "how often do things change?" '
        ),
        "resources": [],
        "relevant_for": ["05", "06"],
    },
]

# Render
st.title("📚 Glossary of Terms")
st.caption("A non-technical guide to statistical concepts in rewilding analysis")

for entry in GLOSSARY:
    st.markdown(f"### **{entry['term']}**")

    body = f"**What it is:** {entry['what']}\n\n**Why it matters:** {entry['why']}"
    if entry.get("plain_english"):
        body += f"\n\n**Plain English:** {entry['plain_english']}"
    st.markdown(body)

    st.markdown("**Relevant for:**")
    for page_id in entry["relevant_for"]:
        path, label = PAGES[page_id]
        st.page_link(path, label=label, icon=":material/open_in_new:")

    if entry.get("resources"):
        res_lines = "\n".join(f"- {r}" for r in entry["resources"])
        st.markdown(f"**Resources:**\n{res_lines}")
    else:
        st.markdown("**Resources:** `TBD`")

    st.divider()
