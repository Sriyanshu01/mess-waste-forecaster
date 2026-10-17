import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Mess Waste Forecaster", page_icon="🍛", layout="wide")

LAGS = [1, 2, 3, 7, 14]
FEATURES = ["dow", "month", "is_weekend"] + [f"lag_{l}" for l in LAGS] + ["roll7", "roll14"]
MIN_DAYS = 30
BASELINE = "Seasonal Naive (baseline)"
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------- data helpers
def load_sample():
    """Synthetic demo data (weekly pattern + slow downward trend + noise)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=120)
    weekly = np.array([0, -2, -1, 0, 3, 8, 6])  # Mon..Sun extra waste
    base = 40 + weekly[dates.dayofweek] + np.linspace(0, -5, len(dates))
    waste = base + rng.normal(0, 3, len(dates))
    return pd.DataFrame({"date": dates, "waste_kg": waste.round(1)})


def clean(df):
    """Return a continuous daily series of waste_kg and the number of filled gaps."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "date" not in df.columns or "waste_kg" not in df.columns:
        raise ValueError("Your data needs two columns named 'date' and 'waste_kg'.")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["waste_kg"] = pd.to_numeric(df["waste_kg"], errors="coerce")
    df = df.dropna(subset=["date", "waste_kg"])
    if df.empty:
        raise ValueError("No valid rows found. Check the date and waste_kg values.")
    # several rows on one date (e.g. breakfast/lunch/dinner) are added together
    s = df.groupby(df["date"].dt.normalize())["waste_kg"].sum().sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    missing = len(full) - len(s)
    s = s.reindex(full).interpolate().bfill().ffill()
    s.index.name = "date"
    s.name = "waste_kg"
    return s, missing


# ---------------------------------------------------------------- ML helpers
def make_features(s):
    df = pd.DataFrame({"y": s})
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    for l in LAGS:
        df[f"lag_{l}"] = df["y"].shift(l)
    df["roll7"] = df["y"].shift(1).rolling(7).mean()
    df["roll14"] = df["y"].shift(1).rolling(14).mean()
    return df


def get_models():
    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
        ),
    }


def fit(model, s):
    d = make_features(s).dropna()
    model.fit(d[FEATURES], d["y"])
    return model


def recursive_forecast(model, s, horizon):
    """Predict one day, append it, use it to predict the next day, and so on."""
    hist = s.copy()
    for _ in range(horizon):
        nxt = hist.index[-1] + pd.Timedelta(days=1)
        hist.loc[nxt] = np.nan
        x = make_features(hist)[FEATURES].iloc[[-1]]
        hist.loc[nxt] = max(0.0, float(model.predict(x)[0]))
    return hist.iloc[-horizon:]


def seasonal_naive(s, horizon):
    """Baseline: each future day = same weekday last week."""
    last_week = s.iloc[-7:].values
    idx = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=horizon)
    return pd.Series([last_week[i % 7] for i in range(horizon)], index=idx)


@st.cache_data(show_spinner=False)
def evaluate(s, test_days):
    """Hold out the last `test_days`, forecast them, and score every model."""
    train, test = s.iloc[:-test_days], s.iloc[-test_days:]
    preds = {BASELINE: seasonal_naive(train, test_days)}
    for name, m in get_models().items():
        fit(m, train)
        preds[name] = recursive_forecast(m, train, test_days)
    rows = {}
    for name, p in preds.items():
        p.index = test.index
        mae = mean_absolute_error(test, p)
        wape = np.abs(test - p).sum() / test.sum() * 100 if test.sum() > 0 else np.nan
        rows[name] = {"Avg error (kg/day)": round(mae, 2), "Error %": round(wape, 1)}
    scores = pd.DataFrame(rows).T.sort_values("Avg error (kg/day)")
    return scores, preds, test


# ---------------------------------------------------------------- finance helpers
def npv(rate, cashflows):
    """cashflows[0] is year 0 (today)."""
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))


def irr(cashflows, lo=-0.99, hi=10.0):
    """IRR by bisection; None if the sign never changes (no IRR exists)."""
    f_lo, f_hi = npv(lo, cashflows), npv(hi, cashflows)
    if f_lo * f_hi > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid, cashflows)
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def payback(cashflows, rate=0.0):
    """Years until cumulative (discounted if rate>0) cash flow turns positive; None if never."""
    cum = cashflows[0]
    for t in range(1, len(cashflows)):
        cf = cashflows[t] / (1 + rate) ** t
        if cum + cf >= 0:
            return t - 1 + (-cum / cf if cf else 0)
        cum += cf
    return None


def project_cashflows(daily_kg, cost_kg, days_yr, reduction, capex, opex, years, inflation):
    """Year-0 investment, then yearly savings (waste avoided x food cost, cost rising with inflation) minus running cost."""
    flows = [-capex]
    for t in range(1, years + 1):
        saving = daily_kg * days_yr * reduction * cost_kg * (1 + inflation) ** (t - 1)
        flows.append(saving - opex * (1 + inflation) ** (t - 1))
    return flows


def inr(x):
    """Indian-style rupee formatting with lakh / crore for big numbers."""
    a = abs(x)
    sign = "-" if x < 0 else ""
    if a >= 1e7:
        return f"{sign}₹{a / 1e7:.2f} Cr"
    if a >= 1e5:
        return f"{sign}₹{a / 1e5:.2f} L"
    return f"{sign}₹{a:,.0f}"


# ---------------------------------------------------------------- UI
st.title("🍛 Hostel Mess Food Waste Forecaster")
st.caption(
    "Forecast food waste with machine learning, then turn it into money: what waste costs the mess, "
    "whether a waste-reduction investment pays off (NPV, IRR, payback), and how the mess performs "
    "against a waste budget."
)

with st.sidebar:
    st.header("Forecast settings")
    horizon = st.slider("Days to forecast", 7, 30, 14)
    st.header("Mess economics")
    st.caption("Example values. Replace them with your mess's real figures.")
    cost_per_kg = st.number_input("Cost of food (₹ per kg cooked)", 0.0, 1000.0, 60.0, step=5.0)
    students = st.number_input("Students eating in the mess", 1, 20000, 500, step=50)
    fee = st.number_input("Mess fee per student (₹ per month)", 0, 50000, 4000, step=250)
    days_yr = st.number_input("Days the mess runs per year", 1, 366, 300, step=5,
                              help="Hostels usually close for vacations, so this is often well below 365.")

src = st.radio(
    "How do you want to give the data?",
    ["Upload CSV / Excel", "Type it in", "Try sample data"],
    horizontal=True,
)

raw = None
if src == "Upload CSV / Excel":
    c1, c2 = st.columns([3, 1])
    with c1:
        f = st.file_uploader("File with two columns: date, waste_kg", type=["csv", "xlsx"])
    with c2:
        st.write("")
        st.download_button(
            "Download template",
            "date,waste_kg\n01/07/2026,42.5\n02/07/2026,38.0\n03/07/2026,40.2\n",
            "waste_template.csv",
            "text/csv",
        )
    if f is not None:
        raw = pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)
elif src == "Type it in":
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=MIN_DAYS - 1)
    default = pd.DataFrame(
        {"date": pd.date_range(start, periods=MIN_DAYS).date, "waste_kg": [np.nan] * MIN_DAYS}
    )
    st.caption(f"Fill at least {MIN_DAYS} days. Add rows at the bottom of the table for more.")
    raw = st.data_editor(
        default,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "date": st.column_config.DateColumn("date", format="DD/MM/YYYY"),
            "waste_kg": st.column_config.NumberColumn("waste_kg", min_value=0.0, format="%.1f"),
        },
    )
else:
    raw = load_sample()
    st.info("Showing synthetic demo data (made up for illustration), not real mess records.")

if raw is None:
    st.stop()
if "waste_kg" in raw.columns and pd.to_numeric(raw["waste_kg"], errors="coerce").notna().sum() == 0:
    st.info("Enter your waste figures above to see the forecast.")
    st.stop()

try:
    s, missing = clean(raw)
except ValueError as e:
    st.error(str(e))
    st.stop()

if len(s) < MIN_DAYS:
    st.warning(f"Only {len(s)} days of data so far. Please give at least {MIN_DAYS} days.")
    st.stop()
if missing:
    st.caption(f"Note: {missing} missing day(s) were filled in by interpolation.")

# ---------------------------------------------------------------- core computation
test_days = max(7, min(14, len(s) // 5))
with st.spinner("Training and comparing models..."):
    scores, preds, test = evaluate(s, test_days)
best = scores.index[0]

if best == BASELINE:
    fc = seasonal_naive(s, horizon)
else:
    fc = recursive_forecast(fit(get_models()[best], s), s, horizon)

resid_sd = float((test - preds[best]).std())
low = (fc - 1.28 * resid_sd).clip(lower=0)
high = fc + 1.28 * resid_sd
daily_fc = float(fc.mean())                     # expected kg wasted per day going forward

# mess-level economics (monthly = 30 days)
annual_waste_kg = daily_fc * days_yr
annual_waste_cost = annual_waste_kg * cost_per_kg
monthly_revenue = students * fee
monthly_waste_cost = daily_fc * 30 * cost_per_kg
waste_pct_rev = monthly_waste_cost / monthly_revenue * 100 if monthly_revenue else np.nan
waste_per_student_yr = annual_waste_cost / students
grams_per_student_day = daily_fc * 1000 / students

tab_fc, tab_fin, tab_inv, tab_bud, tab_how = st.tabs(
    ["📈 Forecast", "💰 Cost of waste", "🏦 Investment appraisal", "🎯 Budget & variance", "ℹ️ Method"]
)

# ================================================================ TAB 1: forecast
with tab_fc:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Days of data", len(s))
    m2.metric("Avg waste / day", f"{s.mean():.1f} kg")
    m3.metric("Total waste so far", f"{s.sum():,.0f} kg")
    m4.metric("Value wasted so far", inr(s.sum() * cost_per_kg))

    st.subheader(f"Forecast for the next {horizon} days")
    st.caption(f"Best model on your data: **{best}** (tested on the last {test_days} days you entered).")

    hist_view = s.iloc[-60:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_view.index, y=hist_view.values, name="Past waste", line=dict(color="#4C78A8")))
    fig.add_trace(go.Scatter(x=high.index, y=high.values, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=low.index, y=low.values, fill="tonexty", fillcolor="rgba(245,133,24,0.2)",
                             line=dict(width=0), name="Likely range (~80%)"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc.values, name="Forecast", line=dict(color="#F58518", width=3)))
    fig.update_layout(height=420, yaxis_title="Food waste (kg)", hovermode="x unified", margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Expected waste (forecast period)", f"{fc.sum():,.0f} kg")
    c2.metric("Expected value wasted", inr(fc.sum() * cost_per_kg))
    c3.metric("Expected avg / day", f"{daily_fc:.1f} kg",
              delta=f"{daily_fc - s.iloc[-14:].mean():+.1f} kg vs last 2 weeks", delta_color="inverse")

    dow_avg = s.groupby(s.index.dayofweek).mean()
    peak, calm = DAY_NAMES[dow_avg.idxmax()], DAY_NAMES[dow_avg.idxmin()]
    st.info(
        f"Waste is highest on **{peak}s** ({dow_avg.max():.1f} kg on average) and lowest on "
        f"**{calm}s** ({dow_avg.min():.1f} kg). Planning smaller batches for {peak} meals is the "
        "first place to look for savings."
    )

    table = pd.DataFrame({
        "Date": fc.index.strftime("%d %b %Y"),
        "Day": [DAY_NAMES[d] for d in fc.index.dayofweek],
        "Expected waste (kg)": fc.values.round(1),
        "Low (kg)": low.values.round(1),
        "High (kg)": high.values.round(1),
        "Value (₹)": (fc.values * cost_per_kg).round(0),
    })
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button("Download forecast (CSV)", table.to_csv(index=False), "waste_forecast.csv", "text/csv")

    with st.expander("Model comparison"):
        st.dataframe(scores, width="stretch")
        st.caption("Each model forecast the last few days of your data without seeing them; lower error is "
                   "better. The baseline simply repeats last week, so ML is only worth using if it beats it.")

# ================================================================ TAB 2: cost of waste
with tab_fin:
    st.subheader("What food waste costs the mess")
    st.caption("Annual figures scale the forecast daily average by the days the mess runs per year.")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Projected annual waste cost", inr(annual_waste_cost))
    a2.metric("Waste as % of mess fee revenue", f"{waste_pct_rev:.1f}%")
    a3.metric("Waste cost per student / year", inr(waste_per_student_yr))
    a4.metric("Waste per student / day", f"{grams_per_student_day:.0f} g")

    st.markdown("#### Scenario analysis")
    st.caption("Worst and best cases use the edges of the forecast range, so they reflect how uncertain the forecast is.")
    scen = pd.DataFrame({
        "Scenario": ["Best case", "Base case", "Worst case"],
        "Waste / day (kg)": [low.mean(), daily_fc, high.mean()],
    })
    scen["Annual waste (tonnes)"] = scen["Waste / day (kg)"] * days_yr / 1000
    scen["Annual cost (₹)"] = scen["Waste / day (kg)"] * days_yr * cost_per_kg
    scen["% of mess revenue"] = scen["Waste / day (kg)"] * 30 * cost_per_kg / monthly_revenue * 100
    show = scen.copy()
    show["Waste / day (kg)"] = show["Waste / day (kg)"].round(1)
    show["Annual waste (tonnes)"] = show["Annual waste (tonnes)"].round(1)
    show["Annual cost (₹)"] = show["Annual cost (₹)"].map(inr)
    show["% of mess revenue"] = show["% of mess revenue"].round(1).astype(str) + "%"
    st.dataframe(show, width="stretch", hide_index=True)

    st.markdown("#### Waste-at-Risk (like Value-at-Risk)")
    z95 = 1.645
    war_kg = fc.sum() + z95 * resid_sd * np.sqrt(horizon)
    w1, w2 = st.columns(2)
    w1.metric(f"Expected waste cost, next {horizon} days", inr(fc.sum() * cost_per_kg))
    w2.metric("95% Waste-at-Risk", inr(war_kg * cost_per_kg),
              delta=f"+{inr((war_kg - fc.sum()) * cost_per_kg)} above expected", delta_color="off")
    st.caption(
        f"There is roughly a 95% chance waste cost over the next {horizon} days stays below this figure. "
        "Approximation: it assumes daily forecast errors are independent and normally distributed, "
        "which multi-day forecasts only roughly satisfy."
    )

    st.markdown("#### Where the money goes, by weekday")
    dow_cost = (s.groupby(s.index.dayofweek).mean() * cost_per_kg).rename(index=dict(enumerate(DAY_NAMES)))
    bar = go.Figure(go.Bar(x=dow_cost.index, y=dow_cost.values, marker_color="#4C78A8",
                           text=[inr(v) for v in dow_cost.values], textposition="outside"))
    bar.update_layout(height=320, yaxis_title="Avg waste cost per day (₹)", margin=dict(t=20))
    st.plotly_chart(bar, width="stretch")

# ================================================================ TAB 3: investment appraisal
with tab_inv:
    st.subheader("Is a waste-reduction investment worth it?")
    st.caption(
        "Appraise any fix (a meal pre-booking app, smart weighing scales, portion control, a biogas unit...) "
        "like a capital budgeting project. All inputs below are example assumptions; replace them with real "
        "vendor quotes and a realistic reduction estimate."
    )
    i1, i2, i3 = st.columns(3)
    with i1:
        proj = st.text_input("Intervention name", "Meal pre-booking app + portion control")
        capex = st.number_input("Upfront investment (₹)", 0, 100_000_000, 150_000, step=10_000)
        opex = st.number_input("Running cost per year (₹)", 0, 50_000_000, 30_000, step=5_000)
    with i2:
        reduction = st.slider("Expected waste reduction (%)", 0, 80, 25) / 100
        years = st.slider("Project life (years)", 1, 10, 5)
    with i3:
        rate = st.slider("Discount rate / cost of capital (%)", 0.0, 25.0, 10.0, 0.5) / 100
        infl = st.slider("Food cost inflation per year (%)", 0.0, 15.0, 5.0, 0.5) / 100

    flows = project_cashflows(daily_fc, cost_per_kg, days_yr, reduction, capex, opex, years, infl)
    p_npv = npv(rate, flows)
    p_irr = irr(flows)
    p_pb = payback(flows)
    p_dpb = payback(flows, rate)
    pv_in = sum(cf / (1 + rate) ** t for t, cf in enumerate(flows) if t > 0)
    pi = pv_in / capex if capex else np.nan

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("NPV", inr(p_npv))
    k2.metric("IRR", f"{p_irr * 100:.1f}%" if p_irr is not None else "n/a")
    k3.metric("Payback", f"{p_pb:.1f} yrs" if p_pb is not None else "Not within life")
    k4.metric("Discounted payback", f"{p_dpb:.1f} yrs" if p_dpb is not None else "Not within life")
    k5.metric("Profitability index", f"{pi:.2f}" if np.isfinite(pi) else "n/a")

    if p_npv > 0:
        st.success(f"**Accept.** {proj} adds {inr(p_npv)} of value at a {rate * 100:.1f}% cost of capital"
                   + (f", and its IRR of {p_irr * 100:.1f}% beats that hurdle." if p_irr is not None else "."))
    else:
        st.error(f"**Reject at these assumptions.** {proj} destroys {inr(-p_npv)} of value at a "
                 f"{rate * 100:.1f}% cost of capital. Check the break-even point below.")

    cf_tbl = pd.DataFrame({
        "Year": list(range(years + 1)),
        "Net cash flow (₹)": [round(x) for x in flows],
        "Discount factor": [round(1 / (1 + rate) ** t, 4) for t in range(years + 1)],
    })
    cf_tbl["Present value (₹)"] = (cf_tbl["Net cash flow (₹)"] * cf_tbl["Discount factor"]).round(0)
    cf_tbl["Cumulative PV (₹)"] = cf_tbl["Present value (₹)"].cumsum()
    st.markdown("#### Cash flow schedule")
    st.dataframe(cf_tbl, width="stretch", hide_index=True)

    # break-even reduction %: NPV is linear in reduction, so solve directly
    npv0 = npv(rate, project_cashflows(daily_fc, cost_per_kg, days_yr, 0, capex, opex, years, infl))
    npv1 = npv(rate, project_cashflows(daily_fc, cost_per_kg, days_yr, 1, capex, opex, years, infl))
    be = -npv0 / (npv1 - npv0) if npv1 != npv0 else np.nan
    st.info(f"**Break-even:** the project needs to cut waste by at least **{be * 100:.1f}%** for NPV to reach zero."
            if np.isfinite(be) and 0 <= be <= 1 else
            "**Break-even:** even a 100% waste cut would not recover this investment at these costs.")

    st.markdown("#### Sensitivity analysis")
    sc1, sc2 = st.columns(2)
    with sc1:
        base_inputs = dict(daily=daily_fc, cost=cost_per_kg, red=reduction, capex=capex, opex=opex, rate=rate)
        labels = {"daily": "Waste forecast (kg/day)", "cost": "Food cost (₹/kg)", "red": "Waste reduction %",
                  "capex": "Upfront investment", "opex": "Running cost", "rate": "Discount rate"}
        rows = []
        for k in labels:
            out = []
            for m in (0.8, 1.2):
                v = dict(base_inputs)
                v[k] = v[k] * m
                out.append(npv(v["rate"], project_cashflows(v["daily"], v["cost"], days_yr, v["red"],
                                                            v["capex"], v["opex"], years, infl)))
            rows.append((labels[k], out[0] - p_npv, out[1] - p_npv))
        rows.sort(key=lambda r: abs(r[2] - r[1]))
        tor = go.Figure()
        tor.add_trace(go.Bar(y=[r[0] for r in rows], x=[r[1] for r in rows], orientation="h",
                             name="Input −20%", marker_color="#E45756"))
        tor.add_trace(go.Bar(y=[r[0] for r in rows], x=[r[2] for r in rows], orientation="h",
                             name="Input +20%", marker_color="#54A24B"))
        tor.update_layout(barmode="overlay", height=340, title="Tornado: change in NPV (₹) when each input moves ±20%",
                          margin=dict(t=50, l=10), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(tor, width="stretch")
    with sc2:
        reds = np.arange(5, 55, 5)
        costs = np.round(cost_per_kg * np.array([0.6, 0.8, 1.0, 1.2, 1.4]))
        grid = [[npv(rate, project_cashflows(daily_fc, c, days_yr, r / 100, capex, opex, years, infl)) / 1e5
                 for c in costs] for r in reds]
        hm = go.Figure(go.Heatmap(
            z=grid, x=[f"₹{int(c)}" for c in costs], y=[f"{r}%" for r in reds],
            colorscale="RdYlGn", zmid=0, text=[[f"{v:.1f}" for v in row] for row in grid],
            texttemplate="%{text}", colorbar=dict(title="NPV (₹ L)")))
        hm.update_layout(height=340, title="NPV (₹ lakh): waste reduction % vs food cost per kg",
                         xaxis_title="Food cost per kg", yaxis_title="Waste reduction", margin=dict(t=50))
        st.plotly_chart(hm, width="stretch")

# ================================================================ TAB 4: budget & variance
with tab_bud:
    st.subheader("Waste budget and variance analysis")
    st.caption("Set a target the way a management accountant sets a standard cost, then check the forecast against it.")
    recent = float(s.iloc[-30:].mean())
    target = st.number_input("Target waste (kg per day)", 0.0, 10000.0, round(recent * 0.85, 1), step=0.5,
                             help="Default is 15% below the last 30 days' average. Set your own goal.")
    budget_cost = target * cost_per_kg
    var = pd.DataFrame({
        "Date": fc.index.strftime("%d %b"),
        "Day": [DAY_NAMES[d][:3] for d in fc.index.dayofweek],
        "Budget (kg)": target,
        "Forecast (kg)": fc.values.round(1),
    })
    var["Variance (kg)"] = (var["Budget (kg)"] - var["Forecast (kg)"]).round(1)
    var["Variance (₹)"] = (var["Variance (kg)"] * cost_per_kg).round(0)
    var["F / A"] = np.where(var["Variance (kg)"] >= 0, "Favourable", "Adverse")

    tot_var = var["Variance (₹)"].sum()
    b1, b2, b3 = st.columns(3)
    b1.metric(f"Budgeted waste cost ({horizon} days)", inr(budget_cost * horizon))
    b2.metric("Forecast waste cost", inr(fc.sum() * cost_per_kg))
    b3.metric("Total variance", inr(abs(tot_var)), delta="Favourable" if tot_var >= 0 else "Adverse",
              delta_color="normal" if tot_var >= 0 else "inverse")
    n_adv = int((var["F / A"] == "Adverse").sum())
    st.write(f"The forecast breaches the target on **{n_adv} of {horizon} days**.")

    vfig = go.Figure(go.Bar(x=var["Date"] + " " + var["Day"], y=var["Variance (₹)"],
                            marker_color=np.where(var["Variance (₹)"] >= 0, "#54A24B", "#E45756")))
    vfig.update_layout(height=320, yaxis_title="Variance (₹): green favourable, red adverse", margin=dict(t=20))
    st.plotly_chart(vfig, width="stretch")
    st.dataframe(var, width="stretch", hide_index=True)

    st.markdown("#### Forecast accuracy (actual vs forecast on the test days)")
    bt = pd.DataFrame({"Actual (kg)": test.values, "Forecast (kg)": preds[best].values.round(1)}, index=test.index)
    bt["Error (kg)"] = (bt["Actual (kg)"] - bt["Forecast (kg)"]).round(1)
    bias = bt["Error (kg)"].mean()
    st.write(
        f"On the {test_days} hidden test days the model was off by {scores.loc[best, 'Avg error (kg/day)']} kg/day "
        f"on average ({scores.loc[best, 'Error %']}%). Bias: it tends to "
        f"**{'under' if bias > 0 else 'over'}-forecast** by {abs(bias):.1f} kg/day."
    )
    bt.index = bt.index.strftime("%d %b")
    st.dataframe(bt, width="stretch")

# ================================================================ TAB 5: method
with tab_how:
    st.markdown(
        """
**Forecasting (machine learning)**
- Inputs the models learn from: day of week, month, weekend flag, waste 1/2/3/7/14 days earlier, 7- and 14-day averages.
- Models tried: Ridge Regression, Random Forest, Gradient Boosting, plus a seasonal-naive baseline.
- The last part of your data is hidden, each model forecasts it, and the lowest-error model wins. It is then retrained on all data.
- Multi-day forecasts feed each prediction back in as the next day's input (recursive forecasting).

**Finance & management**
- *Cost of waste:* forecast kg × food cost per kg, scaled to a year by the days the mess runs.
- *Scenario analysis:* best / base / worst cases from the forecast's uncertainty range.
- *Waste-at-Risk:* the Value-at-Risk idea applied to waste cost, at 95% confidence (normal approximation).
- *Capital budgeting:* NPV, IRR, payback, discounted payback and profitability index for a waste-reduction investment.
  Savings = waste avoided × food cost (rising with inflation) − running cost.
- *Sensitivity:* a tornado chart (±20% on each input) and an NPV grid; break-even waste reduction solved exactly.
- *Budgetary control:* daily waste target vs forecast, with favourable / adverse variances in kg and ₹.

**Assumptions to state honestly:** all mess-economics and investment inputs are user-entered examples, not
researched figures; a month is taken as 30 days; the forecast's average daily waste is assumed to hold for the
year when annualising.
"""
    )
