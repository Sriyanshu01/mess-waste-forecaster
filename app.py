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


def get_models(n_trees=300):
    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=2, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=n_trees, learning_rate=0.05, max_depth=3, random_state=42
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
def evaluate(s, test_days, n_trees=300):
    """Hold out the last `test_days`, forecast them, and score every model."""
    train, test = s.iloc[:-test_days], s.iloc[-test_days:]
    preds = {BASELINE: seasonal_naive(train, test_days)}
    for name, m in get_models(n_trees).items():
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


# ================================================================ PAGE 1: overall
TEAM = ["Pooja Karwa", "Samridhi Aggarwal", "Sriyanshu Mishra"]
INSTITUTE = "Great Lakes Institute of Management (GLIM), Chennai"


def show_credits():
    """Project credit under the page title and at the top of the sidebar."""
    st.markdown(
        f"<div style='margin:-0.6rem 0 0.8rem 0;font-size:0.95rem;opacity:0.85'>"
        f"A project by <b>{', '.join(TEAM[:-1])} &amp; {TEAM[-1]}</b> · {INSTITUTE}</div>",
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.markdown(
            "<div style='padding:0.6rem 0.8rem;border-radius:0.5rem;border:1px solid rgba(128,128,128,0.35);"
            "font-size:0.85rem;line-height:1.5'><b>Project team</b><br>" + "<br>".join(TEAM) +
            f"<br><span style='opacity:0.75'>GLIM Chennai</span></div>",
            unsafe_allow_html=True,
        )


def page_overall():
    st.title("🍛 Hostel Mess Food Waste Forecaster")
    show_credits()
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

    tab_fc, tab_fin, tab_inv, tab_bud, tab_rec, tab_how = st.tabs(
        ["📈 Forecast", "💰 Cost of waste", "🏦 Investment appraisal", "🎯 Budget & variance",
         "💡 Recommendations", "ℹ️ Method"]
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
    # ================================================================ TAB 5: recommendations
    with tab_rec:
        st.subheader("Recommendations")
        st.caption("Generated from your data and the inputs on the other tabs, using fixed rules (thresholds listed "
                   "at the bottom). They point to where to act, and still need the mess manager's judgement.")
        recs = []   # (priority 1 high / 2 medium / 3 good news, text)

        # 1. weekday pattern -> batch sizing
        others = dow_avg.drop(dow_avg.idxmax()).mean()
        if others > 0 and dow_avg.max() >= 1.15 * others:
            extra = dow_avg.max() - others
            yr_save = extra * (days_yr / 7) * cost_per_kg
            recs.append((1 if extra / others >= 0.3 else 2,
                         f"**Cook less on {peak}s.** Waste averages {dow_avg.max():.1f} kg on {peak}s vs "
                         f"{others:.1f} kg on other days, likely because fewer students eat in. Cutting the {peak} "
                         f"batch to bring waste back to normal would save about {inr(yr_save)} a year."))

        # 2. recent trend
        last14, prev14 = s.iloc[-14:].mean(), s.iloc[-28:-14].mean()
        if prev14 > 0:
            chg = (last14 / prev14 - 1) * 100
            if chg >= 10:
                recs.append((1 if chg >= 25 else 2,
                             f"**Waste is rising:** up {chg:.0f}% in the last two weeks ({prev14:.1f} → {last14:.1f} kg/day). "
                             "Check for menu changes, food quality complaints or a drop in attendance."))
            elif chg <= -10:
                recs.append((3, f"**Waste is falling:** down {abs(chg):.0f}% in the last two weeks "
                                f"({prev14:.1f} → {last14:.1f} kg/day). Find out what changed and keep doing it."))

        # 3. forecast direction
        f_chg = (daily_fc / last14 - 1) * 100 if last14 > 0 else 0
        if f_chg >= 10:
            recs.append((2, f"**Expect more waste ahead:** the forecast averages {daily_fc:.1f} kg/day, {f_chg:.0f}% "
                            f"above the last two weeks. Plan smaller batches for the next {horizon} days."))

        # 4. budget variance
        if n_adv > horizon / 2:
            recs.append((1, f"**Over the waste budget:** the forecast misses the {target:.1f} kg/day target on "
                            f"{n_adv} of {horizon} days (adverse variance {inr(abs(tot_var))}). Reaching it needs about "
                            f"{max(0.0, daily_fc - target):.1f} kg/day less waste."))
        elif n_adv == 0:
            recs.append((3, f"**On budget:** the forecast stays within the {target:.1f} kg/day target on every day. "
                            "Consider setting a tighter target."))

        # 5. investment decision
        if p_npv > 0:
            recs.append((1 if p_npv > capex else 2,
                         f"**Invest in '{proj}'.** At your assumptions it has NPV {inr(p_npv)}"
                         + (f", IRR {p_irr * 100:.1f}%" if p_irr is not None else "")
                         + (f" and pays back in {p_pb:.1f} years" if p_pb is not None else "")
                         + (f". It stays worthwhile as long as it cuts waste by at least {be * 100:.1f}%."
                            if np.isfinite(be) and 0 <= be <= 1 else ".")
                         + " Verify the cost and reduction assumptions with a real vendor quote before committing."))
        else:
            recs.append((2, f"**Don't invest in '{proj}' at these assumptions:** NPV is {inr(p_npv)}. "
                            + (f"It would need to cut waste by at least {be * 100:.1f}% to break even. "
                               if np.isfinite(be) and 0 <= be <= 1 else "")
                            + "Try a cheaper option, such as portion control or pre-booking via a free form."))

        # 6. scale of the problem
        if np.isfinite(waste_pct_rev) and waste_pct_rev >= 5:
            recs.append((1, f"**Waste is a material cost:** about {waste_pct_rev:.1f}% of mess fee revenue "
                            f"({inr(annual_waste_cost)} a year, {inr(waste_per_student_yr)} per student). "
                            "Report it monthly to the mess committee as a key metric."))
        else:
            recs.append((3, f"**Waste cost is {waste_pct_rev:.1f}% of mess fee revenue** ({inr(annual_waste_cost)} a year). "
                            "Still worth tracking monthly so it doesn't creep up."))

        # 7. volatility -> pre-booking
        r30 = s.iloc[-30:]
        cv = r30.std() / r30.mean() if r30.mean() > 0 else 0
        if cv >= 0.25:
            recs.append((2, f"**Waste is unpredictable** (it varies by {cv * 100:.0f}% around its average). "
                            "A meal pre-booking or opt-out system would tell the kitchen how many will eat each day."))

        # 8. spike days
        z = (s - s.rolling(28, min_periods=7).mean()) / s.rolling(28, min_periods=7).std()
        spikes = z[z >= 2].sort_values(ascending=False).head(3)
        if len(spikes):
            days_txt = ", ".join(f"{d:%d %b} ({s[d]:.0f} kg)" for d in spikes.index)
            recs.append((2, f"**Unusual spike days:** {days_txt}. Check what happened (fests, special meals, exams, "
                            "holidays). If they recur, plan for them in advance."))

        # 9. risk buffer
        recs.append((3, f"**Risk buffer:** budget for up to {inr(war_kg * cost_per_kg)} of waste over the next "
                        f"{horizon} days (95% Waste-at-Risk), {inr((war_kg - fc.sum()) * cost_per_kg)} above the expected cost."))

        # 10. forecast reliability
        if scores.loc[best, "Error %"] >= 15:
            recs.append((2, f"**Forecast is rough** ({scores.loc[best, 'Error %']}% error). Recording daily headcount "
                            "and the menu alongside waste would make predictions much sharper."))

        labels = {1: ("🔴 High priority", st.error), 2: ("🟠 Medium priority", st.warning),
                  3: ("🟢 Good news & monitoring", st.success)}
        for p in (1, 2, 3):
            group = [r for r in recs if r[0] == p]
            if group:
                st.markdown(f"#### {labels[p][0]}")
                for _, txt in group:
                    labels[p][1](txt)

        st.markdown("#### Also look at")
        st.info("For dish-level advice (which items to cut and how much to cook), open **Item-wise forecast** "
                "in the sidebar and upload item-level records.")
        with st.expander("Rule thresholds"):
            st.markdown("- Weekday problem: one weekday ≥ 1.15× the other days' average (high priority if ≥ 1.3×)\n"
                        "- Trend: last 2 weeks vs previous 2 weeks, ±10% (high priority if ≥ +25%)\n"
                        "- Over budget: forecast misses the target on more than half the days\n"
                        "- Material cost: waste ≥ 5% of mess fee revenue\n"
                        "- Unpredictable: coefficient of variation over the last 30 days ≥ 25%\n"
                        "- Spike day: ≥ 2 standard deviations above the 28-day rolling average\n"
                        "- Rough forecast: model error ≥ 15%\n\n"
                        "These cut-offs are judgement calls, not industry standards. Adjust them in the code if needed.")

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

# ================================================================ PAGE 2: item-wise
ITEM_MIN_DAYS = 28
ITEM_TREES = 150


def load_item_sample(end=None):
    """Synthetic item-level data: a rotating hostel menu with prepared and wasted kg (illustrative only)."""
    rng = np.random.default_rng(11)
    end = pd.Timestamp.today().normalize() if end is None else pd.Timestamp(end)
    dates = pd.date_range(end=end, periods=150)
    # item: (meal, weekdays served or None=daily, prepared kg, base waste share, ₹/kg, weekday waste boost)
    menu = {
        "Idli": ("Breakfast", [0, 3], 40, 0.12, 45, {}),
        "Poha": ("Breakfast", [1, 4], 30, 0.18, 40, {}),
        "Aloo Paratha": ("Breakfast", [2, 5, 6], 35, 0.10, 55, {6: 0.10}),
        "Rice": ("Lunch", None, 90, 0.14, 50, {5: 0.08, 6: 0.10}),
        "Dal": ("Lunch", None, 45, 0.20, 70, {5: 0.10, 6: 0.12}),
        "Sambar": ("Lunch", [0, 2, 4], 40, 0.22, 60, {}),
        "Mixed Veg Curry": ("Lunch", None, 35, 0.30, 75, {5: 0.10, 6: 0.10}),
        "Curd": ("Lunch", None, 25, 0.06, 80, {}),
        "Chapati": ("Dinner", None, 50, 0.10, 65, {}),
        "Paneer Curry": ("Dinner", [4, 6], 30, 0.07, 220, {}),
        "Chicken Curry": ("Dinner", [2, 6], 35, 0.05, 260, {}),
        "Kheer": ("Dinner", [5, 6], 20, 0.15, 90, {}),
    }
    rows = []
    for i, d in enumerate(dates):
        for item, (meal, days, prep, share, cost, boost) in menu.items():
            if days is not None and d.dayofweek not in days:
                continue
            p = prep * rng.normal(1, 0.05)
            w_share = share + boost.get(d.dayofweek, 0)
            if item == "Chapati":
                w_share += 0.12 * i / len(dates)          # waste creeping up over time
            if item == "Mixed Veg Curry":
                w_share *= rng.choice([0.6, 1.0, 1.5])    # very inconsistent
            w = max(0.0, p * w_share * rng.normal(1, 0.12))
            rows.append((d.strftime("%d/%m/%Y"), meal, item, round(p, 1), round(w, 1), cost))
    return pd.DataFrame(rows, columns=["date", "meal", "item", "prepared_kg", "waste_kg", "cost_per_kg"])


def clean_items(df):
    """Accepts long format (date, item, waste_kg [+ prepared_kg, cost_per_kg, meal])
    or wide format (date + one column per item). Returns daily item matrices."""
    df = df.copy()
    known = {"date", "item", "waste_kg", "prepared_kg", "cost_per_kg", "meal"}
    norm = lambda c: str(c).strip().lower().replace(" ", "_")
    # standardise only the known column names; keep item-name columns (wide format) as written
    df.columns = [norm(c) if norm(c) in known else str(c).strip() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("Your file needs a 'date' column.")
    if "item" not in df.columns:
        value_cols = [c for c in df.columns if c != "date" and pd.to_numeric(df[c], errors="coerce").notna().any()]
        if len(value_cols) < 1:
            raise ValueError("Couldn't find item names. Use columns date, item, waste_kg "
                             "(or one column per item).")
        df = df.melt(id_vars="date", value_vars=value_cols, var_name="item", value_name="waste_kg")
    if "waste_kg" not in df.columns:
        raise ValueError("Your file needs a 'waste_kg' column.")
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["waste_kg"] = pd.to_numeric(df["waste_kg"], errors="coerce")
    df["item"] = df["item"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df = df.dropna(subset=["date", "waste_kg"])
    df = df[df["item"].ne("") & df["item"].str.lower().ne("nan")]
    if df.empty:
        raise ValueError("No valid rows found. Check the date, item and waste_kg values.")
    # case-insensitive item names: "dal" and "Dal" become one item
    first_name = df.groupby(df["item"].str.lower())["item"].first()
    df["item"] = df["item"].str.lower().map(first_name)
    df["date"] = df["date"].dt.normalize()

    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    waste = df.pivot_table(index="date", columns="item", values="waste_kg", aggfunc="sum").reindex(full)
    served = waste.notna()
    waste = waste.fillna(0.0)               # item not on the menu that day -> 0 kg wasted
    waste.index.name = "date"

    prepared = None
    if "prepared_kg" in df.columns and pd.to_numeric(df["prepared_kg"], errors="coerce").notna().any():
        df["prepared_kg"] = pd.to_numeric(df["prepared_kg"], errors="coerce")
        prepared = df.pivot_table(index="date", columns="item", values="prepared_kg", aggfunc="sum").reindex(full)
        prepared = prepared.reindex(columns=waste.columns).fillna(0.0)

    costs = None
    if "cost_per_kg" in df.columns and pd.to_numeric(df["cost_per_kg"], errors="coerce").notna().any():
        df["cost_per_kg"] = pd.to_numeric(df["cost_per_kg"], errors="coerce")
        costs = df.groupby("item")["cost_per_kg"].mean()

    meals = None
    if "meal" in df.columns:
        df["meal"] = df["meal"].astype(str).str.strip().str.title()
        meals = df.groupby(["item"])["meal"].agg(lambda x: x.mode().iat[0] if not x.mode().empty else "")

    return waste, served, prepared, costs, meals


@st.cache_data(show_spinner=False)
def item_forecast(s, served_dow, horizon):
    """Forecast one item. served_dow: weekdays (0-6) the item was served in the last 8 weeks."""
    served_share = float((s > 0).mean())
    if len(s) < ITEM_MIN_DAYS or (s > 0).sum() < 12:
        # too little history for ML: weekday average of the last 8 weeks
        recent = s.iloc[-56:]
        wd = recent.groupby(recent.index.dayofweek).mean()
        idx = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=horizon)
        fc = pd.Series([wd.get(d.dayofweek, 0.0) for d in idx], index=idx)
        method, err, sd = "Weekday average (too little data for ML)", np.nan, float(recent.std())
    else:
        test_days = max(7, min(14, len(s) // 5))
        scores, preds, test = evaluate(s, test_days, ITEM_TREES)
        method = scores.index[0]
        err = float(scores.iloc[0]["Error %"])
        sd = float((test - preds[method]).std())
        fc = seasonal_naive(s, horizon) if method == BASELINE else \
            recursive_forecast(fit(get_models(ITEM_TREES)[method], s), s, horizon)
    # not on the menu that weekday recently -> nothing to waste
    fc = fc.where([d.dayofweek in served_dow for d in fc.index], 0.0).clip(lower=0)
    return {"fc": fc, "method": method, "err": err, "sd": sd, "served_share": served_share}


def page_items():
    st.title("🥘 Item-wise Waste Forecast")
    show_credits()
    st.caption("Upload waste records per dish. The app finds every item in your file, forecasts each one, "
               "ranks where the money is going, and recommends how much to cook.")

    with st.sidebar:
        st.header("Forecast settings")
        horizon = st.slider("Days to forecast", 7, 30, 7, key="it_h")
        st.header("Economics")
        st.caption("Example values. Replace them with your mess's real figures.")
        days_yr = st.number_input("Days the mess runs per year", 1, 366, 300, step=5, key="it_days")
        default_cost = st.number_input("Default food cost (₹/kg) for items without a cost", 0.0, 2000.0, 60.0,
                                       step=5.0, key="it_cost")
        buffer = st.slider("Safety buffer when cooking (%)", 0, 30, 10, key="it_buf",
                           help="Extra food cooked above expected consumption so nobody goes hungry.") / 100

    src = st.radio("Data", ["Upload CSV / Excel", "Try sample data"], horizontal=True, key="it_src")
    raw = None
    if src == "Upload CSV / Excel":
        c1, c2 = st.columns([3, 1])
        with c1:
            f = st.file_uploader("Columns: date, item, waste_kg — optional: prepared_kg, cost_per_kg, meal",
                                 type=["csv", "xlsx"], key="it_file")
        with c2:
            st.write("")
            st.download_button(
                "Download template",
                "date,meal,item,prepared_kg,waste_kg,cost_per_kg\n"
                "01/07/2026,Lunch,Rice,90,12.5,50\n01/07/2026,Lunch,Dal,45,9.0,70\n"
                "01/07/2026,Dinner,Chapati,50,4.8,65\n02/07/2026,Lunch,Rice,88,11.0,50\n",
                "item_waste_template.csv", "text/csv", key="it_tpl")
        with st.expander("Accepted formats"):
            st.markdown(
                "- **Long format (recommended):** one row per item per day: `date, item, waste_kg`. "
                "Add `prepared_kg` to get a cooking plan, `cost_per_kg` for item-level costs, and `meal` "
                "for a breakfast/lunch/dinner split.\n"
                "- **Wide format:** `date` plus one column per item, e.g. `date, Rice, Dal, Chapati`.\n"
                "- Dates as DD/MM/YYYY. If an item is missing on a date, it's treated as **not served** (0 kg)."
            )
        if f is not None:
            raw = pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)
    else:
        raw = load_item_sample()
        st.info("Showing synthetic demo data (made up for illustration), not real mess records.")

    if raw is None:
        st.stop()
    try:
        waste, served, prepared, costs, meals = clean_items(raw)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    if len(waste) < 14:
        st.warning(f"Only {len(waste)} days of data. Please give at least 14 days (28+ for ML forecasts).")
        st.stop()

    items = list(waste.columns)

    # ---- cost per item: from file, else editable table
    if costs is None:
        with st.expander("💰 Food cost per item (₹/kg) — edit to match your mess", expanded=False):
            cost_tbl = st.data_editor(
                pd.DataFrame({"item": items, "cost_per_kg": default_cost}),
                hide_index=True, width="stretch", disabled=["item"], key="it_cost_tbl",
                column_config={"cost_per_kg": st.column_config.NumberColumn("₹ per kg", min_value=0.0)})
        costs = cost_tbl.set_index("item")["cost_per_kg"]
    costs = costs.reindex(items).fillna(default_cost)

    # ---- forecast every item
    recent_served = served.iloc[-56:]
    results = {}
    bar = st.progress(0.0, text="Forecasting each item...")
    for i, it in enumerate(items):
        dows = sorted(set(recent_served.index[recent_served[it]].dayofweek))
        s = waste[it].rename("waste_kg")
        results[it] = item_forecast(s, dows, horizon)
        bar.progress((i + 1) / len(items), text=f"Forecasting {it}...")
    bar.empty()

    # ---- item summary table
    rows = []
    for it in items:
        s, srv = waste[it], served[it]
        on = s[srv]
        last14, prev14 = s.iloc[-14:][srv.iloc[-14:]], s.iloc[-28:-14][srv.iloc[-28:-14]]
        trend = (last14.mean() / prev14.mean() - 1) * 100 if len(last14) and len(prev14) and prev14.mean() > 0 else np.nan
        r = results[it]
        fc_daily = r["fc"].mean()
        row = {
            "Item": it,
            "Meal": meals.get(it, "") if meals is not None else "",
            "Served (% of days)": round(srv.mean() * 100),
            "Avg waste when served (kg)": round(on.mean(), 1) if len(on) else 0.0,
            "₹/kg": costs[it],
            f"Forecast next {horizon}d (kg)": round(r["fc"].sum(), 1),
            f"Forecast next {horizon}d (₹)": round(r["fc"].sum() * costs[it]),
            "Annual waste cost (₹)": round(fc_daily * days_yr * costs[it]),
            "Trend vs prev 2 wks (%)": round(trend, 0) if np.isfinite(trend) else np.nan,
            "Volatility (CV)": round(on.std() / on.mean(), 2) if len(on) > 2 and on.mean() > 0 else np.nan,
            "Model": r["method"],
            "Model error %": r["err"],
        }
        if prepared is not None:
            p = prepared[it][srv].sum()
            row["Waste % of cooked"] = round(on.sum() / p * 100, 1) if p > 0 else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("Annual waste cost (₹)", ascending=False).reset_index(drop=True)
    tot_cost = summary["Annual waste cost (₹)"].sum()
    summary["Share of cost (%)"] = (summary["Annual waste cost (₹)"] / tot_cost * 100).round(1) if tot_cost else 0.0
    cum = summary["Share of cost (%)"].cumsum()
    summary["ABC class"] = np.where(cum.shift(fill_value=0) < 70, "A", np.where(cum.shift(fill_value=0) < 90, "B", "C"))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Items found", len(items))
    k2.metric("Avg total waste / day", f"{waste.sum(axis=1).mean():.1f} kg")
    k3.metric("Projected annual waste cost", inr(tot_cost))
    k4.metric("Costliest item", summary.iloc[0]["Item"], f"{summary.iloc[0]['Share of cost (%)']}% of cost", delta_color="off")

    t_rank, t_item, t_rec, t_plan = st.tabs(["🏆 Item ranking", "🔍 Item forecast", "💡 Recommendations", "🍳 Cooking plan"])

    # ================= ranking
    with t_rank:
        st.subheader("Which items cost the most? (Pareto / ABC analysis)")
        st.caption("Class A = items making up the first ~70% of waste cost. Fixing these few items gives most of the savings.")
        par = go.Figure()
        colors = summary["ABC class"].map({"A": "#E45756", "B": "#F58518", "C": "#54A24B"})
        par.add_trace(go.Bar(x=summary["Item"], y=summary["Annual waste cost (₹)"], marker_color=colors,
                             name="Annual waste cost (₹)"))
        par.add_trace(go.Scatter(x=summary["Item"], y=summary["Share of cost (%)"].cumsum(), yaxis="y2",
                                 name="Cumulative %", line=dict(color="#4C78A8", width=3), mode="lines+markers"))
        par.update_layout(height=400, yaxis=dict(title="Annual waste cost (₹)"),
                          yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
                          legend=dict(orientation="h", y=1.1), margin=dict(t=30))
        st.plotly_chart(par, width="stretch")

        show_cols = ["ABC class", "Item", "Meal", "Annual waste cost (₹)", "Share of cost (%)",
                     f"Forecast next {horizon}d (kg)", f"Forecast next {horizon}d (₹)", "Avg waste when served (kg)",
                     "Waste % of cooked", "Trend vs prev 2 wks (%)", "Volatility (CV)", "Served (% of days)",
                     "₹/kg", "Model", "Model error %"]
        show_cols = [c for c in show_cols if c in summary.columns and (c != "Meal" or meals is not None)]
        st.dataframe(summary[show_cols], width="stretch", hide_index=True)
        st.download_button("Download item summary (CSV)", summary[show_cols].to_csv(index=False),
                           "item_waste_summary.csv", "text/csv", key="it_dl_sum")

        st.markdown("#### When is each item wasted? (avg kg on days it's served)")
        heat = waste.where(served).groupby(waste.index.dayofweek).mean().T.reindex(summary["Item"])
        heat.columns = [DAY_NAMES[c][:3] for c in heat.columns]
        hm = go.Figure(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale="OrRd",
                                  text=np.where(np.isnan(heat.values), "", np.round(heat.values, 1).astype(str)),
                                  texttemplate="%{text}", hoverongaps=False, colorbar=dict(title="kg")))
        hm.update_layout(height=max(320, 32 * len(items)), yaxis=dict(autorange="reversed"), margin=dict(t=10))
        st.plotly_chart(hm, width="stretch")
        st.caption("Blank cells = the item wasn't on the menu that weekday.")

        if meals is not None:
            st.markdown("#### Waste cost by meal")
            by_meal = summary.groupby("Meal")["Annual waste cost (₹)"].sum().sort_values(ascending=False)
            mb = go.Figure(go.Bar(x=by_meal.index, y=by_meal.values, marker_color="#4C78A8",
                                  text=[inr(v) for v in by_meal.values], textposition="outside"))
            mb.update_layout(height=300, yaxis_title="Annual waste cost (₹)", margin=dict(t=20))
            st.plotly_chart(mb, width="stretch")

    # ================= single item
    with t_item:
        it = st.selectbox("Choose an item", summary["Item"], key="it_pick")
        r, s = results[it], waste[it]
        hist = s.iloc[-60:].where(served[it].iloc[-60:])
        low = (r["fc"] - 1.28 * r["sd"]).clip(lower=0).where(r["fc"] > 0, 0)
        high = (r["fc"] + 1.28 * r["sd"]).where(r["fc"] > 0, 0)
        g = go.Figure()
        g.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines+markers", name="Past waste",
                               line=dict(color="#4C78A8"), connectgaps=False))
        g.add_trace(go.Scatter(x=high.index, y=high.values, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        g.add_trace(go.Scatter(x=low.index, y=low.values, fill="tonexty", fillcolor="rgba(245,133,24,0.2)",
                               line=dict(width=0), name="Likely range (~80%)"))
        g.add_trace(go.Bar(x=r["fc"].index, y=r["fc"].values, name="Forecast", marker_color="#F58518", opacity=0.85))
        g.update_layout(height=400, yaxis_title=f"{it} waste (kg)", hovermode="x unified", margin=dict(t=20))
        st.plotly_chart(g, width="stretch")
        st.caption("Gaps in the past line = days the item wasn't served. Forecast is 0 on days it isn't on the recent menu.")
        row = summary.set_index("Item").loc[it]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Forecast waste, next {horizon} days", f"{r['fc'].sum():.1f} kg")
        m2.metric("Forecast cost", inr(r["fc"].sum() * costs[it]))
        m3.metric("Annual waste cost", inr(row["Annual waste cost (₹)"]))
        m4.metric("Trend vs previous 2 weeks",
                  f"{row['Trend vs prev 2 wks (%)']:+.0f}%" if np.isfinite(row["Trend vs prev 2 wks (%)"]) else "n/a")
        st.write(f"**Model used:** {r['method']}" +
                 (f" — error on hidden test days: {r['err']}%" if np.isfinite(r["err"]) else ""))

    # ================= recommendations
    with t_rec:
        st.subheader("Recommendations")
        st.caption("Generated from your data by fixed rules (thresholds listed at the bottom). "
                   "They point to where to look, and still need the mess manager's judgement.")
        recs = []
        for _, row in summary.iterrows():
            it, s, srv = row["Item"], waste[row["Item"]], served[row["Item"]]
            on = s[srv]
            if row["ABC class"] == "A":
                recs.append((1, it, f"**Top cost driver:** {it} is {row['Share of cost (%)']}% of all waste cost "
                                    f"({inr(row['Annual waste cost (₹)'])} a year). Halving its waste would save about "
                                    f"{inr(row['Annual waste cost (₹)'] / 2)} a year."))
            if "Waste % of cooked" in row and np.isfinite(row["Waste % of cooked"]) and row["Waste % of cooked"] >= 20:
                recs.append((1 if row["ABC class"] == "A" else 2, it,
                             f"**Over-cooked:** {row['Waste % of cooked']}% of the {it} cooked is thrown away. "
                             f"See the Cooking plan tab for a smaller batch size."))
            tr = row["Trend vs prev 2 wks (%)"]
            if np.isfinite(tr) and tr >= 20:
                recs.append((2, it, f"**Rising waste:** {it} waste is up {tr:.0f}% versus the previous two weeks. "
                                    "Check taste, quality or the recipe, and collect quick student feedback."))
            elif np.isfinite(tr) and tr <= -20:
                recs.append((3, it, f"**Improving:** {it} waste is down {abs(tr):.0f}% versus the previous two weeks. "
                                    "Find out what changed and keep doing it."))
            dow = on.groupby(on.index.dayofweek).mean()
            if len(dow) >= 3:
                top = dow.idxmax()
                others = dow.drop(top).mean()
                if others > 0 and dow[top] >= 1.3 * others:
                    recs.append((2, it, f"**{DAY_NAMES[top]} problem:** {it} waste averages {dow[top]:.1f} kg on "
                                        f"{DAY_NAMES[top]}s vs {others:.1f} kg on other days. Cook a smaller "
                                        f"{DAY_NAMES[top]} batch (attendance is probably lower)."))
            cv = row["Volatility (CV)"]
            if np.isfinite(cv) and cv >= 0.5:
                recs.append((2, it, f"**Unpredictable:** {it} waste swings a lot from day to day (CV {cv}). "
                                    "Batch-cook in smaller rounds or use meal pre-booking instead of one big batch."))
        if not recs:
            st.success("No item crosses the warning thresholds. Waste looks under control.")
        labels = {1: ("🔴 High priority", st.error), 2: ("🟠 Medium priority", st.warning), 3: ("🟢 Good news", st.success)}
        for p in (1, 2, 3):
            group = [r for r in recs if r[0] == p]
            if group:
                st.markdown(f"#### {labels[p][0]}")
                for _, _, txt in group:
                    labels[p][1](txt)

        st.markdown("#### Savings potential")
        cut = st.slider("If the mess cut waste of Class A items by...", 0, 80, 30, step=5, key="it_cut") / 100
        a_cost = summary.loc[summary["ABC class"] == "A", "Annual waste cost (₹)"].sum()
        sv1, sv2, sv3 = st.columns(3)
        sv1.metric("Class A items", int((summary["ABC class"] == "A").sum()))
        sv2.metric("Their annual waste cost", inr(a_cost))
        sv3.metric("Annual saving", inr(a_cost * cut), f"{a_cost * cut / tot_cost * 100:.0f}% of all waste cost" if tot_cost else None,
                   delta_color="off")
        st.caption("Use this saving as the yearly benefit in the Investment appraisal tab on the Overall page.")
        with st.expander("Rule thresholds"):
            st.markdown("- Top cost driver: ABC class A (first ~70% of waste cost)\n"
                        "- Over-cooked: waste ≥ 20% of quantity cooked (needs `prepared_kg`)\n"
                        "- Rising / improving: last 2 weeks vs previous 2 weeks, ±20%\n"
                        "- Weekday problem: one weekday ≥ 1.3× the other days' average\n"
                        "- Unpredictable: coefficient of variation (std ÷ mean) ≥ 0.5\n\n"
                        "These cut-offs are judgement calls, not industry standards. Adjust them in the code if needed.")

    # ================= cooking plan
    with t_plan:
        st.subheader(f"Recommended cooking plan, next {horizon} days")
        plan_rows = []
        for it in items:
            r = results[it]
            for d, w in r["fc"].items():
                if w <= 0:
                    continue
                if prepared is not None:
                    hist_p = prepared[it][served[it]]
                    same = hist_p[hist_p.index.dayofweek == d.dayofweek].iloc[-8:]
                    usual = float(same.mean()) if len(same) else float(hist_p.iloc[-8:].mean())
                    eaten = max(0.0, usual - w)
                    rec = min(usual, eaten * (1 + buffer))
                    plan_rows.append((d, it, round(usual, 1), round(w, 1), round(rec, 1), round(usual - rec, 1),
                                      round((usual - rec) * costs[it])))
                else:
                    cut_kg = w * (1 - buffer)
                    plan_rows.append((d, it, np.nan, round(w, 1), np.nan, round(cut_kg, 1), round(cut_kg * costs[it])))
        plan = pd.DataFrame(plan_rows, columns=["date", "Item", "Usual cooked (kg)", "Expected waste (kg)",
                                                "Recommended to cook (kg)", "Cook less by (kg)", "Saving (₹)"])
        if plan.empty:
            st.info("No items are forecast to be served in this period.")
        else:
            p1, p2 = st.columns(2)
            p1.metric("Food saved if the plan is followed", f"{plan['Cook less by (kg)'].sum():,.0f} kg")
            p2.metric("Money saved", inr(plan["Saving (₹)"].sum()))
            if prepared is not None:
                st.caption(f"Recommended = (usual quantity cooked on that weekday − forecast waste) × (1 + {buffer:.0%} "
                           "safety buffer). The usual quantity is the average of the last 8 same-weekday servings.")
            else:
                st.caption(f"Your file has no `prepared_kg`, so this shows how much less to cook: forecast waste "
                           f"minus a {buffer:.0%} safety buffer. Add `prepared_kg` to get exact cooking quantities.")
            plan.insert(1, "Day", plan["date"].dt.day_name().str[:3])
            plan["date"] = plan["date"].dt.strftime("%d %b")
            if prepared is None:
                plan = plan.drop(columns=["Usual cooked (kg)", "Recommended to cook (kg)"])
            st.dataframe(plan, width="stretch", hide_index=True)
            st.download_button("Download cooking plan (CSV)", plan.to_csv(index=False), "cooking_plan.csv",
                               "text/csv", key="it_dl_plan")
            st.warning("Check the plan against known events (exams, fests, holidays, guests). "
                       "The model only knows the patterns in your past data.")


# ================================================================ navigation
nav = st.navigation([
    st.Page(page_overall, title="Overall waste & finance", icon="📊", default=True),
    st.Page(page_items, title="Item-wise forecast", icon="🥘", url_path="items"),
])
nav.run()
