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


# ---------------------------------------------------------------- UI
st.title("🍛 Hostel Mess Food Waste Forecaster")
st.caption(
    "Enter past daily food-waste records. The app trains several machine-learning models, "
    "checks which one predicts your mess best, and forecasts the coming days."
)

with st.sidebar:
    st.header("Settings")
    horizon = st.slider("Days to forecast", 7, 30, 14)
    cost_per_kg = st.number_input("Approx. cost of food (₹ per kg)", 0.0, 1000.0, 60.0, step=5.0)
    st.caption("Used only to put a rupee value on the expected waste.")

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

# --- summary
m1, m2, m3, m4 = st.columns(4)
m1.metric("Days of data", len(s))
m2.metric("Avg waste / day", f"{s.mean():.1f} kg")
m3.metric("Total waste", f"{s.sum():,.0f} kg")
m4.metric("Value wasted", f"₹{s.sum() * cost_per_kg:,.0f}")

# --- model selection
test_days = max(7, min(14, len(s) // 5))
with st.spinner("Training and comparing models..."):
    scores, preds, test = evaluate(s, test_days)
best = scores.index[0]

# --- final forecast with the best model on all data
if best == BASELINE:
    fc = seasonal_naive(s, horizon)
else:
    fc = recursive_forecast(fit(get_models()[best], s), s, horizon)

resid_sd = float((test - preds[best]).std())
low = (fc - 1.28 * resid_sd).clip(lower=0)
high = fc + 1.28 * resid_sd

st.subheader(f"Forecast for the next {horizon} days")
st.caption(f"Best model on your data: **{best}** (tested on the last {test_days} days you entered).")

hist_view = s.iloc[-60:]
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist_view.index, y=hist_view.values, name="Past waste", line=dict(color="#4C78A8")))
fig.add_trace(go.Scatter(x=high.index, y=high.values, line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(
    go.Scatter(
        x=low.index, y=low.values, fill="tonexty", fillcolor="rgba(245,133,24,0.2)",
        line=dict(width=0), name="Likely range (~80%)",
    )
)
fig.add_trace(go.Scatter(x=fc.index, y=fc.values, name="Forecast", line=dict(color="#F58518", width=3)))
fig.update_layout(height=420, yaxis_title="Food waste (kg)", hovermode="x unified", margin=dict(t=20))
st.plotly_chart(fig, width="stretch")

c1, c2, c3 = st.columns(3)
c1.metric("Expected waste (forecast period)", f"{fc.sum():,.0f} kg")
c2.metric("Expected value wasted", f"₹{fc.sum() * cost_per_kg:,.0f}")
c3.metric("Expected avg / day", f"{fc.mean():.1f} kg", delta=f"{fc.mean() - s.iloc[-14:].mean():+.1f} kg vs last 2 weeks", delta_color="inverse")

dow_avg = s.groupby(s.index.dayofweek).mean()
peak, calm = DAY_NAMES[dow_avg.idxmax()], DAY_NAMES[dow_avg.idxmin()]
st.info(
    f"Waste is highest on **{peak}s** ({dow_avg.max():.1f} kg on average) and lowest on "
    f"**{calm}s** ({dow_avg.min():.1f} kg). Planning smaller batches for {peak} meals is the "
    "first place to look for savings."
)

table = pd.DataFrame(
    {
        "Date": fc.index.strftime("%d %b %Y"),
        "Day": [DAY_NAMES[d] for d in fc.index.dayofweek],
        "Expected waste (kg)": fc.values.round(1),
        "Low (kg)": low.values.round(1),
        "High (kg)": high.values.round(1),
        "Value (₹)": (fc.values * cost_per_kg).round(0),
    }
)
st.dataframe(table, width="stretch", hide_index=True)
st.download_button("Download forecast (CSV)", table.to_csv(index=False), "waste_forecast.csv", "text/csv")

with st.expander("Model comparison"):
    st.dataframe(scores, width="stretch")
    st.caption(
        "Each model forecast the last few days of your data without seeing them; lower error is better. "
        "The baseline simply repeats last week, so an ML model is only worth using if it beats it."
    )

with st.expander("How does this work?"):
    st.markdown(
        """
- **Inputs the models learn from:** day of week, month, weekend flag, waste 1/2/3/7/14 days earlier,
  and the 7- and 14-day average.
- **Models tried:** Ridge Regression, Random Forest, Gradient Boosting, plus a seasonal-naive baseline.
- **Picking the model:** the last part of your data is hidden, each model forecasts it, and the one
  with the lowest average error wins. It is then retrained on all your data.
- **Multi-day forecast:** the model predicts tomorrow, feeds that prediction back in, and repeats.
- **Range:** based on how wrong the best model was during testing, so it is an approximation.
"""
    )
