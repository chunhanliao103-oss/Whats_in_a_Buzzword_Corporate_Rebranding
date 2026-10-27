"""
Shared event-study functions for notebooks 03-07.

All notebooks import from this one module, so every CAR, test and figure is computed the same way.
Design choices:

* day 0 is placed on the **market trading calendar** (CRSP index dates), not on the firm's own rows;
  news after the close moves to the next session, and an event whose stock does not trade within
  ``MAX_DAY0_LAG`` sessions of day 0 is dropped instead of being shifted;
* every CAR window must be **complete**;
* volatility and normal volume come from the **pre-event** window only (no post-event baselines);
  events with too little pre-event history get descriptive market-adjusted returns and no BMP statistic;
* firm characteristics are strictly pre-event (no first-trading-day or post-event fallbacks);
* CAAR figures use exactly the events that have the plotted window, and never fill missing returns with zero.
"""
import numpy as np
from matplotlib.ticker import MaxNLocator
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# ---- design (fixed before looking at any returns) ----
ESTIMATION_WINDOW = (-250, -30)
MIN_ESTIMATION_DAYS = 60          # below this: market-adjusted model, reported as descriptive only
MIN_BASELINE_DAYS = 20            # minimum pre-event days for volatility / normal volume
MAX_DAY0_LAG = 2                  # the stock must trade within 2 sessions of day 0
CAR_WINDOWS = {"car_m1_p1": (-1, 1), "car_m5_p5": (-5, 5), "car_m10_p10": (-10, 10)}
PRIMARY_MEASURE = "car_m1_p1"
SECONDARY_MEASURES = ["car_m5_p5", "cav_m1_p1"]
PLOT_WINDOW = (-10, 10)
N_BOOTSTRAP = 10_000
MIN_BOOTSTRAP_N = 10
MIN_CLUSTERS = 10                 # clustered tests need at least this many calendar months
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


FF12_RANGES = {   # Fama & French (1997) 12-industry classification by 4-digit SIC (Ken French data library)
    "NoDur Consumer non-durables": [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)],
    "Durbl Consumer durables": [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716),
                                (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)],
    "Manuf Manufacturing": [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629),
                            (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799),
                            (3830, 3839), (3860, 3899)],
    "Enrgy Energy": [(1200, 1399), (2900, 2999)],
    "Chems Chemicals": [(2800, 2829), (2840, 2899)],
    "BusEq Business equipment (computers, software, electronics)": [(3570, 3579), (3660, 3692), (3694, 3699),
                                                                     (3810, 3829), (7370, 7379)],
    "Telcm Telecom": [(4800, 4899)],
    "Utils Utilities": [(4900, 4949)],
    "Shops Wholesale, retail, services": [(5000, 5999), (7200, 7299), (7600, 7699)],
    "Hlth Healthcare": [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)],
    "Money Finance": [(6000, 6999)],
}
SPAC_SIC_CODES = {6770, 6799}     # blank-check / other investors: the shell's code, not the operating business


def ff12_industry(sic):
    """
    Map a 4-digit SIC code to the Fama-French 12-industry group.

    Parameters
    ----------
    sic : float or int
        SIC code (missing or 9999 = unclassified).

    Returns
    -------
    str
        Industry label; "Other" for codes outside the 11 ranges, "Unclassified" for missing / 9999.
    """
    if pd.isna(sic) or int(sic) in (0, 9999):
        return "Unclassified"
    code = int(sic)
    for label, ranges in FF12_RANGES.items():
        if any(low <= code <= high for low, high in ranges):
            return label
    return "Other"


def load_daily(target_file):
    """
    Load and clean a CRSP daily stock file.

    Parameters
    ----------
    target_file : pathlib.Path

    Returns
    -------
    pandas.DataFrame
        Sorted by permno/date with ``log_volume`` = log(1 + volume) and ``price`` = |price|.
    """
    daily = pd.read_csv(target_file)
    daily["date"] = pd.to_datetime(daily["dlycaldt"])
    daily = daily.sort_values(["permno", "date"]).drop_duplicates(["permno", "date"]).reset_index(drop=True)
    daily["log_volume"] = np.log1p(daily["dlyvol"])
    daily.loc[daily["dlyvol"].isna(), "log_volume"] = np.nan
    daily["price"] = daily["dlyprc"].abs()
    return daily


def load_market(target_file):
    """
    Load the CRSP daily market index and number its trading sessions.

    Parameters
    ----------
    target_file : pathlib.Path

    Returns
    -------
    pandas.DataFrame
        ``date``, ``market_return`` (value-weighted), ``ewretd``, ``sprtrn`` and ``session`` (0, 1, 2, ...).
    """
    market = pd.read_csv(target_file)
    market["date"] = pd.to_datetime(market["dlycaldt"])
    market = market.rename(columns={"vwretd": "market_return"}).sort_values("date").reset_index(drop=True)
    market["session"] = np.arange(len(market))
    return market[["date", "session", "market_return", "ewretd", "sprtrn"]]


def day0_session(event_date, market, timing="unknown", shift_sessions=0):
    """
    Map a public-information date to the first market session that can react to it.

    Parameters
    ----------
    event_date : pandas.Timestamp
    market : pandas.DataFrame
        Output of :func:`load_market`.
    timing : {"unknown", "before_open", "after_close"}
        Release time; after-close news moves to the next session.
    shift_sessions : int
        Extra sessions (timing sensitivity check: 1 = day 0 one session later).

    Returns
    -------
    int or None
        Session number of day 0, ``None`` if the date is after the data end.
    """
    sessions = market["session"].to_numpy()
    position = np.searchsorted(market["date"].to_numpy(), np.datetime64(event_date), side="left")
    position += int(timing == "after_close") + shift_sessions
    return int(sessions[position]) if position < len(sessions) else None


def locate_event_window(firm, day0):
    """
    Add ``relative_day`` (market sessions relative to day 0) to one firm's daily data.

    Parameters
    ----------
    firm : pandas.DataFrame
        Daily data of one PERMNO with the market ``session`` attached.
    day0 : int
        Session number of day 0.

    Returns
    -------
    pandas.DataFrame
        The firm's rows with ``relative_day``; missing trading days stay missing.
    """
    return firm.assign(relative_day=firm["session"] - day0)


def compute_abnormal_returns(events, daily, market, model_name="market_model", date_column="event_date",
                             timing_column=None, shift_sessions=0):
    """
    Compute daily abnormal returns / volume around each event and the per-event CAR variance inputs.

    Parameters
    ----------
    events : pandas.DataFrame
        Needs ``event_id``, ``permno`` and ``date_column``.
    daily : pandas.DataFrame
        Output of :func:`load_daily`.
    market : pandas.DataFrame
        Output of :func:`load_market`.
    model_name : {"market_model", "strict_market_model", "market_adjusted", "constant_mean"}
        ``market_model`` uses the market-adjusted model (descriptive only) for events with fewer than
        :data:`MIN_ESTIMATION_DAYS`; ``strict_market_model`` drops those events.
    date_column : str
        Column with the public-information date (``event_date`` or ``effective_date``).
    timing_column : str, optional
        Column with the release time (``after_close`` moves day 0 one session later).
    shift_sessions : int
        Extra sessions added to day 0 for every event (sensitivity check).

    Returns
    -------
    abnormal : pandas.DataFrame
        Long table ``event_id``, ``relative_day``, ``abnormal_return``, ``abnormal_volume``, ``market_return``
        for the plot window.
    estimates : pandas.DataFrame
        One row per event: day 0, lag to the first trade, estimation length, benchmark, residual sd,
        whether a pre-event baseline exists, and the drop reason.
    residuals : dict
        ``event_id`` -> estimation-window abnormal returns (for the single-firm SQ test).
    """
    data = daily.merge(market[["date", "session", "market_return"]], on="date", how="inner")
    groups = dict(tuple(data.groupby("permno")))
    frames, estimates, residuals = [], [], {}
    for event in events.itertuples():
        info = {"event_id": event.event_id, "benchmark": model_name, "dropped_reason": ""}
        timing = getattr(event, timing_column) if timing_column else "unknown"
        day0 = day0_session(getattr(event, date_column), market, timing, shift_sessions)
        firm = groups.get(event.permno)
        if day0 is None or firm is None:
            estimates.append({**info, "dropped_reason": "no market or stock data"})
            continue
        firm = locate_event_window(firm, day0)
        info["day0_date"] = market.loc[market["session"] == day0, "date"].iloc[0]
        traded = firm.loc[(firm["relative_day"] >= 0) & firm["dlyret"].notna(), "relative_day"]
        info["day0_lag_sessions"] = int(traded.min()) if len(traded) else np.nan
        if not len(traded) or traded.min() > MAX_DAY0_LAG:
            estimates.append({**info, "dropped_reason": f"no trade within {MAX_DAY0_LAG} sessions of day 0"})
            continue
        est = firm[firm["relative_day"].between(*ESTIMATION_WINDOW)].dropna(subset=["dlyret", "market_return"])
        info["estimation_days"] = len(est)
        window = firm[firm["relative_day"].between(*PLOT_WINDOW)].copy()
        use_model = model_name
        if model_name == "market_model" and len(est) < MIN_ESTIMATION_DAYS:
            use_model = "market_adjusted"
        if model_name == "strict_market_model":
            if len(est) < MIN_ESTIMATION_DAYS:
                estimates.append({**info, "dropped_reason": f"< {MIN_ESTIMATION_DAYS} estimation days"})
                continue
            use_model = "market_model"
        info["benchmark_used"] = use_model
        info["has_baseline"] = len(est) >= MIN_BASELINE_DAYS
        info["descriptive_only"] = len(est) < MIN_ESTIMATION_DAYS
        if use_model == "market_adjusted":
            window["expected_return"] = window["market_return"]
            excess = est["dlyret"] - est["market_return"]
            info["residual_sd"] = excess.std() if info["has_baseline"] else np.nan
            residuals[event.event_id] = excess.to_numpy()
        elif use_model == "market_model":
            beta, alpha = np.polyfit(est["market_return"], est["dlyret"], 1)
            fitted = est["dlyret"] - (alpha + beta * est["market_return"])
            window["expected_return"] = alpha + beta * window["market_return"]
            info.update(alpha=alpha, beta=beta, residual_sd=fitted.std(ddof=2))
            residuals[event.event_id] = fitted.to_numpy()
        else:
            window["expected_return"] = est["dlyret"].mean()
            info["residual_sd"] = est["dlyret"].std(ddof=1) if info["has_baseline"] else np.nan
            residuals[event.event_id] = (est["dlyret"] - est["dlyret"].mean()).to_numpy()
        info.update(market_mean=est["market_return"].mean() if len(est) else np.nan,
                    market_ss=((est["market_return"] - est["market_return"].mean()) ** 2).sum() if len(est) else np.nan)
        volume_est = firm[firm["relative_day"].between(*ESTIMATION_WINDOW)]["log_volume"].dropna()
        normal_volume = volume_est.mean() if len(volume_est) >= MIN_BASELINE_DAYS else np.nan
        window["abnormal_return"] = window["dlyret"] - window["expected_return"]
        window["abnormal_volume"] = window["log_volume"] - normal_volume
        window["event_id"] = event.event_id
        frames.append(window[["event_id", "relative_day", "abnormal_return", "abnormal_volume", "market_return"]])
        estimates.append(info)
    abnormal = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["event_id", "relative_day", "abnormal_return", "abnormal_volume", "market_return"])
    return abnormal, pd.DataFrame(estimates), residuals


def compute_cars(abnormal, estimates, events, keep_columns=None):
    """
    Sum abnormal returns / volume over each **complete** window and standardise CARs (BMP).

    Parameters
    ----------
    abnormal, estimates : pandas.DataFrame
        Outputs of :func:`compute_abnormal_returns`.
    events : pandas.DataFrame
    keep_columns : list of str, optional
        Event columns copied to the output (default: identifiers, theme, dates, flags).

    Returns
    -------
    pandas.DataFrame
        One row per usable event with ``car_*``, ``scar_*`` (NaN without a pre-event baseline) and ``cav_*``.
    """
    est = estimates[estimates["dropped_reason"] == ""].set_index("event_id")
    rows = []
    for event_id, group in abnormal.groupby("event_id"):
        if event_id not in est.index:
            continue
        e = est.loc[event_id]
        row = {"event_id": event_id}
        for name, (start, end) in CAR_WINDOWS.items():
            window = group[group["relative_day"].between(start, end)]
            length = end - start + 1
            complete = window["abnormal_return"].notna().sum() == length
            car = window["abnormal_return"].sum() if complete else np.nan
            if e["benchmark_used"] == "market_model":
                market_sum = window["market_return"].sum()
                variance = e["residual_sd"] ** 2 * (length + length ** 2 / e["estimation_days"]
                                                    + (market_sum - length * e["market_mean"]) ** 2 / e["market_ss"])
            else:
                variance = e["residual_sd"] ** 2 * length
            volume_complete = window["abnormal_volume"].notna().sum() == length
            row[name] = car
            row[name.replace("car", "scar")] = car / np.sqrt(variance) if complete and np.isfinite(variance) else np.nan
            row[name.replace("car", "cav")] = window["abnormal_volume"].sum() if volume_complete else np.nan
        rows.append(row)
    keep = keep_columns or [c for c in ["event_id", "permno", "theme", "event_date", "effective_date",
                                        "date_confidence", "deal_type", "former_spac", "in_wave_period",
                                        "primary_sample", "old_name", "new_name", "group", "year"]
                            if c in events.columns]
    cars = events[keep].merge(pd.DataFrame(rows), on="event_id", how="inner")
    return cars.merge(est[["benchmark_used", "descriptive_only", "day0_date"]].reset_index(), on="event_id", how="left")


def bootstrap_mean(values, n_draws=N_BOOTSTRAP):
    """
    Bootstrap the mean: 95% percentile interval and two-sided p-value of "mean = 0".

    Parameters
    ----------
    values : array-like
    n_draws : int

    Returns
    -------
    tuple of float
        (lower bound, upper bound, p-value)
    """
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(n_draws, len(values)), replace=True).mean(axis=1)
    p_value = (np.abs(draws - values.mean()) >= abs(values.mean())).mean()
    return np.percentile(draws, 2.5), np.percentile(draws, 97.5), p_value


def clustered_mean_pvalue(values, clusters):
    """
    p-value of "mean = 0" with standard errors clustered by calendar month (event clustering).

    Parameters
    ----------
    values : pandas.Series
    clusters : pandas.Series
        Cluster label per value (e.g. the event month).

    Returns
    -------
    tuple
        (number of clusters, p-value); p-value is NaN with fewer than :data:`MIN_CLUSTERS` clusters.
    """
    groups = pd.factorize(clusters)[0]
    n_clusters = len(np.unique(groups))
    if n_clusters < MIN_CLUSTERS:
        return n_clusters, np.nan
    fit = sm.OLS(values.to_numpy(dtype=float), np.ones(len(values))).fit(cov_type="cluster",
                                                                         cov_kwds={"groups": groups})
    return n_clusters, float(fit.pvalues[0])


def test_event_sample(cars, measure, cluster_column="event_month"):
    """
    Describe one measure and test "mean = 0" / "median = 0".

    Reports mean, median, share positive and a t-based 95% CI, plus: t-test, sign test (median, valid
    under skewness), Wilcoxon signed-rank, BMP (CARs with a pre-event
    baseline only), bootstrap (n >= 10) and a month-clustered t-test (>= 10 months).

    Parameters
    ----------
    cars : pandas.DataFrame
    measure : str
    cluster_column : str

    Returns
    -------
    dict
    """
    sample = cars.dropna(subset=[measure])
    values = sample[measure]
    result = {"n": len(values), "mean": values.mean(), "median": values.median(),
              "pct_positive": (values > 0).mean() * 100}
    if len(values) < 3:
        return result
    se = values.std() / np.sqrt(len(values))
    t_crit = stats.t.ppf(0.975, len(values) - 1)
    result.update(ci_low=values.mean() - t_crit * se, ci_high=values.mean() + t_crit * se,
                  mde_80=(t_crit + stats.t.ppf(0.80, len(values) - 1)) * se,
                  t_pvalue=stats.ttest_1samp(values, 0).pvalue,
                  sign_pvalue=stats.binomtest(int((values > 0).sum()), int((values != 0).sum())).pvalue,
                  wilcoxon_pvalue=stats.wilcoxon(values).pvalue)
    if measure.startswith("car"):
        scar = sample[measure.replace("car", "scar")].dropna()
        result["n_bmp"] = len(scar)
        if len(scar) >= 3:
            result["bmp_t"] = scar.mean() * np.sqrt(len(scar)) / scar.std()
            result["bmp_pvalue"] = 2 * stats.t.sf(abs(result["bmp_t"]), df=len(scar) - 1)
    if len(values) >= MIN_BOOTSTRAP_N:
        result["boot_ci_low"], result["boot_ci_high"], result["boot_pvalue"] = bootstrap_mean(values)
    if cluster_column in sample.columns:
        result["n_clusters"], result["cluster_pvalue"] = clustered_mean_pvalue(values, sample[cluster_column])
    return result


def summarise_event_study(cars, samples, measures):
    """
    Tabulate :func:`test_event_sample` for several samples and measures.

    Parameters
    ----------
    cars : pandas.DataFrame
    samples : dict
        Sample label -> boolean mask (or callable on ``cars``).
    measures : list of str

    Returns
    -------
    pandas.DataFrame
    """
    rows = []
    for label, mask in samples.items():
        subset = cars[mask(cars)] if callable(mask) else cars[mask]
        for measure in measures:
            rows.append({"sample": label, "measure": measure, **test_event_sample(subset, measure)})
    return pd.DataFrame(rows)


def holm_adjust(p_values):
    """
    Holm (1979) step-down adjustment of a family of p-values.

    Parameters
    ----------
    p_values : array-like

    Returns
    -------
    numpy.ndarray
        Adjusted p-values in the original order.
    """
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def leave_one_out(cars, measure, label_column="new_name"):
    """
    Influence check: the mean after dropping each event in turn.

    Parameters
    ----------
    cars : pandas.DataFrame
    measure : str
    label_column : str

    Returns
    -------
    pandas.DataFrame
        One row per event: its value, the mean without it, and the change in the mean.
    """
    data = cars.dropna(subset=[measure]).reset_index(drop=True)
    total, n = data[measure].sum(), len(data)
    out = data[["event_id", label_column, measure]].copy()
    out["mean_without"] = (total - data[measure]) / (n - 1)
    out["change_in_mean"] = out["mean_without"] - data[measure].mean()
    return out.sort_values("change_in_mean")


def sq_test(car, estimation_residuals, length):
    """
    Single-firm SQ test (Gelbach, Helland & Klick, 2013).

    Compares the event CAR with the empirical distribution of all ``length``-day sums of the firm's
    pre-event abnormal returns, so it does not assume normal returns.

    Parameters
    ----------
    car : float
    estimation_residuals : numpy.ndarray
        Pre-event abnormal returns of the firm.
    length : int
        Window length in days.

    Returns
    -------
    tuple
        (two-sided p-value, number of pre-event windows); NaN with fewer than 20 windows.
    """
    residuals = np.asarray(estimation_residuals, dtype=float)
    if np.isnan(car) or len(residuals) < length + 20:
        return np.nan, 0
    sums = np.convolve(residuals, np.ones(length), mode="valid")
    return float((np.abs(sums) >= abs(car)).mean()), len(sums)


def build_crsp_characteristics(events, daily, market, date_column="event_date"):
    """
    Measure size, price, volatility and run-up strictly before day 0 (no fallbacks).

    Size and price are taken on the last trading day on or before day −30; volatility and the buy-and-hold
    return over [−250, −30] need at least :data:`MIN_BASELINE_DAYS` returns. Otherwise the value is missing.

    Parameters
    ----------
    events : pandas.DataFrame
    daily, market : pandas.DataFrame
    date_column : str

    Returns
    -------
    pandas.DataFrame
        ``event_id``, ``log_market_cap``, ``price``, ``penny_stock``, ``pre_event_volatility``,
        ``pre_event_return``.
    """
    data = daily.merge(market[["date", "session"]], on="date", how="inner")
    groups = dict(tuple(data.groupby("permno")))
    rows = []
    for event in events.itertuples():
        row = {"event_id": event.event_id}
        day0 = day0_session(getattr(event, date_column), market)
        firm = groups.get(event.permno)
        if day0 is not None and firm is not None:
            firm = locate_event_window(firm, day0)
            est = firm[firm["relative_day"].between(*ESTIMATION_WINDOW)]
            ref = est.dropna(subset=["dlycap"]).tail(1)
            if len(ref) and ref["dlycap"].iloc[0] > 0:
                row.update(log_market_cap=np.log(ref["dlycap"].iloc[0]), price=ref["price"].iloc[0],
                           penny_stock=int(ref["price"].iloc[0] < 5))
            returns = est["dlyret"].dropna()
            if len(returns) >= MIN_BASELINE_DAYS:
                row.update(pre_event_volatility=returns.std(), pre_event_return=(1 + returns).prod() - 1)
        rows.append(row)
    columns = ["event_id", "log_market_cap", "price", "penny_stock", "pre_event_volatility", "pre_event_return"]
    return pd.DataFrame(rows).reindex(columns=columns)


def plot_caar(abnormal, cars, group_column, colors, ax, measure="car_m10_p10", title=None):
    """
    Plot the cumulative average abnormal return (95% CI) per group on the events with a complete window.

    Parameters
    ----------
    abnormal : pandas.DataFrame
    cars : pandas.DataFrame
        Event-level table; only events with a non-missing ``measure`` are plotted.
    group_column : str
        Column that splits the lines (e.g. ``theme``).
    colors : dict
        Group -> colour.
    ax : matplotlib.axes.Axes
    measure : str
        Complete-window CAR that defines the plotted sample (default: the plot window [−10, +10]).
    title : str, optional
    """
    eligible = cars.dropna(subset=[measure])
    data = abnormal.merge(eligible[["event_id", group_column]], on="event_id")
    data = data[data["relative_day"].between(*CAR_WINDOWS[measure])]
    for group, color in colors.items():
        sub = data[data[group_column] == group]
        if sub.empty:
            continue
        cum = sub.pivot(index="event_id", columns="relative_day", values="abnormal_return").cumsum(axis=1)
        mean, se = cum.mean() * 100, cum.std() / np.sqrt(len(cum)) * 100
        ax.plot(mean.index, mean, marker="o", ms=3, color=color, label=f"{group} (n={len(cum)})")
        ax.fill_between(mean.index, mean - 1.96 * se, mean + 1.96 * se, color=color, alpha=0.12)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.axhline(0, color="grey", lw=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))    # trading days are whole numbers
    ax.set(title=title or "CAAR, 95% CI (events with a complete window)", ylabel="CAAR (%)",
           xlabel="Trading days relative to day 0")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
