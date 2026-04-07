from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from tqdm.auto import tqdm


def set_global_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def _resolve_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _safe_mode(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.mode().iloc[0]


def _minmax(series: pd.Series) -> pd.Series:
    arr = pd.to_numeric(series, errors="coerce").astype(float)
    finite = np.isfinite(arr)
    if not finite.any():
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    min_v = float(np.nanmin(arr))
    max_v = float(np.nanmax(arr))
    if max_v - min_v < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (arr - min_v) / (max_v - min_v)


def parse_timestamp_column(df: pd.DataFrame, new_col: str = "datetime") -> pd.DataFrame:
    df = df.copy()
    if new_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[new_col]):
        return df

    ts_col = _resolve_existing_column(df, ["time_stamp", "timestamp", "ts", "datetime"])
    if ts_col is None:
        raise ValueError("No timestamp column found. Expected one of: time_stamp, timestamp, ts, datetime.")

    if ts_col == "datetime" and pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        df[new_col] = df[ts_col]
        return df

    ts = pd.to_numeric(df[ts_col], errors="coerce")
    median_abs = np.nanmedian(np.abs(ts.to_numpy(dtype=float)))
    unit = "ms" if np.isfinite(median_abs) and median_abs > 1e11 else "s"
    df[new_col] = pd.to_datetime(ts, unit=unit, errors="coerce", utc=True).dt.tz_convert(None)
    return df


def load_raw_data(data_dir: str | Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    candidate_dirs = [Path(data_dir), Path("./data"), Path("../data"), Path("../../data"), Path(".")]

    cab_candidates = []
    weather_candidates = []
    for directory in candidate_dirs:
        cab_candidates.extend([
            directory / "cab_rides.csv",
            directory / "cab_rides.CSV",
            directory / "rides.csv",
            directory / "rideshare_kaggle.csv",
            directory / "rideshare.csv",
        ])
        weather_candidates.extend([
            directory / "weather.csv",
            directory / "weather.CSV",
        ])

    cab_path = next((p for p in cab_candidates if p.exists()), None)
    if cab_path is None:
        searched = ", ".join(str(p.parent.resolve()) for p in cab_candidates[:5])
        raise FileNotFoundError(
            "Could not find cab ride data. Place `cab_rides.csv` and `weather.csv` in a `data/` folder "
            f"near the notebook. Searched common locations including: {searched}"
        )

    weather_path = next((p for p in weather_candidates if p.exists()), None)

    cab_df = normalize_columns(pd.read_csv(cab_path))
    weather_df = normalize_columns(pd.read_csv(weather_path)) if weather_path is not None else None
    return cab_df, weather_df


def merge_weather(cab_df: pd.DataFrame, weather_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    cab_df = cab_df.copy()
    cab_df = parse_timestamp_column(cab_df, "ride_datetime")
    cab_df["time_hour"] = cab_df["ride_datetime"].dt.floor("h")

    if weather_df is None:
        return cab_df

    weather_df = weather_df.copy()
    weather_df = parse_timestamp_column(weather_df, "weather_datetime")
    weather_df["time_hour"] = weather_df["weather_datetime"].dt.floor("h")

    if "location" in weather_df.columns:
        weather_df = weather_df.rename(columns={"location": "weather_location"})

    numeric_weather_cols = [
        c for c in ["temp", "clouds", "pressure", "rain", "humidity", "wind"] if c in weather_df.columns
    ]
    categorical_weather_cols = [c for c in ["short_summary", "summary", "icon"] if c in weather_df.columns]

    agg_spec = {col: "mean" for col in numeric_weather_cols}
    agg_spec.update({col: _safe_mode for col in categorical_weather_cols})

    if "weather_location" in weather_df.columns:
        weather_hourly = (
            weather_df.groupby(["time_hour", "weather_location"], dropna=False)
            .agg(agg_spec)
            .reset_index()
        )
        if "source" in cab_df.columns:
            merged = cab_df.merge(
                weather_hourly,
                left_on=["time_hour", "source"],
                right_on=["time_hour", "weather_location"],
                how="left",
            )
        else:
            merged = cab_df.merge(weather_hourly.drop(columns=["weather_location"]), on="time_hour", how="left")
    else:
        weather_hourly = weather_df.groupby("time_hour", dropna=False).agg(agg_spec).reset_index()
        merged = cab_df.merge(weather_hourly, on="time_hour", how="left")

    for col in numeric_weather_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        if merged[col].isna().all():
            continue
        merged[col] = merged[col].fillna(merged[col].median())

    for col in categorical_weather_cols:
        merged[col] = merged[col].fillna("Unknown")

    return merged


def clean_rides(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_cols = ["source", "destination", "cab_type", "name", "distance", "price", "surge_multiplier"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required ride columns: {missing}")

    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["surge_multiplier"] = pd.to_numeric(df["surge_multiplier"], errors="coerce")

    df = df.dropna(subset=["ride_datetime", "source", "destination", "cab_type", "name", "distance", "surge_multiplier"])
    df = df[df["distance"] > 0].copy()
    df = df.dropna(subset=["price"]).copy()

    df["surge_multiplier"] = df["surge_multiplier"].clip(lower=1.0)
    df["route"] = df["source"].astype(str) + " -> " + df["destination"].astype(str)
    df["hour"] = df["ride_datetime"].dt.hour
    df["weekday"] = df["ride_datetime"].dt.dayofweek
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["day"] = df["ride_datetime"].dt.day
    df["month"] = df["ride_datetime"].dt.month
    return df


def build_context_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    weather_num = [c for c in ["temp", "clouds", "pressure", "rain", "humidity", "wind"] if c in df.columns]
    weather_cat = [c for c in ["short_summary", "summary", "icon"] if c in df.columns]

    named_aggs = {
        "distance_mean": ("distance", "mean"),
        "price_mean": ("price", "mean"),
        "price_median": ("price", "median"),
        "surge_mean": ("surge_multiplier", "mean"),
        "surge_max": ("surge_multiplier", "max"),
        "surge_share": ("surge_multiplier", lambda x: float(np.mean(pd.to_numeric(x, errors="coerce") > 1.0))),
        "obs_count": ("source", "size"),
    }
    for col in weather_num:
        named_aggs[col] = (col, "mean")
    for col in weather_cat:
        named_aggs[col] = (col, _safe_mode)

    context = (
        df.groupby(["time_hour", "source", "destination", "cab_type", "name"], dropna=False)
        .agg(**named_aggs)
        .reset_index()
    )

    context["route"] = context["source"].astype(str) + " -> " + context["destination"].astype(str)
    context["hour"] = pd.to_datetime(context["time_hour"]).dt.hour
    context["weekday"] = pd.to_datetime(context["time_hour"]).dt.dayofweek
    context["is_weekend"] = (context["weekday"] >= 5).astype(int)
    context["day"] = pd.to_datetime(context["time_hour"]).dt.day
    context["month"] = pd.to_datetime(context["time_hour"]).dt.month

    context["surge_mean"] = pd.to_numeric(context["surge_mean"], errors="coerce").clip(lower=1.0)
    context["surge_max"] = pd.to_numeric(context["surge_max"], errors="coerce").clip(lower=1.0)
    context["price_mean"] = pd.to_numeric(context["price_mean"], errors="coerce")
    context["distance_mean"] = pd.to_numeric(context["distance_mean"], errors="coerce")
    context["obs_count"] = pd.to_numeric(context["obs_count"], errors="coerce").fillna(1.0)

    context = context.dropna(subset=["price_mean", "distance_mean", "surge_mean"]).copy()
    context["base_price_target"] = context["price_mean"] / context["surge_mean"].clip(lower=1.0)

    scaled_count = _minmax(np.log1p(context["obs_count"]))
    scaled_surge = _minmax(context["surge_mean"])
    context["demand_proxy_target"] = 0.6 * scaled_surge + 0.4 * scaled_count
    context["demand_proxy_target"] = context["demand_proxy_target"].clip(0.0, 1.0)

    for col in weather_num:
        context[col] = pd.to_numeric(context[col], errors="coerce")
    for col in weather_cat:
        context[col] = context[col].fillna("Unknown")

    context = context.sort_values("time_hour").reset_index(drop=True)
    return context


def time_based_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("time_hour").reset_index(drop=True)

    unique_times = np.sort(df["time_hour"].dropna().unique())
    n_times = len(unique_times)

    train_end = int(n_times * train_frac)
    val_end = int(n_times * (train_frac + val_frac))

    train_times = unique_times[:train_end]
    val_times = unique_times[train_end:val_end]
    test_times = unique_times[val_end:]

    train_df = df[df["time_hour"].isin(train_times)].copy()
    val_df = df[df["time_hour"].isin(val_times)].copy()
    test_df = df[df["time_hour"].isin(test_times)].copy()

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical_cols = [c for c in ["source", "destination", "cab_type", "name", "short_summary", "summary", "icon"] if c in df.columns]
    numeric_cols = [c for c in ["distance_mean", "hour", "weekday", "is_weekend", "day", "month", "temp", "clouds", "pressure", "rain", "humidity", "wind"] if c in df.columns]
    return categorical_cols, numeric_cols


def build_preprocessor(categorical_cols: Sequence[str], numeric_cols: Sequence[str]) -> ColumnTransformer:
    transformers = []
    if categorical_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", cat_pipe, list(categorical_cols)))
    if numeric_cols:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipe, list(numeric_cols)))
    if not transformers:
        raise ValueError("No feature columns were supplied to the preprocessor.")
    return ColumnTransformer(transformers=transformers)


def build_regression_pipeline(categorical_cols: Sequence[str], numeric_cols: Sequence[str], model=None) -> Pipeline:
    if model is None:
        model = RandomForestRegressor(
            n_estimators=250,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )
    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


@dataclass
class RewardModel:
    elasticity_base: float = 1.20
    elasticity_slope: float = 0.90
    min_elasticity: float = 0.15
    penalty_weight: float = 0.10
    demand_scale: float = 4.0
    min_demand_units: float = 0.5

    def expected_units(self, demand_score) -> np.ndarray:
        demand_score = np.asarray(demand_score, dtype=float)
        demand_score = np.clip(demand_score, 0.0, 1.0)
        return self.min_demand_units + self.demand_scale * demand_score

    def effective_elasticity(self, demand_score) -> np.ndarray:
        demand_score = np.asarray(demand_score, dtype=float)
        sensitivity = self.elasticity_base - self.elasticity_slope * np.clip(demand_score, 0.0, 1.0)
        return np.clip(sensitivity, self.min_elasticity, None)

    def conversion_units(self, demand_score, action) -> np.ndarray:
        action = np.asarray(action, dtype=float)
        sensitivity = self.effective_elasticity(demand_score)
        return self.expected_units(demand_score) * np.exp(-sensitivity * (action - 1.0))

    def gross_revenue(self, base_price, demand_score, action) -> np.ndarray:
        base_price = np.asarray(base_price, dtype=float)
        action = np.asarray(action, dtype=float)
        return np.maximum(base_price, 0.0) * action * self.conversion_units(demand_score, action)

    def penalty(self, base_price, action) -> np.ndarray:
        base_price = np.asarray(base_price, dtype=float)
        action = np.asarray(action, dtype=float)
        return self.penalty_weight * np.maximum(base_price, 0.0) * np.square(action - 1.0)

    def reward(self, base_price, demand_score, action) -> np.ndarray:
        return self.gross_revenue(base_price, demand_score, action) - self.penalty(base_price, action)


def make_state_builder(train_df: pd.DataFrame, top_n_routes: int = 20) -> Callable[[pd.Series], Tuple]:
    train_df = train_df.copy()
    top_routes = set(train_df["route"].value_counts().head(top_n_routes).index)

    def _bins(series: pd.Series, q: int = 4) -> np.ndarray:
        vals = pd.to_numeric(series, errors="coerce").dropna()
        if vals.empty:
            return np.array([0.0, 1.0])
        if vals.nunique() <= 1:
            value = float(vals.iloc[0])
            return np.array([value, value + 1e-6])
        quantiles = np.unique(np.quantile(vals, np.linspace(0.0, 1.0, q + 1)))
        if len(quantiles) < 2:
            quantiles = np.array([float(vals.min()), float(vals.max()) + 1e-6])
        return quantiles

    price_bins = _bins(train_df["pred_base_price"], q=4)
    demand_bins = _bins(train_df["pred_demand_score"], q=4)
    temp_bins = _bins(train_df["temp"], q=3) if "temp" in train_df.columns else None
    rain_bins = _bins(train_df["rain"], q=3) if "rain" in train_df.columns else None

    def _digitize(val: float, bins: Optional[np.ndarray]) -> str:
        if bins is None or len(bins) < 2 or pd.isna(val):
            return "NA"
        idx = int(np.digitize([float(val)], bins[1:-1], right=True)[0])
        return f"B{idx}"

    def builder(row: pd.Series) -> Tuple:
        route_group = row["route"] if row["route"] in top_routes else "Other"
        return (
            int(row["hour"]),
            int(row["weekday"]),
            str(row["cab_type"]),
            str(row["name"]),
            route_group,
            _digitize(row.get("pred_base_price", np.nan), price_bins),
            _digitize(row.get("pred_demand_score", np.nan), demand_bins),
            _digitize(row.get("temp", np.nan), temp_bins),
            _digitize(row.get("rain", np.nan), rain_bins),
        )

    return builder


class TabularQAgent:
    def __init__(self, actions: Sequence[float], alpha: float = 0.10, gamma: float = 0.95, epsilon_start: float = 1.0, epsilon_end: float = 0.05, epsilon_decay: float = 0.97, seed: int = 42) -> None:
        self.actions = np.array(actions, dtype=float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(lambda: np.zeros(len(self.actions), dtype=float))
        self.history_: List[float] = []

    def fit(self, train_df: pd.DataFrame, state_builder: Callable[[pd.Series], Tuple], reward_model: RewardModel, episodes: int = 40, show_progress: bool = True) -> "TabularQAgent":
        train_df = train_df.reset_index(drop=True)
        if len(train_df) < 2:
            raise ValueError("train_df must contain at least 2 rows.")
        epsilon = self.epsilon_start
        self.history_ = []
        episode_iter = range(episodes)
        if show_progress:
            episode_iter = tqdm(episode_iter, total=episodes, desc="Training Tabular Q-Learning")
        for episode in episode_iter:
            total_reward = 0.0
            for t in range(len(train_df) - 1):
                row = train_df.iloc[t]
                next_row = train_df.iloc[t + 1]
                state = state_builder(row)
                next_state = state_builder(next_row)
                if self.rng.random() < epsilon:
                    action_idx = int(self.rng.integers(len(self.actions)))
                else:
                    action_idx = int(np.argmax(self.q_table[state]))
                action = self.actions[action_idx]
                reward = float(reward_model.reward(row["pred_base_price"], row["pred_demand_score"], action))
                total_reward += reward
                best_next = float(np.max(self.q_table[next_state]))
                td_target = reward + self.gamma * best_next
                td_error = td_target - self.q_table[state][action_idx]
                self.q_table[state][action_idx] += self.alpha * td_error
            self.history_.append(total_reward)
            epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)
            if show_progress and hasattr(episode_iter, "set_postfix"):
                episode_iter.set_postfix({"episode_reward": f"{total_reward:,.0f}", "epsilon": f"{epsilon:.3f}"})
        return self

    def action(self, row: pd.Series, state_builder: Callable[[pd.Series], Tuple]) -> float:
        state = state_builder(row)
        return float(self.actions[int(np.argmax(self.q_table[state]))])

class TabularSARSAAgent:
    def __init__(self, actions: Sequence[float], alpha: float = 0.10, gamma: float = 0.95, epsilon_start: float = 1.0, epsilon_end: float = 0.05, epsilon_decay: float = 0.97, seed: int = 42) -> None:
        self.actions = np.array(actions, dtype=float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(lambda: np.zeros(len(self.actions), dtype=float))
        self.history_: List[float] = []

    def _choose_action_idx(self, state: Tuple, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(len(self.actions)))
        return int(np.argmax(self.q_table[state]))

    def fit(self, train_df: pd.DataFrame, state_builder: Callable[[pd.Series], Tuple], reward_model: RewardModel,episodes: int = 40, show_progress: bool = True) -> "TabularSARSAAgent":
        train_df = train_df.reset_index(drop=True)
        if len(train_df) < 2:
            raise ValueError("train_df must contain at least 2 rows.")

        epsilon = self.epsilon_start
        self.history_ = []

        episode_iter = range(episodes)
        if show_progress:
            episode_iter = tqdm(episode_iter, total=episodes, desc="Training SARSA")

        for episode in episode_iter:
            total_reward = 0.0

            first_row = train_df.iloc[0]
            state = state_builder(first_row)
            action_idx = self._choose_action_idx(state, epsilon)

            for t in range(len(train_df) - 1):
                row = train_df.iloc[t]
                next_row = train_df.iloc[t + 1]

                next_state = state_builder(next_row)
                next_action_idx = self._choose_action_idx(next_state, epsilon)

                action = self.actions[action_idx]
                reward = float(
                    reward_model.reward(
                        row["pred_base_price"],
                        row["pred_demand_score"],
                        action,
                    )
                )
                total_reward += reward

                td_target = reward + self.gamma * self.q_table[next_state][next_action_idx]
                td_error = td_target - self.q_table[state][action_idx]
                self.q_table[state][action_idx] += self.alpha * td_error

                state = next_state
                action_idx = next_action_idx

            self.history_.append(total_reward)
            epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)

            if show_progress and hasattr(episode_iter, "set_postfix"):
                episode_iter.set_postfix({
                    "episode_reward": f"{total_reward:,.0f}",
                    "epsilon": f"{epsilon:.3f}",
                })

        return self

    def action(self, row: pd.Series, state_builder: Callable[[pd.Series], Tuple]) -> float:
        state = state_builder(row)
        return float(self.actions[int(np.argmax(self.q_table[state]))])

# State-conditional Thompson-style contextual bandit. This is a practical extension baseline, not a full RL algorithm.
class GaussianThompsonSamplingAgent:
    def __init__(self, actions: Sequence[float], prior_mean: float = 0.0, exploration_scale: float = 1.0, seed: int = 42) -> None:
        self.actions = np.array(actions, dtype=float)
        self.prior_mean = prior_mean
        self.exploration_scale = exploration_scale
        self.rng = np.random.default_rng(seed)
        self.reward_sums_: Dict[Tuple, np.ndarray] = defaultdict(lambda: np.zeros(len(self.actions), dtype=float))
        self.reward_counts_: Dict[Tuple, np.ndarray] = defaultdict(lambda: np.zeros(len(self.actions), dtype=float))
        self.history_: List[float] = []

    def _sample_action_idx(self, state: Tuple) -> int:
        sums = self.reward_sums_[state]
        counts = self.reward_counts_[state]

        means = np.where(counts > 0, sums / np.maximum(counts, 1.0), self.prior_mean)
        stds = self.exploration_scale / np.sqrt(counts + 1.0)

        sampled_values = self.rng.normal(means, stds)
        return int(np.argmax(sampled_values))

    def fit(self, train_df: pd.DataFrame, state_builder: Callable[[pd.Series], Tuple], reward_model: RewardModel, episodes: int = 10, show_progress: bool = True) -> "GaussianThompsonSamplingAgent":
        train_df = train_df.reset_index(drop=True)
        if len(train_df) < 1:
            raise ValueError("train_df must contain at least 1 row.")

        self.history_ = []

        episode_iter = range(episodes)
        if show_progress:
            episode_iter = tqdm(episode_iter, total=episodes, desc="Training Thompson Sampling")

        for episode in episode_iter:
            total_reward = 0.0

            for t in range(len(train_df)):
                row = train_df.iloc[t]
                state = state_builder(row)

                action_idx = self._sample_action_idx(state)
                action = self.actions[action_idx]

                reward = float(
                    reward_model.reward(
                        row["pred_base_price"],
                        row["pred_demand_score"],
                        action,
                    )
                )
                total_reward += reward

                self.reward_sums_[state][action_idx] += reward
                self.reward_counts_[state][action_idx] += 1.0

            self.history_.append(total_reward)

            if show_progress and hasattr(episode_iter, "set_postfix"):
                episode_iter.set_postfix({
                    "episode_reward": f"{total_reward:,.0f}",
                })

        return self

    def action(self, row: pd.Series, state_builder: Callable[[pd.Series], Tuple]) -> float:
        state = state_builder(row)
        sums = self.reward_sums_[state]
        counts = self.reward_counts_[state]
        means = np.where(counts > 0, sums / np.maximum(counts, 1.0), self.prior_mean)
        return float(self.actions[int(np.argmax(means))])


class FittedQIterationAgent:
    def __init__(self, actions: Sequence[float], gamma: float = 0.95, n_iterations: int = 10, random_state: int = 42) -> None:
        self.actions = np.array(actions, dtype=float)
        self.gamma = gamma
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.model_: Optional[Pipeline] = None
        self.state_cols_: Optional[List[str]] = None
        self.history_: List[float] = []

    def fit(self, train_df: pd.DataFrame, state_cols: Sequence[str], reward_model: RewardModel, show_progress: bool = True) -> "FittedQIterationAgent":
        df = train_df.reset_index(drop=True).copy()
        if len(df) < 2:
            raise ValueError("train_df must contain at least 2 rows.")
        
        self.history_ = []
        
        current_states = df.iloc[:-1].reset_index(drop=True)
        next_states = df.iloc[1:].reset_index(drop=True)

        expanded_states = current_states.loc[current_states.index.repeat(len(self.actions))].reset_index(drop=True)
        expanded_states["action"] = np.tile(self.actions, len(current_states))

        rewards = reward_model.reward(
            expanded_states["pred_base_price"].to_numpy(),
            expanded_states["pred_demand_score"].to_numpy(),
            expanded_states["action"].to_numpy(),
        )

        next_expanded = next_states.loc[next_states.index.repeat(len(self.actions))].reset_index(drop=True)
        next_expanded["action"] = np.tile(self.actions, len(next_states))

        clean_state_cols = []
        for c in list(state_cols):
            if is_datetime64_any_dtype(expanded_states[c]):
                continue
            clean_state_cols.append(c)

        cat_cols = [c for c in clean_state_cols if not is_numeric_dtype(expanded_states[c])]
        num_cols = [c for c in clean_state_cols if is_numeric_dtype(expanded_states[c])] + ["action"]

        base_model = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=1,
            verbose=0
        )
        q_model = build_regression_pipeline(cat_cols, num_cols, model=base_model)
        print("Expanded training shape:", expanded_states[clean_state_cols + ["action"]].shape)

        targets = rewards.copy()
        iter_range = range(self.n_iterations)
        if show_progress:
            iter_range = tqdm(iter_range, total=self.n_iterations, desc="Training Fitted Q-Iteration")
        for iteration in iter_range:
            if show_progress and hasattr(iter_range, "write"):
                iter_range.write(f"Starting FQI iteration {iteration + 1}/{self.n_iterations}")
            else:
                print(f"Starting FQI iteration {iteration + 1}/{self.n_iterations}")
            q_model = clone(q_model)
            q_model.fit(expanded_states[clean_state_cols + ["action"]], targets)
            q_next = q_model.predict(next_expanded[clean_state_cols + ["action"]])
            q_next = q_next.reshape(len(next_states), len(self.actions)).max(axis=1)
            targets = rewards + self.gamma * np.repeat(q_next, len(self.actions))
            self.history_.append(float(np.mean(targets)))
            if show_progress and hasattr(iter_range, "set_postfix"):
                iter_range.set_postfix({"iteration": iteration + 1, "mean_target": f"{np.mean(targets):.2f}"}) 

        self.model_ = q_model
        self.state_cols_ = clean_state_cols
        return self

    def q_values(self, state_df: pd.DataFrame) -> np.ndarray:
        if self.model_ is None or self.state_cols_ is None:
            raise ValueError("Model has not been fitted yet.")
        repeated = state_df.loc[state_df.index.repeat(len(self.actions))].reset_index(drop=True).copy()
        repeated["action"] = np.tile(self.actions, len(state_df))
        q_vals = self.model_.predict(repeated[self.state_cols_ + ["action"]])
        return q_vals.reshape(len(state_df), len(self.actions))

    def action(self, row: pd.Series) -> float:
        q_vals = self.q_values(pd.DataFrame([row]))[0]
        return float(self.actions[int(np.argmax(q_vals))])


def evaluate_policy(df: pd.DataFrame, policy_name: str, policy_fn: Callable[[pd.Series], float], reward_model: RewardModel) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    records = []
    for i in range(len(df) - 1):
        row = df.iloc[i]
        action = float(policy_fn(row))
        gross = float(reward_model.gross_revenue(row["pred_base_price"], row["pred_demand_score"], action))
        penalty = float(reward_model.penalty(row["pred_base_price"], action))
        reward = gross - penalty
        demand_units = float(reward_model.conversion_units(row["pred_demand_score"], action))
        records.append({
            "policy": policy_name,
            "time_hour": row["time_hour"],
            "route": row["route"],
            "source": row["source"],
            "destination": row["destination"],
            "cab_type": row["cab_type"],
            "name": row["name"],
            "hour": int(row["hour"]),
            "weekday": int(row["weekday"]),
            "action": action,
            "pred_base_price": float(row["pred_base_price"]),
            "pred_demand_score": float(row["pred_demand_score"]),
            "expected_units": demand_units,
            "gross_revenue": gross,
            "penalty": penalty,
            "reward": reward,
        })
    return pd.DataFrame(records)


def summarize_policy_results(eval_df: pd.DataFrame) -> pd.DataFrame:
    return (
        eval_df.groupby("policy", as_index=False)
        .agg(
            total_reward=("reward", "sum"),
            avg_reward=("reward", "mean"),
            total_gross_revenue=("gross_revenue", "sum"),
            avg_action=("action", "mean"),
            avg_expected_units=("expected_units", "mean"),
            n_steps=("reward", "size"),
        )
        .sort_values("total_reward", ascending=False)
        .reset_index(drop=True)
    )


def fixed_policy(multiplier: float) -> Callable[[pd.Series], float]:
    def _policy(_: pd.Series) -> float:
        return float(multiplier)
    return _policy


def simple_peak_weather_policy(row: pd.Series) -> float:
    peak_hour = int(row["hour"]) in {7, 8, 9, 16, 17, 18, 19}
    rainy = bool(("rain" in row.index) and pd.notna(row["rain"]) and float(row["rain"]) > 0.05)
    cold = bool(("temp" in row.index) and pd.notna(row["temp"]) and float(row["temp"]) < 35)
    return 1.30 if peak_hour or rainy or cold else 1.00
