import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import tempfile
from prophet import Prophet
import skfda
from skfda.preprocessing.dim_reduction import FPCA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import holidays
from io import BytesIO
import zipfile
import platform

# ============================================================
# 中文字体设置（跨平台）
# ============================================================
def setup_chinese_font():
    """设置中文字体，支持 Windows、macOS、Linux"""
    system = platform.system()
    
    if system == "Linux":
        # Streamlit Cloud 环境
        plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "DejaVu Sans"]
    elif system == "Darwin":  # macOS
        plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
    elif system == "Windows":
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    
    plt.rcParams["axes.unicode_minus"] = False
    return plt.rcParams["font.sans-serif"][0] if plt.rcParams["font.sans-serif"] else "default"

used_font = setup_chinese_font()

st.set_page_config(
    page_title="陕西电力负荷预测系统",
    page_icon="⚡",
    layout="wide"
)

# ============================================================
# EleCurve 类（完整版）
# ============================================================
class EleCurve:

    def __init__(
        self,
        prophet_params=None,
        score_model=None,
        features=None,
        fpca_var_threshold=0.95,
        fpca_max_components=5,
        clip_prop_nonnegative=True,
        country_holidays='CN',
        holiday_years=None,
    ):
        self.prophet_params = prophet_params or {
            "yearly_seasonality": False,
            "weekly_seasonality": True,
            "daily_seasonality": False,
            "changepoint_prior_scale": 0.05
        }

        self.features = features or [
            "avg_temp",
            "max_temperature",
            "min_temperature",
            "is_weekend",
            "is_holiday",
            "festival_long",
            "festival_middle",
            "is_makeup_workday",
        ]

        self.score_model_base = score_model if score_model is not None else Ridge(alpha=1.0)
        self.fpca_var_threshold = fpca_var_threshold
        self.fpca_max_components = fpca_max_components
        self.clip_prop_nonnegative = clip_prop_nonnegative

        self.country_holidays = country_holidays
        self.holiday_years = holiday_years
        self.cn_holidays = None

        self.date_format = "%Y年%m月%d日"

        self.model_ele = None
        self.model_score = None
        self.fpca = None

        self.df_raw = None
        self._original_df_day = None
        self._original_df_prop = None
        self.df_day = None
        self.df_prop = None
        self.sf_imputation_dates = None

        self.curve_mat_train = None
        self.grid_points = None
        self.mean_func = None
        self.components = None
        self.df_scores = None
        self.pc_cols = None
        self.k = None

        self.forecast_ele = None
        self.df_pc_forecast = None
        self.X_prop_pred = None
        self.X_load_pred = None
        
        # 保存训练集和测试集数据用于可视化
        self.ele_train = None
        self.ele_test = None
        self.prop_train = None
        self.prop_test = None
        self.test_forecast = None
        self.test_curve_result = None

    def _parse_date_series(self, s: pd.Series) -> pd.Series:
        """Convert date column to pandas datetime."""
        s = s.astype(str)
    
        if pd.api.types.is_datetime64_any_dtype(s):
            return s

        if pd.api.types.is_numeric_dtype(s):
            return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")

        dt_ch = pd.to_datetime(s, format=self.date_format, errors="coerce")
        mask_fail = dt_ch.isna()
        if mask_fail.any():
            dt_slash = pd.to_datetime(s[mask_fail], format="%Y/%m/%d", errors="coerce")
            dt_ch[mask_fail] = dt_slash

        return dt_ch

    def _init_holidays(self, dates: pd.Series):
        years = sorted(dates.dt.year.unique().tolist())
        if self.holiday_years is not None:
            years = sorted(list(set(years) | set(self.holiday_years)))
        self.cn_holidays = holidays.country_holidays(self.country_holidays, years=years)

    def _add_calendar_features(self, df_day: pd.DataFrame) -> pd.DataFrame:
        if "date" in df_day.columns:
            date_col = "date"
        else:
            date_col = "ds"

        if self.cn_holidays is None:
            self._init_holidays(df_day[date_col])

        df = df_day.copy()
        d = df[date_col]

        df["is_weekend"] = (d.dt.weekday >= 5).astype(int)

        is_holiday = d.dt.date.map(lambda x: x in self.cn_holidays)
        df["is_holiday"] = is_holiday.astype(int)

        df["is_makeup_workday"] = ((d.dt.weekday >= 5) & (~is_holiday)).astype(int)

        hol_names = d.dt.date.map(lambda x: self.cn_holidays.get(x, ""))

        is_spring = hol_names.str.contains("春节", na=False)
        is_national = hol_names.str.contains("国庆", na=False)

        is_qingming = hol_names.str.contains("清明", na=False)

        df["festival_qingming"] = (is_qingming & is_holiday).astype(int)
        df["festival_long"] = ((is_spring | is_national) & is_holiday).astype(int)
        df["festival_middle"] = (is_holiday & (~df["festival_long"].astype(bool))).astype(int)
        df["is_spring_festival"] = is_spring.astype(int)

        return df

    def prepare_data(self, df):
        df = df.copy()
        df = df.loc[:, ["date", "time", "ele", "temp"]].copy()
        df["date"] = self._parse_date_series(df["date"])
        df = df.sort_values(["date", "time"]).reset_index(drop=True)
        self.df_raw = df

        df_day = df.groupby("date", as_index=False).agg(
            ele_day=("ele", "sum"),
            avg_temp=("temp", "mean"),
            max_temperature=("temp", "max"),
            min_temperature=("temp", "min")
        )

        df_day = self._add_calendar_features(df_day)
        df_day = df_day.rename(columns={"date": "ds", "ele_day": "y"})
        df_day = df_day.sort_values("ds").reset_index(drop=True)
        self._original_df_day = df_day.copy()

        self.sf_imputation_dates = df_day[df_day["is_spring_festival"] == 1]["ds"].unique()
        self.sf_imputation_dates = pd.to_datetime(self.sf_imputation_dates)

        df_prop = df.copy()
        df_prop["ele_day_sum"] = df_prop.groupby("date")["ele"].transform("sum")
        df_prop = df_prop[df_prop["ele_day_sum"] > 0].copy()
        df_prop["ele_prop"] = df_prop["ele"] / df_prop["ele_day_sum"]

        self._original_df_prop = df_prop.copy()

        self.df_day = self._original_df_day.copy()
        self.df_prop = self._original_df_prop.copy()

        return self

    def _impute_sf_segment_helper(self, df_day_train_for_imputation, df_prop_train_for_imputation, sf_dates_df_day_to_predict):
        temp_model = EleCurve(
            prophet_params=self.prophet_params,
            score_model=self.score_model_base,
            features=self.features,
            fpca_var_threshold=self.fpca_var_threshold,
            fpca_max_components=self.fpca_max_components,
            clip_prop_nonnegative=self.clip_prop_nonnegative,
            country_holidays=self.country_holidays,
            holiday_years=self.holiday_years
        )

        temp_model.df_day = df_day_train_for_imputation.copy()
        temp_model.df_prop = df_prop_train_for_imputation.copy()

        temp_model.ele_fit(temp_model.df_day)
        temp_model.prop_fpca_fit(temp_model.df_prop)
        temp_model.prop_score_fit(temp_model.df_day)

        sf_dates_for_pred = sf_dates_df_day_to_predict.copy()
        
        for col in ["is_holiday", "festival_long", "festival_middle", "is_makeup_workday"]:
            if col in sf_dates_for_pred.columns:
                sf_dates_for_pred[col] = 0

        sf_ele_forecast = temp_model.ele_predict(sf_dates_for_pred.drop(columns='y', errors='ignore'))
        sf_df_pc_forecast = temp_model.prop_score_predict(sf_dates_for_pred.drop(columns='y', errors='ignore'))

        X_prop_pred_sf = sf_df_pc_forecast[temp_model.pc_cols].values @ temp_model.components + temp_model.mean_func
        row_sum = X_prop_pred_sf.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        X_prop_pred_sf = X_prop_pred_sf / row_sum
        if temp_model.clip_prop_nonnegative:
            X_prop_pred_sf = np.clip(X_prop_pred_sf, 0, None)
            row_sum = X_prop_pred_sf.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            X_prop_pred_sf = X_prop_pred_sf / row_sum

        ele_pred_series_sf = sf_ele_forecast.set_index("ds").loc[sf_df_pc_forecast.index, "yhat"].values
        X_load_pred_sf = X_prop_pred_sf * ele_pred_series_sf[:, None]

        imputed_df_day = pd.DataFrame({
            "ds": sf_dates_for_pred["ds"],
            "y": ele_pred_series_sf
        }).merge(sf_dates_for_pred.drop(columns=['y', 'is_spring_festival'], errors='ignore'), on='ds', how='left')
        imputed_df_day['is_spring_festival'] = 1 

        imputed_df_prop_list = []
        for i, date_obj in enumerate(sf_dates_for_pred["ds"]):
            df_temp = pd.DataFrame({
                "date": date_obj,
                "time": temp_model.grid_points,
                "ele_prop": X_prop_pred_sf[i],
                "ele": X_load_pred_sf[i]
            })
            imputed_df_prop_list.append(df_temp)
        imputed_df_prop = pd.concat(imputed_df_prop_list, ignore_index=True)
        
        imputed_df_prop = imputed_df_prop.merge(
            imputed_df_day[['ds', 'y']].rename(columns={'ds':'date', 'y':'ele_day_sum'}),
            on='date',
            how='left'
        )

        return imputed_df_day, imputed_df_prop

    def perform_sf_imputation(self, sf_dates_to_impute: pd.Series = None):
        if self._original_df_day is None or self._original_df_prop is None:
            raise ValueError("请先调用 prepare_data()")

        dates_to_impute_series = pd.Series([], dtype='datetime64[ns]')

        if sf_dates_to_impute is not None:
            dates_to_impute_series = pd.Series(sf_dates_to_impute).dt.normalize().unique()
        elif self.sf_imputation_dates is not None and len(self.sf_imputation_dates) > 0:
            dates_to_impute_series = pd.Series(self.sf_imputation_dates).dt.normalize().unique()
        else:
            self.df_day = self._original_df_day.copy()
            self.df_prop = self._original_df_prop.copy()
            return

        if dates_to_impute_series.empty:
            self.df_day = self._original_df_day.copy()
            self.df_prop = self._original_df_prop.copy()
            return

        mask_sf = self._original_df_day["ds"].isin(dates_to_impute_series)
        df_day_sf_original = self._original_df_day[mask_sf].copy()
        df_day_no_sf = self._original_df_day[~mask_sf].copy()

        if df_day_sf_original.empty:
            self.df_day = self._original_df_day.copy()
            self.df_prop = self._original_df_prop.copy()
            return

        mask_sf_prop = self._original_df_prop["date"].isin(dates_to_impute_series)
        df_prop_no_sf = self._original_df_prop[~mask_sf_prop].copy()

        imputed_df_day, imputed_df_prop = self._impute_sf_segment_helper(
            df_day_no_sf, df_prop_no_sf, df_day_sf_original
        )

        self.df_day = pd.concat([df_day_no_sf, imputed_df_day]).sort_values("ds").reset_index(drop=True)
        self.df_prop = pd.concat([df_prop_no_sf, imputed_df_prop]).sort_values(["date", "time"]).reset_index(drop=True)

    def prepare_future_data(self, df_pred):
        df_pred = df_pred.copy()
        need_cols = ["date", "time", "temp"]
        missing_cols = [c for c in need_cols if c not in df_pred.columns]
        if missing_cols:
            raise ValueError(f"df_pred 缺少必要列: {missing_cols}")

        df_pred["date"] = self._parse_date_series(df_pred["date"])
        df_pred = df_pred.sort_values(["date", "time"]).reset_index(drop=True)

        df_day_pred = df_pred.groupby("date", as_index=False).agg(
            avg_temp=("temp", "mean"),
            max_temperature=("temp", "max"),
            min_temperature=("temp", "min")
        )

        df_day_pred = self._add_calendar_features(df_day_pred)
        df_day_pred = df_day_pred.rename(columns={"date": "ds"})
        df_day_pred = df_day_pred.sort_values("ds").reset_index(drop=True)

        return df_day_pred

    def ele_fit(self, ele_train):
        model = Prophet(**self.prophet_params)
        for r in self.features:
            if r in ele_train.columns:
                model.add_regressor(r)
        model.fit(ele_train.copy())
        self.model_ele = model
        return self

    def ele_predict(self, ele_test, return_metrics=True):
        if self.model_ele is None:
            raise ValueError("请先调用 ele_fit()")

        future = ele_test[["ds"] + [f for f in self.features if f in ele_test.columns]].copy()
        for f in self.features:
            if f not in future.columns:
                future[f] = 0

        forecast = self.model_ele.predict(future)
        self.forecast_ele = forecast.copy()

        if return_metrics and "y" in ele_test.columns:
            y_true = ele_test["y"].values
            y_pred = forecast["yhat"].values
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6)))
            metrics = {"mae": mae, "rmse": rmse, "mape": mape}
            return forecast, metrics

        return forecast

    def prop_fpca_fit(self, prop_train, plot=False):
        curve_mat_train = prop_train.pivot(index="date", columns="time", values="ele_prop")
        curve_mat_train = curve_mat_train.sort_index(axis=1)
        curve_mat_train = curve_mat_train.dropna(axis=0).copy()

        X_train_prop = curve_mat_train.to_numpy()
        grid_points = curve_mat_train.columns.to_numpy()

        fd_train = skfda.FDataGrid(data_matrix=X_train_prop, grid_points=grid_points)

        n_comp_init = min(self.fpca_max_components, X_train_prop.shape[0], X_train_prop.shape[1])
        fpca_tmp = FPCA(n_components=n_comp_init)
        fpca_tmp.fit(fd_train)

        cum_ratio = np.cumsum(fpca_tmp.explained_variance_ratio_)
        k = np.argmax(cum_ratio >= self.fpca_var_threshold) + 1
        if k == 0 and len(cum_ratio) > 0: k = 1
        if k == 0 and n_comp_init > 0: k = n_comp_init
        if k == 0:
            raise ValueError("FPCA 无法确定组件数量")

        fpca = FPCA(n_components=k)
        fpca.fit(fd_train)

        scores_train = fpca.transform(fd_train)
        df_scores = pd.DataFrame(
            scores_train,
            index=curve_mat_train.index,
            columns=[f"PC{i+1}" for i in range(scores_train.shape[1])]
        ).reset_index()

        mean_func = fpca.mean_.data_matrix
        if mean_func.ndim == 2:
            mean_func = mean_func[..., 0]
        mean_func = np.asarray(mean_func).reshape(-1)

        components = fpca.components_.data_matrix
        if components.ndim == 3:
            components = components[..., 0]

        self.fpca = fpca
        self.curve_mat_train = curve_mat_train
        self.grid_points = grid_points
        self.mean_func = mean_func
        self.components = components
        self.df_scores = df_scores
        self.pc_cols = [c for c in df_scores.columns if c.startswith("PC")]
        self.k = k

        return {"fpca": fpca, "k": k, "cum_ratio": cum_ratio, "df_scores": df_scores}

    def prop_score_fit(self, ele_train):
        if self.df_scores is None:
            raise ValueError("请先调用 prop_fpca_fit()")

        df_pc_model = self.df_scores.merge(
            ele_train[["ds"] + [f for f in self.features if f in ele_train.columns]].rename(columns={"ds": "date"}),
            on="date", how="left"
        ).sort_values("date").reset_index(drop=True)

        df_pc_model.dropna(subset=self.features, inplace=True)
        X_pc_train = df_pc_model[self.features]
        Y_pc_train = df_pc_model[self.pc_cols]

        if X_pc_train.empty or Y_pc_train.empty:
            raise ValueError("没有有效数据训练 FPCA 分数模型")

        model_score = MultiOutputRegressor(self.score_model_base)
        model_score.fit(X_pc_train, Y_pc_train)
        self.model_score = model_score
        return self

    def prop_score_predict(self, ele_test):
        if self.model_score is None:
            raise ValueError("请先调用 prop_score_fit()")
        if self.fpca is None:
            raise ValueError("请先调用 prop_fpca_fit()")

        df_test_feat = ele_test[["ds"] + [f for f in self.features if f in ele_test.columns]].rename(columns={"ds": "date"}).copy()
        for f in self.features:
            if f not in df_test_feat.columns:
                df_test_feat[f] = 0

        X_pc_test = df_test_feat[self.features]
        Y_pc_pred = self.model_score.predict(X_pc_test)

        df_pc_forecast = pd.DataFrame(Y_pc_pred, index=df_test_feat["date"], columns=self.pc_cols)
        self.df_pc_forecast = df_pc_forecast
        return df_pc_forecast

    def ele_curve_predict(self, ele_test, prop_test=None, return_metrics=True):
        if self.forecast_ele is None:
            self.ele_predict(ele_test, return_metrics=False)
        if self.df_pc_forecast is None:
            self.prop_score_predict(ele_test)
        if self.components is None or self.mean_func is None:
            raise ValueError("FPCA 组件未初始化")

        common_ds = pd.Index(self.forecast_ele['ds'].dt.normalize()).intersection(
            pd.Index(self.df_pc_forecast.index.normalize())
        )
        
        forecast_ele_aligned = self.forecast_ele[self.forecast_ele['ds'].dt.normalize().isin(common_ds)].set_index('ds').sort_index()
        df_pc_forecast_aligned = self.df_pc_forecast.loc[common_ds].sort_index()
        
        X_prop_pred = df_pc_forecast_aligned[self.pc_cols].values @ self.components + self.mean_func
        row_sum = X_prop_pred.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        X_prop_pred = X_prop_pred / row_sum

        if self.clip_prop_nonnegative:
            X_prop_pred = np.clip(X_prop_pred, 0, None)
            row_sum = X_prop_pred.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            X_prop_pred = X_prop_pred / row_sum

        self.X_prop_pred = X_prop_pred
        ele_pred_series = forecast_ele_aligned["yhat"].values
        X_load_pred = X_prop_pred * ele_pred_series[:, None]
        self.X_load_pred = X_load_pred

        result = {
            "dates": df_pc_forecast_aligned.index,
            "times": self.grid_points,
            "ele_pred_series": ele_pred_series,
            "df_pc_forecast": df_pc_forecast_aligned.copy(),
            "X_prop_pred": X_prop_pred,
            "X_load_pred": X_load_pred
        }

        if return_metrics and prop_test is not None:
            curve_mat_test_load = prop_test.pivot(index="date", columns="time", values="ele")
            curve_mat_test_load = curve_mat_test_load.sort_index(axis=1)

            common_times = pd.Index(self.grid_points).intersection(curve_mat_test_load.columns)
            if not common_times.empty:
                curve_mat_test_load = curve_mat_test_load.loc[:, common_times].dropna(axis=0)
                X_load_true = curve_mat_test_load.to_numpy()

                pred_idx = pd.Index(df_pc_forecast_aligned.index)
                true_idx = curve_mat_test_load.index
                common_dates = pred_idx.intersection(true_idx)

                if not common_dates.empty:
                    pred_pos = pred_idx.get_indexer(common_dates)
                    true_pos = true_idx.get_indexer(common_dates)

                    X_load_pred_aligned = X_load_pred[pred_pos]
                    X_load_true_aligned = X_load_true[true_pos]

                    curve_mae = np.mean(np.abs(X_load_true_aligned - X_load_pred_aligned))
                    curve_rmse = np.sqrt(np.mean((X_load_true_aligned - X_load_pred_aligned) ** 2))
                    result["curve_metrics"] = {"mae": curve_mae, "rmse": curve_rmse}

        return result

    def predict_future_curve(self, df_pred, return_long=True):
        df_day_pred = self.prepare_future_data(df_pred)
        forecast_ele = self.ele_predict(df_day_pred, return_metrics=False)

        # 4月5日特殊处理
        target_date = pd.to_datetime('2026-04-05').normalize()
        date_before = pd.to_datetime('2026-04-04').normalize()
        date_after = pd.to_datetime('2026-04-06').normalize()

        forecast_df_indexed = forecast_ele.set_index('ds')
        if target_date in forecast_df_indexed.index and date_before in forecast_df_indexed.index and date_after in forecast_df_indexed.index:
            val_before = forecast_df_indexed.loc[date_before, 'yhat']
            val_after = forecast_df_indexed.loc[date_after, 'yhat']
            forecast_df_indexed.loc[target_date, 'yhat'] = (val_before + val_after) / 2

        forecast_ele = forecast_df_indexed.reset_index()
        df_pc_forecast = self.prop_score_predict(df_day_pred)

        self.forecast_ele = forecast_ele
        result = self.ele_curve_predict(ele_test=df_day_pred, prop_test=None, return_metrics=False)

        result["df_day_pred"] = df_day_pred
        result["forecast_ele"] = forecast_ele
        result["pc_score"] = df_pc_forecast

        if return_long:
            dates = result["dates"]
            times = result["times"]

            df_curve_pred = pd.DataFrame(result["X_load_pred"], index=dates, columns=times).reset_index().rename(columns={"index": "date"})
            df_curve_pred_long = df_curve_pred.melt(id_vars="date", var_name="time", value_name="ele_pred")

            df_prop_pred = pd.DataFrame(result["X_prop_pred"], index=dates, columns=times).reset_index().rename(columns={"index": "date"})
            df_prop_pred_long = df_prop_pred.melt(id_vars="date", var_name="time", value_name="prop_pred")

            result["df_curve_pred_wide"] = df_curve_pred
            result["df_curve_pred_long"] = df_curve_pred_long
            result["df_prop_pred_wide"] = df_prop_pred
            result["df_prop_pred_long"] = df_prop_pred_long

        return result

    def split_last_n_days(self, test_days=3):
        if self.df_day is None or self.df_prop is None:
            raise ValueError("请先调用 prepare_data()")

        if test_days is None:
            ele_train = self.df_day.copy()
            prop_train = self.df_prop.copy()
            return ele_train, prop_train, pd.DataFrame(), pd.DataFrame()

        if test_days <= 0:
            raise ValueError("test_days 必须是正整数或 None")

        if len(self.df_day) < test_days:
            test_days = max(1, len(self.df_day) // 5)

        ele_train = self.df_day.iloc[:-test_days].copy()
        ele_test = self.df_day.iloc[-test_days:].copy()

        train_dates = ele_train["ds"]
        test_dates = ele_test["ds"]

        prop_train = self.df_prop[self.df_prop["date"].isin(train_dates)].copy()
        prop_test = self.df_prop[self.df_prop["date"].isin(test_dates)].copy()

        # 保存到实例变量供可视化使用
        self.ele_train = ele_train
        self.ele_test = ele_test
        self.prop_train = prop_train
        self.prop_test = prop_test

        return ele_train, ele_test, prop_train, prop_test


# ============================================================
# 数据处理函数
# ============================================================
def process_weather_data(uploaded_file):
    """处理天气数据"""
    try:
        df = pd.read_excel(uploaded_file)
        
        if 'record_time' not in df.columns:
            st.error(f"天气数据缺少 'record_time' 列！实际列名: {list(df.columns)}")
            return None
        
        if 'value' not in df.columns:
            st.error(f"天气数据缺少 'value' 列！")
            return None
        
        df['record_time'] = pd.to_datetime(df['record_time'], errors='coerce')
        df = df.dropna(subset=['record_time'])
        df = df.sort_values('record_time')
        
        df = df.set_index('record_time')
        df_15min = df.resample('15min').interpolate(method='time')
        df_15min = df_15min.reset_index()
        
        df_15min['date'] = (
            df_15min['record_time'].dt.year.astype(str) + '年' +
            df_15min['record_time'].dt.month.astype(str) + '月' +
            df_15min['record_time'].dt.day.astype(str) + '日'
        )
        
        df_15min['time'] = df_15min.groupby('date').cumcount() + 1
        df_15min = df_15min[df_15min['time'] <= 96]
        
        df_15min = df_15min[['date', 'time', 'value']]
        df_15min = df_15min.rename(columns={'value': 'temp'})
        
        return df_15min
    except Exception as e:
        st.error(f"处理天气数据出错: {e}")
        return None


def process_single_day_data(filepath, filename):
    """处理单个用户侧用电量文件"""
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    
    if not date_match:
        return pd.DataFrame()
    
    date_str = date_match.group(1)
    date_obj = pd.to_datetime(date_str)
    formatted_date = f"{date_obj.year}年{date_obj.month}月{date_obj.day}日"

    try:
        df_raw = pd.read_excel(filepath, header=0)
    except Exception:
        return pd.DataFrame()

    times = []
    elect_sums = []

    for i in range(1, 97):
        col_name = f"段{i}"
        if col_name in df_raw.columns:
            segment_data = pd.to_numeric(df_raw[col_name], errors='coerce')
            daily_sum = segment_data.sum()
            times.append(i)
            elect_sums.append(daily_sum)
        else:
            times.append(i)
            elect_sums.append(0.0)

    df_day = pd.DataFrame({
        'date': formatted_date,
        'time': times,
        'ele': elect_sums
    })

    return df_day


def consolidate_customer_data(uploaded_files):
    """整合所有用户侧用电量数据"""
    all_data_frames = []
    
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        df_day = process_single_day_data(tmp_path, filename)
        os.unlink(tmp_path)
        
        if not df_day.empty:
            all_data_frames.append(df_day)

    if not all_data_frames:
        return pd.DataFrame()

    final_df = pd.concat(all_data_frames, ignore_index=True)
    
    final_df['sort_date_key'] = pd.to_datetime(
        final_df['date'].str.replace('年', '-').str.replace('月', '-').str.replace('日', ''),
        format='%Y-%m-%d',
        errors='coerce'
    )
    final_df = final_df.dropna(subset=['sort_date_key'])
    final_df = final_df.sort_values(by=['sort_date_key', 'time']).drop(columns=['sort_date_key']).reset_index(drop=True)

    return final_df


def merge_weather_and_customer(weather_df, customer_df):
    """合并天气和用电数据"""
    weather_df['date_key'] = pd.to_datetime(
        weather_df['date'].str.replace('年', '-').str.replace('月', '-').str.replace('日', ''),
        format='%Y-%m-%d',
        errors='coerce'
    )
    customer_df['date_key'] = pd.to_datetime(
        customer_df['date'].str.replace('年', '-').str.replace('月', '-').str.replace('日', ''),
        format='%Y-%m-%d',
        errors='coerce'
    )
    
    max_customer_date = customer_df['date_key'].max()
    min_customer_date = customer_df['date_key'].min()
    
    weather_historical = weather_df[
        (weather_df['date_key'] >= min_customer_date) & 
        (weather_df['date_key'] <= max_customer_date)
    ].copy()
    
    merged_df = customer_df.merge(
        weather_historical[['date', 'time', 'temp']],
        on=['date', 'time'],
        how='left'
    )
    
    merged_df = merged_df.drop(columns=['date_key'])
    merged_df = merged_df[['date', 'time', 'ele', 'temp']]
    
    return merged_df


def create_future_weather(weather_df, customer_df):
    """创建未来天气数据"""
    weather_df['date_key'] = pd.to_datetime(
        weather_df['date'].str.replace('年', '-').str.replace('月', '-').str.replace('日', ''),
        format='%Y-%m-%d',
        errors='coerce'
    )
    customer_df['date_key'] = pd.to_datetime(
        customer_df['date'].str.replace('年', '-').str.replace('月', '-').str.replace('日', ''),
        format='%Y-%m-%d',
        errors='coerce'
    )
    
    max_customer_date = customer_df['date_key'].max()
    
    future_weather = weather_df[weather_df['date_key'] > max_customer_date].copy()
    future_weather = future_weather[['date', 'time', 'temp']]
    
    return future_weather


# ============================================================
# 绘图函数（中文标签）
# ============================================================
def plot_daily_forecast(df_forecast, start_date=None, end_date=None, title="日总用电量预测"):
    """绘制日总负荷预测"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    plot_df = df_forecast.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    
    if start_date and end_date:
        plot_df = plot_df[(plot_df["date"] >= start_date) & (plot_df["date"] <= end_date)]
    
    ax.plot(plot_df["date"], plot_df["ele_day_pred"], marker="o", linewidth=2, markersize=4, color="#1f77b4")
    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel("日总用电量 (kWh)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def plot_validation_daily(ele_train, ele_test, forecast_test):
    """绘制验证集日总负荷对比"""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # 训练集
    ax.plot(ele_train["ds"], ele_train["y"], label="训练集实际值", color="#1f77b4", linewidth=2)
    
    # 测试集实际值
    ax.plot(ele_test["ds"], ele_test["y"], label="测试集实际值", color="#2ca02c", linewidth=2, marker="o", markersize=4)
    
    # 测试集预测值
    forecast_test_aligned = forecast_test.set_index('ds').loc[ele_test["ds"]]
    ax.plot(ele_test["ds"], forecast_test_aligned["yhat"], label="测试集预测值", color="#ff7f0e", linewidth=2, marker="s", markersize=4, linestyle="--")
    
    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel("日总用电量 (kWh)", fontsize=12)
    ax.set_title("验证集日总负荷预测对比", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def plot_96point_curve(result, date_str, title_prefix="预测负荷曲线"):
    """绘制96点负荷曲线"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    times = result["times"]
    dates = result["dates"]
    X_load_pred = result["X_load_pred"]
    
    dates_str = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in dates]
    
    if date_str in dates_str:
        date_idx = dates_str.index(date_str)
        dt = dates[date_idx]
        ax.plot(times, X_load_pred[date_idx], marker="o", linestyle="--", linewidth=1.5, markersize=2, color="#ff7f0e")
        ax.set_xlabel("时段 (1-96)", fontsize=12)
        ax.set_ylabel("负荷 (kWh)", fontsize=12)
        ax.set_title(f"{title_prefix} - {pd.Timestamp(dt).strftime('%Y-%m-%d')}", fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
    else:
        ax.text(0.5, 0.5, f"日期 {date_str} 不在预测范围内", 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
    
    return fig


def plot_validation_curve_comparison(model, test_date_str):
    """绘制验证集某一天的96点曲线对比"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # 获取测试集曲线结果
    if model.test_curve_result is None:
        # 计算测试集曲线预测
        model.test_curve_result = model.ele_curve_predict(
            ele_test=model.ele_test,
            prop_test=model.prop_test,
            return_metrics=True
        )
    
    result = model.test_curve_result
    times = result["times"]
    dates = result["dates"]
    X_load_pred = result["X_load_pred"]
    
    # 获取真实负荷曲线
    prop_test = model.prop_test
    curve_mat_test = prop_test.pivot(index="date", columns="time", values="ele")
    curve_mat_test = curve_mat_test.sort_index(axis=1)
    
    dates_str = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in dates]
    
    if test_date_str in dates_str:
        date_idx = dates_str.index(test_date_str)
        dt = dates[date_idx]
        
        # 预测值
        ax.plot(times, X_load_pred[date_idx], marker="o", linestyle="--", linewidth=1.5, markersize=2, 
                color="#ff7f0e", label="预测值")
        
        # 真实值
        if dt in curve_mat_test.index:
            true_curve = curve_mat_test.loc[dt].values
            ax.plot(times, true_curve, marker="s", linestyle="-", linewidth=1.5, markersize=2, 
                    color="#2ca02c", label="实际值", alpha=0.7)
        
        ax.set_xlabel("时段 (1-96)", fontsize=12)
        ax.set_ylabel("负荷 (kWh)", fontsize=12)
        ax.set_title(f"验证集负荷曲线对比 - {pd.Timestamp(dt).strftime('%Y-%m-%d')}", fontsize=14)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
    
    return fig


def to_excel_bytes(df):
    """将DataFrame转换为Excel字节流"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()


# ============================================================
# Streamlit 主界面（中文）
# ============================================================
def main():
    st.title("⚡ 陕西电力负荷预测系统")
    st.markdown(f"*当前使用字体: {used_font}*")
    st.markdown("---")
    
    # 侧边栏 - 文件上传和配置
    with st.sidebar:
        st.header("📁 数据上传")
        
        weather_file = st.file_uploader(
            "上传天气数据 (weather_hourly_data.xlsx)",
            type=["xlsx"],
            key="weather"
        )
        
        customer_files = st.file_uploader(
            "上传用户侧用电量数据 (多个文件)",
            type=["xlsx"],
            accept_multiple_files=True,
            key="customer"
        )
        
        st.markdown("---")
        st.header("⚙️ 预测配置")
        
        test_days = st.number_input(
            "测试集天数",
            min_value=3,
            max_value=30,
            value=7,
            help="用于验证的历史数据天数"
        )
        
        sf_start = st.date_input(
            "春节填充开始日期",
            value=pd.to_datetime("2026-02-09"),
            help="需要填充的春节假期开始日期"
        )
        
        sf_end = st.date_input(
            "春节填充结束日期",
            value=pd.to_datetime("2026-02-25"),
            help="需要填充的春节假期结束日期"
        )
        
        predict_start = st.date_input(
            "预测起始日期",
            value=pd.to_datetime("2026-04-02"),
            help="需要输出预测结果的起始日期"
        )
        
        predict_end = st.date_input(
            "预测结束日期",
            value=pd.to_datetime("2026-04-14"),
            help="需要输出预测结果的结束日期"
        )
        
        st.markdown("---")
        
        process_btn = st.button("🚀 开始处理数据并预测", type="primary", use_container_width=True)
    
    # 初始化session_state
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'prediction_done' not in st.session_state:
        st.session_state.prediction_done = False
    
    # 处理数据并预测
    if process_btn and weather_file and customer_files:
        try:
            with st.spinner("正在处理数据..."):
                # 处理天气数据
                st.info("📊 步骤1/4: 处理天气数据...")
                weather_df = process_weather_data(weather_file)
                if weather_df is None:
                    st.error("天气数据处理失败")
                    return
                st.success(f"✅ 天气数据处理完成，共 {len(weather_df)} 条记录")
                
                # 整合用户用电数据
                st.info("📊 步骤2/4: 整合用户用电数据...")
                customer_df = consolidate_customer_data(customer_files)
                if customer_df.empty:
                    st.error("用户用电数据处理失败")
                    return
                st.success(f"✅ 用户用电数据整合完成，共 {len(customer_df)} 条记录")
                
                # 合并数据
                st.info("📊 步骤3/4: 合并历史数据...")
                merged_df = merge_weather_and_customer(weather_df, customer_df)
                st.success(f"✅ 历史数据合并完成，共 {len(merged_df)} 条记录")
                
                # 创建未来天气数据
                st.info("📊 步骤4/4: 准备未来天气数据...")
                future_weather_df = create_future_weather(weather_df, customer_df)
                st.success(f"✅ 未来天气数据准备完成，共 {len(future_weather_df)} 条记录")
                
                st.session_state.merged_df = merged_df
                st.session_state.future_weather_df = future_weather_df
                
                # 数据预览
                st.markdown("---")
                st.subheader("📋 数据预览")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**历史数据 (含用电量和温度)**")
                    st.dataframe(merged_df.head(10), use_container_width=True)
                
                with col2:
                    st.write("**未来天气数据**")
                    st.dataframe(future_weather_df.head(10), use_container_width=True)
            
            # 模型训练和预测
            with st.spinner("正在训练模型并预测..."):
                st.markdown("---")
                st.subheader("🔮 模型训练与预测")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 初始化模型
                status_text.text("初始化模型...")
                progress_bar.progress(10)
                model = EleCurve()
                
                # 准备数据
                status_text.text("准备数据...")
                progress_bar.progress(20)
                model.prepare_data(merged_df)
                
                # 春节填充
                status_text.text("执行春节数据填充...")
                progress_bar.progress(30)
                custom_sf_dates = pd.date_range(start=sf_start, end=sf_end)
                model.perform_sf_imputation(sf_dates_to_impute=custom_sf_dates)
                
                # 分割数据
                status_text.text("分割训练集和测试集...")
                progress_bar.progress(40)
                ele_train, ele_test, prop_train, prop_test = model.split_last_n_days(test_days=test_days)
                
                # 训练日用电量模型
                status_text.text("训练日用电量预测模型...")
                progress_bar.progress(50)
                model.ele_fit(ele_train)
                
                # 预测测试集
                status_text.text("评估测试集...")
                progress_bar.progress(60)
                forecast_ele, ele_metrics = model.ele_predict(ele_test)
                model.test_forecast = forecast_ele  # 保存测试集预测结果
                
                # FPCA
                status_text.text("执行FPCA分析...")
                progress_bar.progress(70)
                model.prop_fpca_fit(prop_train)
                
                # 训练分数模型
                status_text.text("训练负荷曲线模型...")
                progress_bar.progress(80)
                model.prop_score_fit(ele_train)
                
                # 计算测试集曲线预测
                status_text.text("计算测试集曲线预测...")
                model.test_curve_result = model.ele_curve_predict(
                    ele_test=ele_test,
                    prop_test=prop_test,
                    return_metrics=True
                )
                
                # 预测未来
                status_text.text("预测未来负荷...")
                progress_bar.progress(90)
                future_result = model.predict_future_curve(future_weather_df, return_long=True)
                
                progress_bar.progress(100)
                status_text.text("训练完成！")
                
                # 保存结果到session_state
                st.session_state.model = model
                st.session_state.future_result = future_result
                st.session_state.ele_metrics = ele_metrics
                st.session_state.ele_train = ele_train
                st.session_state.ele_test = ele_test
                st.session_state.prop_test = prop_test
                st.session_state.forecast_ele = forecast_ele
                st.session_state.prediction_done = True
                
                # 准备日总预测数据
                df_apr_day_forecast = future_result["forecast_ele"][["ds", "yhat"]].copy()
                df_apr_day_forecast.rename(columns={"ds": "date", "yhat": "ele_day_pred"}, inplace=True)
                st.session_state.df_apr_day_forecast = df_apr_day_forecast
                
                # 显示评估指标
                st.success("✅ 模型训练完成！")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("日总 MAE", f"{ele_metrics['mae']:.2f}")
                with col2:
                    st.metric("日总 RMSE", f"{ele_metrics['rmse']:.2f}")
                with col3:
                    st.metric("日总 MAPE", f"{ele_metrics['mape']*100:.2f}%")
                with col4:
                    if model.test_curve_result and "curve_metrics" in model.test_curve_result:
                        st.metric("曲线 MAE", f"{model.test_curve_result['curve_metrics']['mae']:.2f}")
        
        except Exception as e:
            st.error(f"处理过程中出错: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    elif process_btn:
        st.error("请先上传所有必需的数据文件！")
    
    # 显示验证集结果
    if st.session_state.prediction_done:
        st.markdown("---")
        st.subheader("📊 验证集评估结果")
        
        model = st.session_state.model
        ele_train = st.session_state.ele_train
        ele_test = st.session_state.ele_test
        forecast_ele = st.session_state.forecast_ele
        prop_test = st.session_state.prop_test
        
        # 日总负荷对比图
        st.markdown("#### 日总负荷预测对比")
        fig_val = plot_validation_daily(ele_train, ele_test, forecast_ele)
        st.pyplot(fig_val)
        plt.close(fig_val)
        
        # 96点曲线对比
        st.markdown("#### 96点负荷曲线对比")
        
        # 获取测试集日期
        test_dates = ele_test["ds"].dt.strftime('%Y-%m-%d').tolist()
        
        if len(test_dates) > 0:
            selected_test_date = st.selectbox(
                "选择测试集日期查看96点曲线对比",
                options=test_dates
            )
            
            fig_val_curve = plot_validation_curve_comparison(model, selected_test_date)
            st.pyplot(fig_val_curve)
            plt.close(fig_val_curve)
        else:
            st.warning("没有可用的测试集日期")
        
        # 测试集数据表格
        st.markdown("#### 测试集日总负荷数据")
        test_comparison = pd.DataFrame({
            "日期": ele_test["ds"].dt.strftime('%Y-%m-%d'),
            "实际值": ele_test["y"].values,
            "预测值": forecast_ele.set_index('ds').loc[ele_test["ds"]]["yhat"].values
        })
        test_comparison["误差"] = test_comparison["预测值"] - test_comparison["实际值"]
        test_comparison["误差率(%)"] = (test_comparison["误差"] / test_comparison["实际值"] * 100).round(2)
        st.dataframe(test_comparison, use_container_width=True)
    
    # 显示预测结果
    if st.session_state.prediction_done:
        st.markdown("---")
        st.subheader("📈 未来负荷预测结果")
        
        future_result = st.session_state.future_result
        df_apr_day_forecast = st.session_state.df_apr_day_forecast
        
        # 筛选指定日期范围
        df_apr_day_forecast["date"] = pd.to_datetime(df_apr_day_forecast["date"])
        mask = (df_apr_day_forecast["date"] >= pd.to_datetime(predict_start)) & \
               (df_apr_day_forecast["date"] <= pd.to_datetime(predict_end))
        df_filtered = df_apr_day_forecast[mask]
        
        # 计算总量
        month_total = df_filtered["ele_day_pred"].sum()
        
        st.metric(
            f"📅 {predict_start.strftime('%Y-%m-%d')} 至 {predict_end.strftime('%Y-%m-%d')} 总用电量预测",
            f"{month_total:,.2f} kWh"
        )
        
        # 日总负荷预测图
        st.markdown("#### 日总用电量预测")
        fig1 = plot_daily_forecast(df_apr_day_forecast, pd.to_datetime(predict_start), pd.to_datetime(predict_end), 
                                   title="未来日总用电量预测")
        st.pyplot(fig1)
        plt.close(fig1)
        
        # 96点曲线
        st.markdown("#### 96点负荷曲线预测")
        
        curve_dates = future_result["df_curve_pred_wide"]["date"].unique()
        curve_dates_filtered = []
        for d in curve_dates:
            d_dt = pd.to_datetime(d)
            if pd.to_datetime(predict_start) <= d_dt <= pd.to_datetime(predict_end):
                curve_dates_filtered.append(d_dt.strftime('%Y-%m-%d'))
        
        if len(curve_dates_filtered) > 0:
            selected_date = st.selectbox(
                "选择日期查看未来96点负荷曲线",
                options=curve_dates_filtered,
                key="future_curve_select"
            )
            
            fig2 = plot_96point_curve(future_result, selected_date, title_prefix="未来负荷曲线预测")
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.warning("没有可用的预测日期")
        
        # 数据表格
        st.markdown("#### 日总用电量预测数据")
        st.dataframe(df_filtered, use_container_width=True)
        
        # 下载按钮
        st.markdown("---")
        st.subheader("📥 下载预测结果")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            excel_bytes = to_excel_bytes(df_filtered)
            st.download_button(
                label="📊 下载日总预测",
                data=excel_bytes,
                file_name=f"日总预测_{predict_start}_{predict_end}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            df_curve_wide = future_result["df_curve_pred_wide"].copy()
            df_curve_wide["date"] = pd.to_datetime(df_curve_wide["date"])
            df_curve_filtered = df_curve_wide[
                (df_curve_wide["date"] >= pd.to_datetime(predict_start)) & 
                (df_curve_wide["date"] <= pd.to_datetime(predict_end))
            ]
            excel_bytes2 = to_excel_bytes(df_curve_filtered)
            st.download_button(
                label="📈 下载96点曲线(宽表)",
                data=excel_bytes2,
                file_name=f"96点曲线_宽表_{predict_start}_{predict_end}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col3:
            df_curve_long = future_result["df_curve_pred_long"].copy()
            df_curve_long["date"] = pd.to_datetime(df_curve_long["date"])
            df_curve_long_filtered = df_curve_long[
                (df_curve_long["date"] >= pd.to_datetime(predict_start)) & 
                (df_curve_long["date"] <= pd.to_datetime(predict_end))
            ]
            excel_bytes3 = to_excel_bytes(df_curve_long_filtered)
            st.download_button(
                label="📋 下载96点曲线(长表)",
                data=excel_bytes3,
                file_name=f"96点曲线_长表_{predict_start}_{predict_end}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # 完整结果打包下载
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"日总预测_{predict_start}_{predict_end}.xlsx", excel_bytes)
            zf.writestr("96点曲线_宽表_完整.xlsx", to_excel_bytes(future_result["df_curve_pred_wide"]))
            zf.writestr("96点曲线_长表_完整.xlsx", to_excel_bytes(future_result["df_curve_pred_long"]))
            zf.writestr("比例曲线_宽表.xlsx", to_excel_bytes(future_result["df_prop_pred_wide"]))
            zf.writestr("日总预测_完整.xlsx", to_excel_bytes(df_apr_day_forecast))
            
            # 添加测试集对比数据
            if 'ele_test' in st.session_state:
                test_comparison = pd.DataFrame({
                    "日期": st.session_state.ele_test["ds"].dt.strftime('%Y-%m-%d'),
                    "实际值": st.session_state.ele_test["y"].values,
                    "预测值": st.session_state.forecast_ele.set_index('ds').loc[st.session_state.ele_test["ds"]]["yhat"].values
                })
                test_comparison["误差"] = test_comparison["预测值"] - test_comparison["实际值"]
                test_comparison["误差率(%)"] = (test_comparison["误差"] / test_comparison["实际值"] * 100).round(2)
                zf.writestr("测试集对比结果.xlsx", to_excel_bytes(test_comparison))
        
        st.download_button(
            label="📦 下载所有预测结果 (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="负荷预测结果.zip",
            mime="application/zip",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
