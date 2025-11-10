import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class OptimizedAIAnalyzer:
    def __init__(self, model_path: str = "ai_models", use_light_models: bool = True):
        self.model_path = model_path
        self.use_light_models = use_light_models
        self.regime_model = None
        self.signal_validator = None
        self.scaler = None
        self.is_trained = False
        self._load_models()

    def _load_models(self):
        try:
            reg_p = os.path.join(self.model_path, "regime_model.pkl")
            sig_p = os.path.join(self.model_path, "signal_validator.pkl")
            scl_p = os.path.join(self.model_path, "scaler.pkl")
            if os.path.exists(reg_p) and os.path.exists(sig_p) and os.path.exists(scl_p):
                self.regime_model = joblib.load(reg_p)
                self.signal_validator = joblib.load(sig_p)
                self.scaler = joblib.load(scl_p)
                self.is_trained = True
                print("[ai] models loaded from cache")
        except Exception as e:
            print(f"[ai] load error: {e}")
            self.is_trained = False

    def save_models(self):
        if self.is_trained:
            os.makedirs(self.model_path, exist_ok=True)
            joblib.dump(self.regime_model, os.path.join(self.model_path, "regime_model.pkl"))
            joblib.dump(self.signal_validator, os.path.join(self.model_path, "signal_validator.pkl"))
            joblib.dump(self.scaler, os.path.join(self.model_path, "scaler.pkl"))
            print("[ai] models saved")

    def extract_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) < 10:
            return pd.DataFrame(index=df.index if df is not None else None)

        features = pd.DataFrame(index=df.index)
        price = df['close'].astype(float).values
        volume = df['volume'].astype(float).values if 'volume' in df.columns else np.ones(len(df))

        returns = np.diff(np.log(price), prepend=np.log(price[0]))
        vol = pd.Series(returns, index=df.index).rolling(10, min_periods=5).std().fillna(0.01)

        # price efficiency
        eff = vol.copy()*0 + 0.5
        if len(price) > 10:
            r = pd.Series(returns, index=df.index)
            num = r.rolling(10, min_periods=10).sum()
            den = r.abs().rolling(10, min_periods=10).sum() + 1e-9
            eff = (num/den).fillna(0.5)

        # volume anomaly
        vol_ma = pd.Series(volume, index=df.index).rolling(10, min_periods=5).mean().fillna(method="bfill").fillna(0)
        vol_std = pd.Series(volume, index=df.index).rolling(10, min_periods=5).std().fillna(method="bfill").fillna(1e-9)
        vol_anom = (pd.Series(volume, index=df.index) - vol_ma) / (vol_std + 1e-9)

        features['price_efficiency'] = eff.astype('float32')
        features['volume_anomaly'] = vol_anom.astype('float32')
        features['volatility_regime'] = (vol / (vol.mean() + 1e-9)).astype('float32')
        features['liquidity_score'] = (pd.Series(price, index=df.index) * pd.Series(volume, index=df.index) / (vol + 1e-9)).astype('float32')

        return features.fillna(0.0)

    def train_models(self, historical_data: Dict[str, pd.DataFrame]):
        try:
            all_features = []
            all_regime_labels = []
            all_signal_labels = []

            for symbol, df in historical_data.items():
                if df is None or len(df) < 120:
                    continue
                feats = self.extract_advanced_features(df)
                if feats.empty:
                    continue

                volatility = df['close'].pct_change().rolling(20).std().fillna(0.01)
                regime_labels = pd.cut(volatility, bins=[0, 0.005, 0.015, 1], labels=[0, 1, 2]).fillna(1)

                future_returns = df['close'].pct_change(6).shift(-6).fillna(0)
                signal_labels = ((future_returns > 0.002) | (future_returns < -0.002)).astype(int)

                common_idx = feats.index.intersection(regime_labels.index).intersection(signal_labels.index)
                if len(common_idx) < 60:
                    continue

                all_features.append(feats.loc[common_idx])
                all_regime_labels.append(regime_labels.loc[common_idx])
                all_signal_labels.append(signal_labels.loc[common_idx])

            if not all_features:
                print("[ai] no sufficient data for training")
                return

            X = pd.concat(all_features, axis=0)
            y_regime = pd.concat(all_regime_labels, axis=0).astype(int)
            y_signal = pd.concat(all_signal_labels, axis=0).astype(int)

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            if self.use_light_models:
                self.regime_model = LogisticRegression(C=0.1, solver='liblinear', random_state=42, max_iter=1000)
                self.signal_validator = LogisticRegression(C=0.1, solver='liblinear', random_state=42, max_iter=1000)
            else:
                self.regime_model = RandomForestClassifier(n_estimators=80, max_depth=12, random_state=42, n_jobs=-1)
                self.signal_validator = RandomForestClassifier(n_estimators=60, max_depth=10, random_state=42, n_jobs=-1)

            self.regime_model.fit(X_scaled, y_regime)
            self.signal_validator.fit(X_scaled, y_signal)
            self.is_trained = True

            print(f"[ai] models trained on {len(X)} samples")
            self.save_models()

        except Exception as e:
            print(f"[ai] training error: {e}")
            self.is_trained = False

    def train_if_needed(self, historical_data: Dict[str, pd.DataFrame]):
        if not self.is_trained:
            self.train_models(historical_data)

    def predict_regime_series(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Batch regime prediction for the whole series (fast). Returns (class, confidence)."""
        if not self.is_trained or df is None or df.empty:
            idx = df.index if df is not None else pd.Index([])
            return pd.Series(1, index=idx, dtype='int8'), pd.Series(np.nan, index=idx)
        try:
            feats = self.extract_advanced_features(df)
            if feats.empty:
                return pd.Series(1, index=df.index, dtype='int8'), pd.Series(np.nan, index=df.index)
            X_scaled = self.scaler.transform(feats.values)
            if hasattr(self.regime_model, "predict_proba"):
                proba = self.regime_model.predict_proba(X_scaled)
                cls = proba.argmax(axis=1).astype('int8')
                conf = proba.max(axis=1).astype('float32')
                return pd.Series(cls, index=feats.index), pd.Series(conf, index=feats.index)
            else:
                pred = self.regime_model.predict(X_scaled)
                return pd.Series(pred.astype('int8'), index=feats.index), pd.Series(np.nan, index=feats.index)
        except Exception as e:
            print(f"[ai] predict_regime_series error: {e}")
            return pd.Series(1, index=df.index, dtype='int8'), pd.Series(np.nan, index=df.index)

    def predict_regime_with_conf(self, df: pd.DataFrame) -> Tuple[int, float]:
        ser, conf = self.predict_regime_series(df.tail(1))
        if len(ser) == 0:
            return 1, 0.5
        return int(ser.iloc[-1]), float(conf.iloc[-1]) if conf.notna().any() else 0.5

    def predict_regime(self, df: pd.DataFrame) -> int:
        r, _ = self.predict_regime_with_conf(df)
        return r

    def validate_signal(self, df: pd.DataFrame, signal_direction: float) -> Tuple[bool, float]:
        if not self.is_trained:
            return True, 0.5
        try:
            feats = self.extract_advanced_features(df)
            if feats.empty:
                return True, 0.5
            X_scaled = self.scaler.transform(feats.iloc[[-1]])
            if hasattr(self.signal_validator, "predict_proba"):
                success_prob = float(self.signal_validator.predict_proba(X_scaled)[0][1])
            else:
                pred = int(self.signal_validator.predict(X_scaled)[0])
                success_prob = 0.6 if pred == 1 else 0.4
            return success_prob > 0.55, success_prob
        except Exception:
            return True, 0.5

ai_analyzer = OptimizedAIAnalyzer()
