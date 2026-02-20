import numpy as np
from collections import deque


FEATURE_COLUMNS = [
    'hurst_short', 'hurst_med', 'hurst_long', 'hurst_divergence',
    'tail_index', 'mf_width',
    'realized_vol', 'vol_of_vol', 'iv_rv_ratio',
    'rsi', 'bb_pctb', 'atr', 'volume_ratio', 'vwap_dev',
    'above_sma50', 'sma_trend',
    'macd_line', 'macd_signal', 'macd_hist',
    'ret_1d', 'ret_5d', 'ret_20d', 'drawdown_20d',
    'regime', 'danger_score', 'strength', 'direction',
]

NUM_FEATURES = len(FEATURE_COLUMNS)


def extract_ml_features(signal):
    try:
        vec = np.array([float(signal.get(col, 0.0)) for col in FEATURE_COLUMNS], dtype=np.float64)
        vec = np.where(np.isfinite(vec), vec, 0.0)
        return vec
    except (TypeError, ValueError):
        return None

class TradeDataCollector:
    def __init__(self, max_samples=2000):
        self.pending = {}
        self.X = deque(maxlen=max_samples)
        self.y = deque(maxlen=max_samples)
        self.meta = deque(maxlen=max_samples)

    def record_entry(self, ticker, signal, date=None):
        vec = extract_ml_features(signal)
        if vec is not None:
            self.pending[ticker] = {
                'features': vec,
                'date': date,
                'reason': signal.get('reason', 'unknown'),
            }

    def record_exit(self, ticker, pnl_pct):
        if ticker not in self.pending:
            return
        entry = self.pending.pop(ticker)
        if pnl_pct is None:
            return

        label = 1 if pnl_pct > 0.0 else 0
        self.X.append(entry['features'])
        self.y.append(label)
        self.meta.append({
            'ticker': ticker,
            'date': entry['date'],
            'reason': entry['reason'],
            'pnl_pct': pnl_pct,
        })

    def get_training_data(self):
        if len(self.X) < 30:
            return None, None
        return np.array(self.X), np.array(self.y)

    @property
    def n_samples(self):
        return len(self.X)

class MLTradeFilter:
    def __init__(self, retrain_interval=30, min_samples=30, threshold=0.45):
        self.model = None
        self.is_trained = False
        self.retrain_interval = retrain_interval
        self.min_samples = min_samples
        self.threshold = threshold
        self.trades_since_train = 0
        self.train_count = 0
        self.last_accuracy = 0.0
        self._xgb_available = None

    def _check_xgb(self):
        if self._xgb_available is None:
            try:
                import xgboost
                self._xgb_available = True
            except ImportError:
                self._xgb_available = False
        return self._xgb_available

    def train(self, X, y):
        if not self._check_xgb():
            return False
        if X is None or y is None or len(X) < self.min_samples:
            return False

        import xgboost as xgb
        from sklearn.model_selection import cross_val_score

        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        scale_pos_weight = n_neg / max(n_pos, 1)

        params = {
            'n_estimators': min(100, max(30, len(X) // 3)),
            'max_depth': 3,
            'learning_rate': 0.08,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 1.0,
            'reg_lambda': 2.0,
            'scale_pos_weight': scale_pos_weight,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'verbosity': 0,
            'random_state': 42,
        }

        model = xgb.XGBClassifier(**params)

        if len(X) >= 50:
            try:
                cv_scores = cross_val_score(model, X, y, cv=min(5, len(X) // 10), scoring='accuracy')
                self.last_accuracy = float(np.mean(cv_scores))
            except Exception:
                self.last_accuracy = 0.0

        model.fit(X, y)
        self.model = model
        self.is_trained = True
        self.trades_since_train = 0
        self.train_count += 1
        return True

    def predict_win_probability(self, signal):
        if not self.is_trained or self.model is None:
            return 0.5

        vec = extract_ml_features(signal)
        if vec is None:
            return 0.5

        try:
            proba = self.model.predict_proba(vec.reshape(1, -1))[0]
            return float(proba[1])
        except Exception:
            return 0.5

    def should_take_trade(self, signal):
        if not self.is_trained:
            return True

        p_win = self.predict_win_probability(signal)
        return p_win >= self.threshold

    def maybe_retrain(self, collector):
        self.trades_since_train += 1
        if self.trades_since_train >= self.retrain_interval:
            X, y = collector.get_training_data()
            if X is not None:
                return self.train(X, y)
        return False

    def get_feature_importance(self, top_n=10):
        if not self.is_trained or self.model is None:
            return {}
        importances = self.model.feature_importances_
        pairs = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True)
        return {name: float(imp) for name, imp in pairs[:top_n]}

    def get_status(self):
        return {
            'is_trained': self.is_trained,
            'train_count': self.train_count,
            'threshold': self.threshold,
            'last_cv_accuracy': self.last_accuracy,
            'trades_since_train': self.trades_since_train,
        }