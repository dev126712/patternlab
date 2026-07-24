"""Central configuration. Every detection threshold lives here — never inline in detectors.

All price-distance thresholds are expressed as multiples of ATR so the engine is
instrument-agnostic (works identically on XAUUSD@4000 and EURUSD@1.08).
"""
from __future__ import annotations

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class DetectionParams(BaseModel):
    # --- ATR / candles ---
    atr_period: int = 14
    momentum_body_ratio: float = 0.60      # body / range to call a candle "momentum"
    momentum_body_atr: float = 1.20        # body must also exceed this * ATR

    # --- swings (zigzag) ---
    swing_reversal_atr: float = 2.0        # reversal threshold to confirm a pivot

    # --- S/R level clustering ---
    level_cluster_atr: float = 0.8         # max intra-cluster distance (in ATR)
    level_min_touches: int = 2
    level_touch_tol_atr: float = 0.5       # "touch" = pivot within this of zone
    level_recency_half_life_bars: int = 2000

    # --- regimes ---
    regime_window: int = 30                # rolling window (bars) for ER + range
    er_consolidation_max: float = 0.25     # Kaufman ER below -> consolidation candidate
    er_impulse_min: float = 0.55           # ER above -> impulse candidate
    consol_range_atr_max: float = 3.0      # rolling range below k*ATR -> basing
    impulse_net_move_atr_min: float = 3.0  # net move over window to call impulse
    min_regime_bars: int = 15

    # --- angle of approach ---
    approach_lookback: int = 30            # bars of OLS slope into the level
    steep_slope_atr: float = 0.30          # ATR-per-bar; >= is "steep/vertical"
    grinding_slope_atr: float = 0.12       # <= is "grinding"

    # --- time at resistance ---
    at_level_tol_atr: float = 0.6          # high within +/- tol*ATR counts as "at" level
    rejection_min_gap_bars: int = 5        # distinct rejection events separated by this

    # --- stair-stepping ---
    stairstep_max_pullback: float = 0.50   # pullback must retrace < 50% of the leg
    stairstep_hold_tol_atr: float = 0.5    # pullback low must stay >= level - tol

    # --- context ---
    context_lookback_bars: int = 1440      # 1 trading day of M1
    fresh_base_dd_max: float = 0.25        # rally origin drawdown fraction of lookback range
    recovery_leg_dd_min: float = 0.50

    # --- matching ---
    dtw_downsample_to: int = 200           # resample closes before DTW (O(n^2))
    weight_fingerprint: float = 0.6
    weight_dtw: float = 0.4


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deployment_env: str = "dev"
    llm_provider: str = ""                 # empty -> derived from deployment_env
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"
    groq_model: str = "llama-3.3-70b-versatile"
    anthropic_model: str = "claude-sonnet-4-6"
    oanda_api_token: str = ""
    oanda_account_id: str = ""
    tv_username: str = ""                  # TradingView login for tvDatafeed (optional)
    tv_password: str = ""
    database_url: str = ""

    @property
    def effective_llm_provider(self) -> str:
        if self.llm_provider:
            return self.llm_provider
        return "ollama" if self.deployment_env == "prod" else "groq"


settings = Settings()
params = DetectionParams()
