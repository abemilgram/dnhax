"""Deterministic hysteresis and semantic change-only cue reduction."""

from dataclasses import dataclass
import math

from .schema import (
    Cue,
    IntentKind,
    PlanRanking,
    RiskBand,
    TrajectoryCandidate,
)


@dataclass(frozen=True, slots=True)
class CueConfig:
    switch_margin: float = 0.15
    consecutive_cycles: int = 2
    minimum_dwell: float = 1.5
    emergency_risk_threshold: float | None = 0.75
    emergency_improvement: float = 0.20
    low_risk_threshold: float = 0.25
    high_risk_threshold: float = 0.55

    def __post_init__(self) -> None:
        for name in (
            "switch_margin",
            "minimum_dwell",
            "emergency_improvement",
            "low_risk_threshold",
            "high_risk_threshold",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.low_risk_threshold >= self.high_risk_threshold:
            raise ValueError("risk thresholds must be strictly increasing")
        if self.high_risk_threshold > 1.0:
            raise ValueError("high_risk_threshold must not exceed 1")
        if self.emergency_risk_threshold is not None:
            threshold = float(self.emergency_risk_threshold)
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError("emergency_risk_threshold must be in [0, 1]")
            object.__setattr__(self, "emergency_risk_threshold", threshold)
        if (
            isinstance(self.consecutive_cycles, bool)
            or not isinstance(self.consecutive_cycles, int)
            or self.consecutive_cycles < 1
        ):
            raise ValueError("consecutive_cycles must be a positive integer")


class CueReducer:
    """Apply intent hysteresis and emit only navigation-state changes."""

    def __init__(self, config: CueConfig | None = None) -> None:
        self.config = config or CueConfig()
        self._selected: IntentKind | None = None
        self._selected_since: float | None = None
        self._challenger: IntentKind | None = None
        self._challenger_cycles = 0
        self._semantic_state: tuple[object, ...] | None = None
        self._sequence = 0
        self._last_t: float | None = None

    @property
    def selected_intent(self) -> IntentKind | None:
        return self._selected

    def update(
        self,
        ranking: PlanRanking,
        *,
        tracking_degraded: bool = False,
    ) -> Cue | None:
        if self._last_t is not None and ranking.t < self._last_t:
            raise ValueError("cue time must be monotonically non-decreasing")
        if not isinstance(tracking_degraded, bool):
            raise ValueError("tracking_degraded must be a bool")
        candidates = {item.intent: item for item in ranking.candidates}
        top = ranking.candidates[0]
        changed_intent = False
        initial = self._selected is None
        if initial:
            self._selected = top.intent
            self._selected_since = ranking.t
        else:
            current = candidates.get(self._selected)
            if current is None:
                self._selected = top.intent
                self._selected_since = ranking.t
                changed_intent = True
            elif top.intent == self._selected:
                self._clear_challenger()
            elif self._should_emergency_switch(current, top):
                self._selected = top.intent
                self._selected_since = ranking.t
                self._clear_challenger()
                changed_intent = True
            elif top.valid and top.utility >= current.utility + self.config.switch_margin:
                if self._challenger == top.intent:
                    self._challenger_cycles += 1
                else:
                    self._challenger = top.intent
                    self._challenger_cycles = 1
                dwell = ranking.t - float(self._selected_since)
                if (
                    self._challenger_cycles >= self.config.consecutive_cycles
                    and dwell >= self.config.minimum_dwell
                ):
                    self._selected = top.intent
                    self._selected_since = ranking.t
                    self._clear_challenger()
                    changed_intent = True
            else:
                self._clear_challenger()

        selected = candidates.get(self._selected, top)
        band = self._risk_band(selected.score.risk_score)
        semantic = (
            selected.intent,
            band,
            selected.valid,
            tracking_degraded,
        )
        previous = self._semantic_state
        if previous == semantic:
            self._last_t = ranking.t
            return None

        reasons: list[str] = []
        if initial:
            reasons.append("initial_selection")
            if not selected.valid:
                reasons.append("route_invalid")
            if tracking_degraded:
                reasons.append("tracking_degraded")
        elif changed_intent or (previous is not None and previous[0] != selected.intent):
            reasons.append("top_intent_changed")
        if previous is not None and previous[1] != band:
            reasons.append("risk_band_changed")
        if previous is not None and previous[2] != selected.valid:
            reasons.append("route_restored" if selected.valid else "route_invalid")
        if previous is not None and previous[3] != tracking_degraded:
            reasons.append(
                "tracking_degraded" if tracking_degraded else "tracking_recovered"
            )
        self._semantic_state = semantic
        self._last_t = ranking.t
        self._sequence += 1
        return Cue(
            sequence=self._sequence,
            t=ranking.t,
            intent=selected.intent,
            risk_band=band,
            risk=round(selected.score.risk_score, 2),
            route_valid=selected.valid,
            tracking_degraded=tracking_degraded,
            reasons=tuple(reasons),
            route=selected.route,
        )

    reduce = update

    def _should_emergency_switch(
        self,
        current: TrajectoryCandidate,
        challenger: TrajectoryCandidate,
    ) -> bool:
        threshold = self.config.emergency_risk_threshold
        return bool(
            threshold is not None
            and challenger.valid
            and current.score.risk_score >= threshold
            and challenger.score.risk_score
            <= current.score.risk_score - self.config.emergency_improvement
        )

    def _risk_band(self, risk: float) -> RiskBand:
        if risk < self.config.low_risk_threshold:
            return RiskBand.LOW
        if risk < self.config.high_risk_threshold:
            return RiskBand.MEDIUM
        return RiskBand.HIGH

    def _clear_challenger(self) -> None:
        self._challenger = None
        self._challenger_cycles = 0
