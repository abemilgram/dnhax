"""Finite, weighted graph trajectories conditioned on pixel observations.

This is an explicit behavior model, not trained dynamics or calibrated probability.
Particles retain alternate futures through occlusion; a new observation reanchors
the proposal ensemble while carrying intent support. Display ticks never resample.
"""

import hashlib
from types import SimpleNamespace
import numpy as np

from .branch_motion import generate_trajectories
from .branch_summary import summarize_branching
from .rollout import HypothesisBank


def _proposal_measurement(track):
    """Use the measurement-level same-time CI result, never the coasting mean."""
    obs = track.last_observation
    keys = sorted(getattr(track, "_fusion_keys", ()) or [(obs.sensor_id, obs.frame_id)])
    if (getattr(track, "_fusion_t", None) == track.last_seen
            and getattr(track, "_fusion_mean", None) is not None
            and getattr(track, "_fusion_covariance", None) is not None):
        xy = track._fusion_mean
        obs = obs.model_copy(update={"xyz": (float(xy[0]), float(xy[1]), float(obs.xyz[2])),
                                     "covariance_xy": track._fusion_covariance.tolist()})
    return obs, keys


class BranchingBank:
    def __init__(self, geometry, config, track, support=None):
        self.geometry, self.config = geometry, config
        self.track_id, self.revision = track.id, track.revision
        self.anchor_t = track.last_seen
        self.last_negative = {}
        self.prior_support = {i.id: i.prior / sum(j.prior for j in geometry.prior.intents)
                              for i in geometry.prior.intents}
        self.support = dict(support if support is not None else self.prior_support)
        self.trajectories, self.weights, self.invalid = {}, {}, {}
        obs, source_keys = _proposal_measurement(track)
        proposal_track = SimpleNamespace(last_observation=obs)
        for intent in geometry.prior.intents:
            # A source observation, not an unrelated empty frame or presentation tick,
            # identifies this proposal. Python's randomized hash is never used.
            seed = f"{config.random_seed}:{track.id}:{source_keys!r}:{obs.t}:{intent.id}"
            digest = hashlib.sha256(seed.encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            trajectories = generate_trajectories(geometry, config, proposal_track, intent, rng)
            self.trajectories[intent.id] = trajectories
            count = len(trajectories)
            self.weights[intent.id] = np.full(count, 1 / count) if count else np.empty(0)
            self.invalid[intent.id] = config.particles - count
        self.last_update = {"kind": "initial_observation", "t": self.anchor_t,
                            "sensor_id": track.last_observation.sensor_id,
                            "frame_id": track.last_observation.frame_id,
                            "reason": "Position proposals drawn from projected measurement uncertainty; same-time views use covariance intersection"}

    def positions(self, intent_id, t):
        elapsed = np.clip(t - self.anchor_t, 0, self.config.horizon_s)
        return np.asarray([p.positions(elapsed) for p in self.trajectories[intent_id]]).reshape(-1, 3)

    def observe(self, track):
        """Score old futures with a new measurement before proposing fresh futures."""
        obs, _ = _proposal_measurement(track)
        diagnostic = {"kind": "positive_observation", "t": obs.t, "sensor_id": obs.sensor_id,
                      "frame_id": obs.frame_id, "applied": False,
                      "support_before": self.support.copy()}
        elapsed = track.last_seen - self.anchor_t
        if elapsed <= 1e-8:
            diagnostic["reason"] = "Same-time views refine position, not independent motion evidence"
        elif elapsed >= self.config.horizon_s - 1e-7:
            # A censored ensemble cannot score a later observation at its frozen endpoints.
            self.support = self.prior_support.copy()
            diagnostic["reason"] = "Old forecast expired; reset behavior support to disclosed priors"
        else:
            cov = np.asarray(obs.covariance_xy) + np.eye(2) * .5 ** 2
            inverse = np.linalg.inv(cov)
            factors = {}
            for identity, trajectories in self.trajectories.items():
                if not trajectories:
                    factors[identity] = 1.  # missing geometry is not disproof
                    continue
                points = self.positions(identity, track.last_seen)
                error = points[:, :2] - np.asarray(obs.xyz)[:2]
                likelihood = np.exp(-.5 * np.einsum("ni,ij,nj->n", error, inverse, error))
                # Distinct floors cannot corroborate the same observation by sharing XY.
                likelihood *= np.exp(-.5 * ((points[:, 2] - obs.xyz[2]) / .35) ** 2)
                factors[identity] = max(.02, float(self.weights[identity] @ likelihood))
            self._update_support(factors)
            diagnostic.update(applied=True, likelihood_by_intent=factors,
                              reason="New pixel observation scores old graph futures; fresh proposals reanchor position")
        diagnostic["support_after"] = self.support.copy()
        self.last_update = diagnostic
        return diagnostic

    def negative_evidence(self, packet):
        result = {"applied": False, "visible_fraction": {}, "reason": "coverage not established"}
        if not packet.coverage_complete or packet.detections:
            return result
        if packet.t < self.anchor_t:
            return {**result, "reason": "frame predates hypothesis anchor"}
        if packet.t - self.anchor_t >= self.config.horizon_s - 1e-7:
            return {**result, "reason": "beyond modeled rollout horizon"}
        sensor = packet.camera.sensor_id
        if packet.t - self.last_negative.get(sensor, -float("inf")) < 1.:
            return {**result, "reason": "correlated-frame cooldown"}
        factors, pending = {}, {}
        for identity, trajectories in self.trajectories.items():
            if not trajectories:
                factors[identity] = 1.
                continue
            points = self.positions(identity, packet.t)
            visible = self.geometry.visible_many(packet.camera, points + [0, 0, .9])
            weights = self.weights[identity]
            result["visible_fraction"][identity] = float(weights @ visible)
            likelihood = 1 - visible.astype(float) * packet.detection_probability
            factors[identity] = float(weights @ likelihood)
            updated = weights * likelihood
            pending[identity] = updated / updated.sum()
        if not any(value > 0 for value in result["visible_fraction"].values()):
            return {**result, "reason": "No modeled locations in the verified camera view"}
        before = self.support.copy()
        self.weights.update(pending)
        self._update_support(factors)
        self.last_negative[sensor] = packet.t
        self.last_update = {"kind": "visible_miss", "t": packet.t, "sensor_id": sensor,
                            "frame_id": packet.frame_id, "support_before": before,
                            "support_after": self.support.copy(),
                            "visible_fraction": result["visible_fraction"],
                            "detection_probability_assumed": packet.detection_probability,
                            "coverage_assumption": packet.coverage_assumption}
        return {**result, "applied": True, "reason": "Assumed detector coverage; uncalibrated likelihood"}

    # Retain the baseline's disclosed two-percent prior mixture and support floor.
    # Both models must keep unsupported behavior from becoming false certainty.
    _update_support = HypothesisBank._update_support

    def summarize_all(self, now, player):
        ranking, belief = summarize_branching(
            self.geometry, self.config, self.trajectories, self.weights, self.support,
            self.invalid, self.anchor_t, now, player)
        belief["last_evidence_update"] = self.last_update
        belief["model"] = "branching_graph_trajectories_v1"
        belief["assumptions"] = {
            "learned_dynamics": False, "calibrated_probabilities": False,
            "position_refresh": "Reanchor to each projected observation; retain intent support, redraw trajectories",
            "association": "Anonymous Gaussian gating; long-occlusion identity remains uncertain",
            "simultaneous_views": "Position uses measurement covariance intersection; first processed view supplies motion support; sort equal-time frames by sensor_id then frame_id",
            "exploration_probability": self.config.branch_exploration_probability,
            "stop_rate_per_s": self.config.branch_stop_rate_per_s,
            "reverse_rate_per_s": self.config.branch_reverse_rate_per_s,
            "speed_mps": self.config.speed_mps, "speed_std_mps": self.config.speed_std_mps,
            "slow_mode_probability": .2, "slow_speed_multiplier": .5,
            "pause_duration_uniform_s": [.5, 2.],
            "graph_choices": "Goal-biased graph walk with at most one exploratory detour per node; reversals retrace it",
            "goal_boundary": "First visit to any modeled exit is absorbing; movement beyond exits is not modeled",
            "support_regularization": "Two-percent original-prior mixture after unequal intent likelihoods",
        }
        return ranking, belief

    def summarize(self, now, player):
        return self.summarize_all(now, player)[0]
