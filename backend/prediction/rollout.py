"""Bounded trajectory hypotheses on an explicit game-scene navigation prior.

Relative support and exposure scores are heuristic, not calibrated probabilities.
The bank is anchored to past observations. A tick never creates new evidence.
"""

from dataclasses import dataclass
import hashlib
import numpy as np

from .geometry import WorldGeometry, sample_path
from .schema import Camera, EngineConfig, FramePacket
from .tracking import Track


@dataclass
class Particle:
    path: np.ndarray
    speed: float
    weight: float


class HypothesisBank:
    def __init__(self, geometry: WorldGeometry, config: EngineConfig, track: Track,
                 support: dict[str, float] | None = None):
        self.geometry, self.config = geometry, config
        self.track_id, self.revision = track.id, track.revision
        self.anchor_t = track.last_seen
        self.last_negative: dict[str, float] = {}
        self.particles: dict[str, list[Particle]] = {}
        priors = np.array([i.prior for i in geometry.prior.intents], dtype=float)
        priors /= priors.sum()
        self.support = support or dict(zip([i.id for i in geometry.prior.intents], priors.tolist()))
        self.invalid: dict[str, int] = {}
        # Stable seed, independent of Python's per-process hash randomization.
        digest = hashlib.sha256(f"{config.random_seed}:{track.id}:{track.revision}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        covariance = track.covariance[:2, :2]
        starts = rng.multivariate_normal(track.mean[:2], covariance, size=config.particles)
        speeds = np.clip(rng.normal(config.speed_mps, config.speed_std_mps, config.particles), .15, 8)
        for intent in geometry.prior.intents:
            particles = []
            for xy, speed in zip(starts, speeds):
                start = np.r_[xy, track.z]
                if intent.kind == "hold":
                    # Validate even HOLD against the walkable surface.
                    from .geometry import point_in_polygon
                    valid = any(abs(s.z - track.z) < .1 and point_in_polygon(xy, s.polygon)
                                for s in geometry.prior.surfaces)
                    inside_wall = any(np.all(start + [0, 0, .8] > w.minimum) and
                                      np.all(start + [0, 0, .8] < w.maximum)
                                      for w in geometry.prior.walls)
                    path = np.array([start]) if valid and not inside_wall else None
                else:
                    path = geometry.route(start, intent.goal_node)
                if path is not None:
                    particles.append(Particle(path, float(speed), 1.))
            self.particles[intent.id] = particles
            self.invalid[intent.id] = config.particles - len(particles)
            if particles:
                for p in particles:
                    p.weight = 1 / len(particles)

    def positions(self, intent_id: str, t: float) -> np.ndarray:
        elapsed = max(0., t - self.anchor_t)
        return np.array([sample_path(p.path, np.array(elapsed * p.speed))
                         for p in self.particles[intent_id]]).reshape(-1, 3)

    def observe(self, track: Track):
        """Reweight the previous futures using a subsequent pixel-derived observation."""
        cov = np.asarray(track.last_observation.covariance_xy) + np.eye(2) * .5 ** 2
        inverse = np.linalg.inv(cov)
        likelihoods = {}
        for identity, particles in self.particles.items():
            if not particles:
                likelihoods[identity] = 1.  # unavailable geometry is not disproof
                continue
            positions = self.positions(identity, track.last_seen)
            error = positions[:, :2] - np.asarray(track.last_observation.xyz)[:2]
            likelihood = np.exp(-.5 * np.einsum("ni,ij,nj->n", error, inverse, error))
            # A floor retains unmodelled behaviour; this is intentionally not a posterior claim.
            likelihoods[identity] = max(.02, float(np.dot(likelihood, [p.weight for p in particles])))
        self._update_support(likelihoods)

    def negative_evidence(self, packet: FramePacket) -> dict:
        result = {"applied": False, "visible_fraction": {}, "reason": "coverage not established"}
        # Association ambiguity makes assigning a miss unsafe: initially use only empty frames.
        if not packet.coverage_complete or packet.detections:
            return result
        if packet.t < self.anchor_t:
            return {**result, "reason": "frame predates hypothesis anchor"}
        if packet.t - self.anchor_t > self.config.horizon_s:
            return {**result, "reason": "beyond validated rollout horizon"}
        sensor = packet.camera.sensor_id
        # Consecutive video frames have correlated detector errors. At most one miss update
        # per second per source, also requiring caller's disclosed detection assumption.
        if packet.t - self.last_negative.get(sensor, -float("inf")) < 1.:
            return {**result, "reason": "correlated-frame cooldown"}
        self.last_negative[sensor] = packet.t
        factors = {}
        for identity, particles in self.particles.items():
            points = self.positions(identity, packet.t)
            if not particles:
                factors[identity] = 1.
                continue
            visible = self.geometry.visible_many(packet.camera, points + [0, 0, .9])
            weights = np.array([p.weight for p in particles])
            result["visible_fraction"][identity] = float(np.dot(weights, visible))
            likelihood = 1 - visible.astype(float) * packet.detection_probability
            factors[identity] = float(np.dot(weights, likelihood))
            weights *= likelihood
            weights /= weights.sum()
            for p, weight in zip(particles, weights):
                p.weight = float(weight)
        self._update_support(factors)
        return {**result, "applied": True, "reason": "assumed detector coverage; uncalibrated likelihood"}

    def _update_support(self, factors: dict[str, float]):
        if not factors or max(factors.values()) - min(factors.values()) < 1e-12:
            return
        values = {key: max(1e-6, value * factors.get(key, 1.)) for key, value in self.support.items()}
        total = sum(values.values())
        # Explicit two-percent mixture with the original prior prevents forced certainty.
        prior_total = sum(i.prior for i in self.geometry.prior.intents)
        self.support = {i.id: .98 * values[i.id] / total + .02 * i.prior / prior_total
                        for i in self.geometry.prior.intents}

    def summarize(self, now: float, player: Camera | None) -> list[dict]:
        # Hypotheses expire rather than inventing a new horizon without an observation.
        elapsed = max(0., now - self.anchor_t)
        remaining = max(0., self.config.horizon_s - elapsed)
        offsets = np.arange(0, remaining + 1e-8, self.config.step_s)
        result = []
        for intent in self.geometry.prior.intents:
            particles = self.particles[intent.id]
            item = {"intent_id": intent.id, "label": intent.label, "kind": intent.kind,
                    "relative_support": float(self.support[intent.id]),
                    "samples": len(particles), "rejected_samples": self.invalid[intent.id],
                    "available": bool(particles) and remaining > 0,
                    "remaining_horizon_s": remaining, "arrival_remaining_s": None,
                    "exposure_fraction": None, "rank_score": None, "path_preview": []}
            if not item["available"]:
                result.append(item)
                continue
            weights = np.array([p.weight for p in particles])
            paths = np.array([sample_path(p.path, (elapsed + offsets) * p.speed) for p in particles])
            # A real particle path, not a mean line that may pass through a wall.
            representative = int(np.argmax(weights))
            stride = max(1, len(offsets) // 20)
            item["path_preview"] = [{"dt": float(offsets[j]), "xyz": paths[representative, j].tolist()}
                                    for j in range(0, len(offsets), stride)]
            if intent.kind == "route":
                arrival = np.array([np.linalg.norm(np.diff(p.path, axis=0), axis=1).sum() / p.speed - elapsed
                                    for p in particles])
                order = np.argsort(arrival)
                quantiles = np.interp([.1, .5, .9], np.cumsum(weights[order]), arrival[order])
                item["arrival_remaining_s"] = {"p10": float(quantiles[0]), "p50": float(quantiles[1]),
                                               "p90": float(quantiles[2]),
                                               "meaning": "conditional on this route and assumed speed; negative means already due"}
            if player is not None:
                # Only a fresh player pose is used. Camera motion is not predicted.
                position = np.asarray(player.camera_to_world)[:3, 3]
                exposure = self.geometry.line_of_sight_many(position, (paths + [0, 0, .9]).reshape(-1, 3)).reshape(paths.shape[:2])
                item["exposure_fraction"] = float(np.dot(weights, exposure.mean(axis=1)))
                item["rank_score"] = self.support[intent.id] * (.5 + .5 * item["exposure_fraction"])
            else:
                item["rank_score"] = self.support[intent.id]
            result.append(item)
        return sorted(result, key=lambda x: (-(x["rank_score"] if x["rank_score"] is not None else -1), x["intent_id"]))


class CueGate:
    """Stabilize the leading hypothesis; report only initial lead or changed lead."""
    def __init__(self, config: EngineConfig):
        self.config = config
        self.leader: str | None = None
        self.candidate: str | None = None
        self.since = 0.

    def update(self, ranking: list[dict], now: float, *, allow_change: bool) -> dict | None:
        if not allow_change:
            return None
        ranked = [item for item in ranking if item["available"]]
        if not ranked:
            self.candidate = None
            return None
        top = ranked[0]
        if self.leader is None:
            self.leader = top["intent_id"]
            return {"kind": "initial_hypothesis", "intent_id": self.leader,
                    "text": f"Leading model hypothesis: {top['label']}."}
        scores = {item["intent_id"]: item["rank_score"] for item in ranked}
        if top["intent_id"] == self.leader or top["rank_score"] - scores.get(self.leader, 0) < self.config.hysteresis_margin:
            self.candidate = None
            return None
        if self.candidate != top["intent_id"]:
            self.candidate, self.since = top["intent_id"], now
        if now - self.since < self.config.hysteresis_dwell_s:
            return None
        old, self.leader, self.candidate = self.leader, top["intent_id"], None
        return {"kind": "leading_hypothesis_changed", "previous_intent_id": old,
                "intent_id": self.leader, "text": f"Leading model hypothesis changed to {top['label']}."}
