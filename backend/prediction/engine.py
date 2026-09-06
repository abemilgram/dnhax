"""Causal, source-clock session joining pixel projection, belief and simulation."""

from collections import OrderedDict, deque
import math
import hashlib
import time
import numpy as np

from .geometry import WorldGeometry
from .rollout import CueGate, HypothesisBank
from .schema import EngineConfig, FramePacket, ScenePrior
from .tracking import Tracker


class PredictionEngine:
    def __init__(self, prior: ScenePrior, *, clock_id: str, evidence_source: str,
                 config: EngineConfig | None = None):
        if not clock_id or evidence_source not in {"video_detector", "synthetic_fixture"}:
            raise ValueError("A clock and explicit evidence mode are required")
        if prior.synthetic != (evidence_source == "synthetic_fixture"):
            raise ValueError("Synthetic geometry and video evidence cannot be mixed")
        self.config = config or EngineConfig()
        budget = self.config.particles * len(np.arange(0, self.config.horizon_s + self.config.step_s, self.config.step_s))
        if budget * self.config.max_tracks * len(prior.intents) > 500_000:
            raise ValueError("Combined track, intent and particle budget exceeds 500000 trajectory samples")
        self.geometry = WorldGeometry(prior)
        if self.config.prediction_model == "branching":
            from .branching import BranchingBank
            self.bank_type = BranchingBank
        else:
            self.bank_type = HypothesisBank
        self.clock_id, self.evidence_source = clock_id, evidence_source
        self.tracker = Tracker(self.config, transition_validator=self._transition_allowed)
        self.banks: dict[str, HypothesisBank] = {}
        self.gates: dict[str, CueGate] = {}
        self.cameras: dict[str, tuple[float, object]] = {}
        self.seen: OrderedDict[tuple[str, str], str] = OrderedDict()
        self.events = deque(maxlen=256)
        self.event_sequence = 0
        self.now = 0.
        self.capture_watermark = 0.
        self.last_ingest_ms = 0.
        self.accepted_frames = 0
        self.last_evidence: dict = {}
        self._cached = None

    def _transition_allowed(self, track, xyz, surface_id, t):
        elapsed = t - track.last_seen
        if elapsed <= 0:
            return False  # simultaneous views cannot move one person between floors
        distance = self.geometry.route_distance(track.last_observation.xyz, xyz)
        if distance is None:
            return False
        spread = 3 * float(np.sqrt(np.trace(track.last_observation.covariance_xy)))
        travel_budget = min(8., self.config.speed_mps + 3 * self.config.speed_std_mps) * elapsed + spread
        return distance <= travel_budget

    def _validate_now(self, now: float):
        if not math.isfinite(now) or now < self.now:
            raise ValueError("Processing time must be finite and monotonic in the session clock")

    def ingest(self, packet: FramePacket, *, now: float) -> dict:
        started = time.perf_counter()
        self._validate_now(now)
        if packet.clock_id != self.clock_id:
            raise ValueError("Clock mismatch; synchronize sensors before ingesting")
        if packet.camera.coordinate_frame != self.geometry.prior.coordinate_frame:
            raise ValueError("Coordinate frame mismatch; align and calibrate the scene first")
        if packet.evidence_source != self.evidence_source:
            raise ValueError("Evidence mode cannot change within a session")
        identity = (packet.camera.sensor_id, packet.frame_id)
        fingerprint = hashlib.sha256(packet.model_dump_json().encode()).hexdigest()
        if identity in self.seen:
            if self.seen[identity] != fingerprint:
                raise ValueError("Frame ID was reused with different evidence")
            # Idempotence is deliberately before the late-frame check for network retries.
            return {"duplicate": True, "state": self.state()}
        if packet.available_t > now or packet.t > now:
            raise ValueError("Future evidence cannot affect the current belief")
        if packet.t < self.capture_watermark:
            raise ValueError("Out-of-order capture; merge sensor streams by capture time before ingestion")
        if now - packet.t > self.config.max_frame_lag_s:
            raise ValueError("Frame too old for this session's latency budget")
        if packet.clock_uncertainty_s > self.config.max_clock_uncertainty_s:
            raise ValueError("Clock uncertainty exceeds the session budget")
        if packet.camera.sensor_id not in self.cameras and len(self.cameras) >= 8:
            raise ValueError("A session supports at most eight camera sources")
        if len(self.seen) >= 20_000:
            raise ValueError("Session frame limit reached; start a new session for the next take")
        observations, rejected = self.geometry.project(packet)
        accepted = [o for o in observations if o.confidence >= self.config.min_detection_confidence]
        # Timing uncertainty grows measurement covariance instead of pretending perfect sync.
        extra_variance = (packet.clock_uncertainty_s * self.config.speed_mps) ** 2
        accepted = [o.model_copy(update={"covariance_xy":
                    (np.asarray(o.covariance_xy) + np.eye(2) * extra_variance).tolist()}) for o in accepted]
        associations = self.tracker.ingest(accepted, packet.t, packet.camera.sensor_id)
        for track_id in associations["matched"] + associations["created"]:
            track = self.tracker.tracks[track_id]
            support = None
            cooldown = {}
            update = None
            if track_id in self.banks:
                old = self.banks[track_id]
                update = old.observe(track)
                support = old.support.copy()
                cooldown = old.last_negative.copy()
            self.banks[track_id] = self.bank_type(self.geometry, self.config, track, support)
            self.banks[track_id].last_negative = cooldown
            if self.config.prediction_model == "branching" and update is not None:
                self.banks[track_id].last_update = update
            self.gates.setdefault(track_id, CueGate(self.config))
        negative = {}
        for identity_key, bank in self.banks.items():
            if identity_key in self.tracker.tracks:
                negative[identity_key] = bank.negative_evidence(packet)
        self.cameras[packet.camera.sensor_id] = (packet.t, packet.camera)
        self.capture_watermark = packet.t
        self.now = now
        self.accepted_frames += 1
        self.last_evidence = {"sensor_id": packet.camera.sensor_id, "frame_id": packet.frame_id,
                              "captured_t": packet.t, "available_t": packet.available_t,
                              "processing_t": now, "capture_to_processing_s": now - packet.t,
                              "projected": len(accepted), "rejected_projections": rejected,
                              "associations": associations, "negative_evidence": negative}
        self.seen[identity] = fingerprint
        for track_id in set(self.banks) - self.tracker.tracks.keys():
            del self.banks[track_id]
            self.gates.pop(track_id, None)
        self._cached = None
        state = self._snapshot(allow_change=True)
        self.last_ingest_ms = (time.perf_counter() - started) * 1000
        state["metrics"]["last_ingest_ms"] = self.last_ingest_ms
        self._cached = state
        return {"duplicate": False, "state": state}

    def tick(self, now: float) -> dict:
        self._validate_now(now)
        if now == self.now and self._cached is not None:
            return self._cached
        self.now = now
        # Ticks do not advance the capture-time filter, permitting ordinary transport delay.
        # Nor do they make absence claims. Only newly processed frames may change a cue.
        self._cached = self._snapshot(allow_change=False)
        return self._cached

    def state(self) -> dict:
        if self._cached is None:
            self._cached = self._snapshot(allow_change=False)
        return self._cached

    def _snapshot(self, *, allow_change: bool) -> dict:
        player_record = self.cameras.get(self.config.player_sensor_id)
        player = player_record[1] if player_record and self.now - player_record[0] <= self.config.stale_after_s else None
        tracks = []
        for identity, track in self.tracker.tracks.items():
            age = self.now - track.last_seen
            if age > self.config.expire_after_s:
                continue
            item = track.snapshot()
            # Display covariance propagated to presentation time; filter remains at capture time.
            dt = max(0., self.now - track.t)
            transition = np.eye(4)
            transition[0, 2] = transition[1, 3] = dt
            g = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
            cov = transition @ track.covariance @ transition.T + g @ g.T * self.config.acceleration_std_mps2 ** 2
            mean = transition @ track.mean
            item.update({"xyz": [float(mean[0]), float(mean[1]), track.z],
                         "covariance_xy": cov[:2, :2].tolist(), "age_s": age,
                         "status": "conflicting" if track.status == "conflicting" else
                                   ("stale" if age > self.config.stale_after_s else "observed"),
                         "last_observed_xyz": list(track.last_observation.xyz),
                         "estimate_model": "unconstrained constant-velocity Gaussian; paths below obey the navigation prior"})
            bank = self.banks.get(identity)
            if bank and self.config.prediction_model == "branching":
                ranking, item["map_belief"] = bank.summarize_all(self.now, player)
                # Keep legacy xyz useful as a ghost marker, never a through-wall mean.
                # The actual spatial belief is the multi-location distribution above.
                item["association_gaussian"] = {"xyz": item["xyz"],
                                                "covariance_xy": item["covariance_xy"],
                                                "velocity_xy": item["velocity_xy"],
                                                "purpose": "Association only; unconstrained by navigation geometry"}
                item.update(xyz=list(track.last_observation.xyz),
                            covariance_xy=track.last_observation.covariance_xy,
                            velocity_xy=None, xyz_semantics="last_observed_marker",
                            estimate_model="Weighted branching graph trajectories in map_belief; xyz marks last observation")
            else:
                ranking = bank.summarize(self.now, player) if bank else []
            item["hypotheses"] = ranking
            gate = self.gates.get(identity)
            event = gate.update(ranking, self.now, allow_change=allow_change and item["status"] != "conflicting") if gate else None
            if event:
                self.event_sequence += 1
                self.events.append({**event, "id": self.event_sequence, "t": self.now,
                                    "track_id": identity, "basis": "uncalibrated model ranking"})
            item["leading_hypothesis"] = gate.leader if gate and any(h["available"] for h in ranking) else None
            tracks.append(item)
        return {"t": self.now, "clock_id": self.clock_id,
                "prediction_model": self.config.prediction_model,
                "coordinate_frame": self.geometry.prior.coordinate_frame, "units": "meters",
                "evidence_source": self.evidence_source, "synthetic": self.geometry.prior.synthetic,
                "prior": {"id": self.geometry.prior.id, "provenance": self.geometry.prior.provenance,
                          "scale_source": self.geometry.prior.scale_source},
                "tracks": tracks, "events": list(self.events), "event_cursor": self.event_sequence,
                "last_evidence": self.last_evidence,
                "metrics": {"accepted_frames": self.accepted_frames, "last_ingest_ms": self.last_ingest_ms},
                "interpretation": {"support": "Relative model support, not calibrated exit probability",
                                   "arrival": "Conditional windows from geometry and assumed behavior; branching model retains no-arrival mass",
                                   "risk": "Heuristic support weighted by geometric exposure to a fresh player pose",
                                   "ground_truth": "No enemy game-state or scripted-path input"}}
