export type Cloud = {
  source: string;
  count: number;
  points: string;
  colors: string;
  ply: string;
  transform: number[][];
  capture_id?: string;
  cameras?: {
    frame: string;
    t: number;
    intrinsics?: number[][];
    world_to_camera?: number[][];
  }[];
};
export type Diagnostics = {
  status: string;
  method: string;
  matches: number;
  inliers: number;
  fit_count: number;
  inlier_ratio: number;
  median_error: number;
  p90_error: number;
  heldout_error: number;
  heldout_count: number;
  relative_scale: number;
  units: string;
  threshold: number;
  spatial_extent: number[];
};
export type CameraLocationSample = {
  frame_id: number;
  source: string;
  epoch: string;
  seq: number;
  captured: number;
  received: number;
  position: number[];
  camera_to_world: number[][];
};
export type CameraLocations = {
  coordinate_system: string;
  units: string;
  samples: CameraLocationSample[];
};
export type Scene = {
  id: string;
  created: number;
  sample: boolean;
  title: string;
  clouds: Cloud[];
  diagnostics: Diagnostics | null;
  scale_source: string;
  provenance: string;
  reconstruction?: {
    method: 'joint_vggt' | 'joint_amb3r';
    model?: string;
    model_variant?: string;
    frames_per_source: number[];
    anchor_times: number[];
    anchor_reciprocal_matches: number;
    overlap_verified: boolean;
    quality_note: string;
    elapsed_seconds: number;
    compute_device: string;
  };
  live?: {
    session_id: string;
    batch: number;
    segment: string;
    elapsed_seconds: number;
    continuity: { status: string; reason: string };
    camera_locations?: CameraLocations;
  };
  landmarks?: { source_points: number[][]; target_points: number[][] };
};
export type State = {
  captures: {
    id: string;
    source: string;
    name: string;
    created: number;
    url: string;
  }[];
  jobs: {
    id: string;
    kind: string;
    status: string;
    stage: string;
    error?: string;
    created: number;
  }[];
  scenes: Scene[];
  worker: { online: boolean; device: string };
  mode: string;
};
