'use client';
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Button } from '@/components/ui/button';
import { Camera as CameraIcon, RotateCcw, ScanLine } from 'lucide-react';
import type { CameraLocationSample, Scene } from './types';

type CameraOverlaySample = {
  object: THREE.Group;
  pose: THREE.Matrix4;
  source: string;
};
type CameraOverlayTrajectory = {
  object: THREE.Line;
  transform: THREE.Matrix4;
  source: string;
};
type CameraOverlay = {
  group: THREE.Group;
  samples: CameraOverlaySample[];
  trajectories: CameraOverlayTrajectory[];
  transforms: Map<string, THREE.Matrix4>;
  latest: Map<string, CameraLocationSample>;
};

const SOURCE_COLORS = [
  0x55c6aa, 0xe3a871, 0x8ea9e8, 0xd08bd7, 0xe8d273, 0x83c99b,
];

function sourceColor(source: string) {
  if (source === 'A') return SOURCE_COLORS[0];
  if (source === 'B') return SOURCE_COLORS[1];
  let hash = 0;
  for (let i = 0; i < source.length; i++)
    hash = (hash * 31 + source.charCodeAt(i)) | 0;
  return SOURCE_COLORS[Math.abs(hash) % SOURCE_COLORS.length];
}

function matrixFromRows(value: unknown): THREE.Matrix4 | null {
  if (!Array.isArray(value) || value.length !== 4) return null;
  const rows = value as unknown[];
  if (!rows.every((row) => Array.isArray(row) && row.length === 4)) return null;
  const values = rows.flat() as unknown[];
  if (
    !values.every(
      (entry) => typeof entry === 'number' && Number.isFinite(entry),
    )
  )
    return null;
  // Matrix4.set takes values in row-major order, matching the manifest.
  return new THREE.Matrix4().set(
    ...(values as [
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
    ]),
  );
}

function positionFromSample(sample: CameraLocationSample) {
  if (!Array.isArray(sample.position) || sample.position.length < 3)
    return null;
  if (!sample.position.slice(0, 3).every((value) => Number.isFinite(value)))
    return null;
  return new THREE.Vector3(
    sample.position[0],
    sample.position[1],
    sample.position[2],
  );
}

function formatCoordinate(value: number) {
  if (
    Math.abs(value) >= 10000 ||
    (Math.abs(value) > 0 && Math.abs(value) < 0.001)
  )
    return value.toExponential(2);
  return value.toFixed(3);
}

function setTransform(object: THREE.Object3D, matrix: THREE.Matrix4) {
  object.matrixAutoUpdate = false;
  object.matrix.copy(matrix);
  object.matrixWorldNeedsUpdate = true;
}

function disposeMaterial(material: THREE.Material) {
  material.dispose();
}

function disposeObjectResources(root: THREE.Object3D) {
  const geometries = new Set<THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();
  root.traverse((object) => {
    if ('geometry' in object && object.geometry instanceof THREE.BufferGeometry)
      geometries.add(object.geometry);
    if ('material' in object) {
      const material = object.material;
      if (Array.isArray(material))
        material.forEach((entry) => materials.add(entry));
      else if (material instanceof THREE.Material) materials.add(material);
    }
  });
  geometries.forEach((geometry) => geometry.dispose());
  materials.forEach(disposeMaterial);
}

function latestSamples(samples: CameraLocationSample[]) {
  const latest = new Map<string, CameraLocationSample>();
  for (const sample of samples) {
    if (
      !sample ||
      typeof sample.source !== 'string' ||
      !positionFromSample(sample)
    )
      continue;
    const previous = latest.get(sample.source);
    if (
      !previous ||
      (sample.epoch === previous.epoch
        ? sample.seq > previous.seq
        : sample.received > previous.received)
    )
      latest.set(sample.source, sample);
  }
  return latest;
}

function sortedTrajectories(samples: CameraLocationSample[]) {
  const groups = new Map<string, CameraLocationSample[]>();
  samples.forEach((sample) => {
    if (typeof sample.source !== 'string' || typeof sample.epoch !== 'string')
      return;
    if (!positionFromSample(sample)) return;
    const key = `${sample.source}\u0000${sample.epoch}`;
    const group = groups.get(key) || [];
    group.push(sample);
    groups.set(key, group);
  });
  return [...groups.values()].map((group) =>
    group.sort((a, b) => a.seq - b.seq || a.received - b.received),
  );
}

function updateCameraOverlay(
  overlay: CameraOverlay,
  aligned: boolean,
  visible: Record<string, boolean>,
  showCameras: boolean,
) {
  overlay.group.visible = showCameras;
  for (const sample of overlay.samples) {
    const transform = overlay.transforms.get(sample.source);
    const matrix =
      aligned && transform
        ? transform.clone().multiply(sample.pose)
        : sample.pose;
    setTransform(sample.object, matrix);
    sample.object.visible = visible[sample.source] !== false;
  }
  for (const trajectory of overlay.trajectories) {
    setTransform(
      trajectory.object,
      aligned ? trajectory.transform : new THREE.Matrix4(),
    );
    trajectory.object.visible = visible[trajectory.source] !== false;
  }
}

function createCameraOverlay(
  samples: CameraLocationSample[],
  transforms: Map<string, THREE.Matrix4>,
  scale: number,
) {
  const group = new THREE.Group();
  group.name = 'camera-locations';
  const overlay: CameraOverlay = {
    group,
    samples: [],
    trajectories: [],
    transforms,
    latest: latestSamples(samples),
  };
  const markerRadius = Math.max(scale * 0.018, 0.001);
  const frustumDepth = Math.max(scale * 0.16, markerRadius * 5);
  const frustumHalfWidth = Math.max(scale * 0.085, markerRadius * 3);
  const frustumHalfHeight = Math.max(scale * 0.06, markerRadius * 2);
  const markerGeometry = new THREE.SphereGeometry(markerRadius, 12, 8);
  // Predicted poses use OpenCV camera axes: +Z points forward.
  const frustumVertices = new Float32Array([
    0,
    0,
    0,
    -frustumHalfWidth,
    frustumHalfHeight,
    frustumDepth,
    0,
    0,
    0,
    frustumHalfWidth,
    frustumHalfHeight,
    frustumDepth,
    0,
    0,
    0,
    frustumHalfWidth,
    -frustumHalfHeight,
    frustumDepth,
    0,
    0,
    0,
    -frustumHalfWidth,
    -frustumHalfHeight,
    frustumDepth,
    -frustumHalfWidth,
    frustumHalfHeight,
    frustumDepth,
    frustumHalfWidth,
    frustumHalfHeight,
    frustumDepth,
    frustumHalfWidth,
    frustumHalfHeight,
    frustumDepth,
    frustumHalfWidth,
    -frustumHalfHeight,
    frustumDepth,
    frustumHalfWidth,
    -frustumHalfHeight,
    frustumDepth,
    -frustumHalfWidth,
    -frustumHalfHeight,
    frustumDepth,
    -frustumHalfWidth,
    -frustumHalfHeight,
    frustumDepth,
    -frustumHalfWidth,
    frustumHalfHeight,
    frustumDepth,
  ]);
  for (const sample of samples) {
    const position = positionFromSample(sample);
    const pose = matrixFromRows(sample.camera_to_world);
    if (!position || !pose) continue;
    // Position is the authoritative location field. Keep orientation from the
    // pose matrix while preventing stale matrix translation from moving it.
    pose.setPosition(position);
    const color = sourceColor(sample.source);
    const marker = new THREE.Mesh(
      markerGeometry.clone(),
      new THREE.MeshBasicMaterial({ color, toneMapped: false }),
    );
    const frustumGeometry = new THREE.BufferGeometry();
    frustumGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(frustumVertices.slice(), 3),
    );
    const frustum = new THREE.LineSegments(
      frustumGeometry,
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.78 }),
    );
    const cameraObject = new THREE.Group();
    cameraObject.name = `camera-${sample.source}-${sample.epoch}-${sample.seq}`;
    cameraObject.userData.source = sample.source;
    cameraObject.add(marker, frustum);
    setTransform(cameraObject, pose);
    group.add(cameraObject);
    overlay.samples.push({ object: cameraObject, pose, source: sample.source });
  }
  for (const trajectory of sortedTrajectories(samples)) {
    if (trajectory.length < 2) continue;
    const geometry = new THREE.BufferGeometry().setFromPoints(
      trajectory.map((sample) => positionFromSample(sample)!),
    );
    const source = trajectory[0].source;
    const line = new THREE.Line(
      geometry,
      new THREE.LineBasicMaterial({
        color: sourceColor(source),
        transparent: true,
        opacity: 0.5,
      }),
    );
    line.name = `trajectory-${source}-${trajectory[0].epoch}`;
    line.userData.source = source;
    setTransform(line, new THREE.Matrix4());
    group.add(line);
    overlay.trajectories.push({
      object: line,
      transform: transforms.get(source)?.clone() || new THREE.Matrix4(),
      source,
    });
  }
  markerGeometry.dispose();
  return overlay;
}

export default function Viewer({
  scene,
  aligned,
  visible,
  pointSize,
  colorBySource,
  onPick,
  pickSource,
}: {
  scene: Scene;
  aligned: boolean;
  visible: Record<string, boolean>;
  pointSize: number;
  colorBySource: boolean;
  onPick?: (point: number[]) => void;
  pickSource?: string;
}) {
  const host = useRef<HTMLDivElement>(null);
  const objects = useRef<THREE.Points[]>([]);
  const cameraOverlay = useRef<CameraOverlay | null>(null);
  const context = useRef<{
    world: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    grid: THREE.GridHelper;
  } | null>(null);
  const displayedSegment = useRef('');
  const reset = useRef<() => void>(() => {});
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [showCameras, setShowCameras] = useState(Boolean(scene.live));
  const [cameraReadouts, setCameraReadouts] = useState<CameraLocationSample[]>(
    [],
  );
  const cameraSceneKind = useRef<boolean | null>(Boolean(scene.live));
  const showCamerasRef = useRef(showCameras);
  showCamerasRef.current = showCameras;
  const pick = useRef({ onPick, pickSource });
  pick.current = { onPick, pickSource };
  const setupState = useRef({ scene, aligned, visible, pointSize });
  setupState.current = { scene, aligned, visible, pointSize };
  useEffect(() => {
    const target = host.current;
    if (!target) return;
    setError('');
    setLoading(true);
    let disposed = false;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    } catch {
      setError(
        'WebGL is unavailable. Enable hardware acceleration or use another browser.',
      );
      setLoading(false);
      return;
    }
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setClearColor('#101c24');
    target.appendChild(renderer.domElement);
    const world = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 100000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    const grid = new THREE.GridHelper(12, 24, 0x3a505d, 0x24343e);
    world.add(grid);
    context.current = { world, camera, controls, grid };
    objects.current = [];
    cameraOverlay.current = null;
    let frame = 0;
    function draw() {
      if (disposed) return;
      controls.update();
      renderer.render(world, camera);
      frame = requestAnimationFrame(draw);
    }
    draw();
    const resize = new ResizeObserver(() => {
      const { width, height } = target.getBoundingClientRect();
      renderer.setSize(width, height);
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
    });
    resize.observe(target);
    const raycaster = new THREE.Raycaster();
    let pointerStart = [0, 0];
    const down = (event: PointerEvent) => {
      pointerStart = [event.clientX, event.clientY];
    };
    const click = (event: PointerEvent) => {
      const { onPick: callback, pickSource: source } = pick.current;
      if (
        !callback ||
        !source ||
        Math.hypot(
          event.clientX - pointerStart[0],
          event.clientY - pointerStart[1],
        ) > 5
      )
        return;
      const rect = renderer.domElement.getBoundingClientRect();
      raycaster.setFromCamera(
        new THREE.Vector2(
          ((event.clientX - rect.left) / rect.width) * 2 - 1,
          (-(event.clientY - rect.top) / rect.height) * 2 + 1,
        ),
        camera,
      );
      const cloud = objects.current.find((p) => p.userData.source === source);
      if (!cloud) return;
      raycaster.params.Points!.threshold =
        camera.position.distanceTo(controls.target) * 0.007;
      const hit = raycaster.intersectObject(cloud)[0];
      if (hit?.index !== undefined) {
        const attr = cloud.geometry.getAttribute('position');
        callback([
          attr.getX(hit.index),
          attr.getY(hit.index),
          attr.getZ(hit.index),
        ]);
      }
    };
    renderer.domElement.addEventListener('pointerdown', down);
    renderer.domElement.addEventListener('pointerup', click);
    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      resize.disconnect();
      controls.dispose();
      renderer.domElement.removeEventListener('pointerdown', down);
      renderer.domElement.removeEventListener('pointerup', click);
      disposeObjectResources(world);
      context.current = null;
      renderer.dispose();
      renderer.domElement.remove();
      objects.current = [];
      cameraOverlay.current = null;
    };
  }, []);
  useEffect(() => {
    const {
      scene: snapshot,
      aligned: initialAligned,
      visible: initialVisible,
      pointSize: initialPointSize,
    } = setupState.current;
    const current = context.current;
    if (!current) return;
    const { world, camera, controls, grid } = current;
    const abort = new AbortController();
    let disposed = false;
    setLoading(true);
    setError('');
    async function load() {
      const next: THREE.Points[] = [];
      let nextOverlay: CameraOverlay | null = null;
      try {
        const loaded = await Promise.all(
          snapshot.clouds.map(async (cloud) => {
            const [p, c] = await Promise.all([
              fetch(cloud.points, { signal: abort.signal }),
              fetch(cloud.colors, { signal: abort.signal }),
            ]);
            if (!p.ok || !c.ok) throw Error('Could not load scene geometry.');
            const positions = new Float32Array(await p.arrayBuffer());
            const colors = new Uint8Array(await c.arrayBuffer());
            if (
              positions.length !== cloud.count * 3 ||
              colors.length !== cloud.count * 3 ||
              positions.some((value) => !Number.isFinite(value))
            )
              throw Error('Scene geometry has an invalid length.');
            return { cloud, positions, colors };
          }),
        );
        if (disposed) return;
        const transforms = new Map<string, THREE.Matrix4>();
        for (const { cloud, positions, colors } of loaded) {
          const geometry = new THREE.BufferGeometry();
          geometry.setAttribute(
            'position',
            new THREE.BufferAttribute(positions, 3),
          );
          // Image pixels are sRGB; Three's vertex colors use linear light.
          const linear = new Float32Array(colors.length);
          const lut = new Float32Array(256);
          for (let i = 0; i < 256; i++) {
            const c = i / 255;
            lut[i] =
              c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
          }
          for (let i = 0; i < colors.length; i++) linear[i] = lut[colors[i]];
          geometry.setAttribute('color', new THREE.BufferAttribute(linear, 3));
          geometry.computeBoundingBox();
          const diagonal = geometry
            .boundingBox!.getSize(new THREE.Vector3())
            .length();
          // Surface-spacing estimate in this cloud's native coordinate units.
          const spacing =
            Math.max(diagonal, 1e-6) / Math.sqrt(Math.max(cloud.count, 1));
          const material = new THREE.PointsMaterial({
            size: spacing * initialPointSize,
            sizeAttenuation: true,
            vertexColors: true,
            toneMapped: false,
            color: '#ffffff',
          });
          // Round, opaque footprints close small gaps without Gaussian training.
          material.onBeforeCompile = (shader) => {
            shader.fragmentShader = shader.fragmentShader.replace(
              '#include <clipping_planes_fragment>',
              '#include <clipping_planes_fragment>\n vec2 offset = gl_PointCoord - vec2(0.5); if (dot(offset, offset) > 0.25) discard;',
            );
          };
          const points = new THREE.Points(geometry, material);
          const transform =
            matrixFromRows(cloud.transform) || new THREE.Matrix4();
          transforms.set(cloud.source, transform.clone());
          points.userData = {
            source: cloud.source,
            transform,
            spacing,
          };
          setTransform(
            points,
            initialAligned ? transform : new THREE.Matrix4(),
          );
          points.visible = initialVisible[cloud.source] !== false;
          next.push(points);
        }
        const bounds = new THREE.Box3();
        next.forEach((o) => {
          o.updateMatrixWorld(true);
          bounds.union(new THREE.Box3().setFromObject(o));
        });
        const center = bounds.isEmpty()
          ? new THREE.Vector3()
          : bounds.getCenter(new THREE.Vector3());
        const size = bounds.isEmpty()
          ? 1
          : Math.max(bounds.getSize(new THREE.Vector3()).length(), 1);
        const cameraLocations = snapshot.live?.camera_locations;
        if (cameraLocations && Array.isArray(cameraLocations.samples)) {
          nextOverlay = createCameraOverlay(
            cameraLocations.samples,
            transforms,
            size,
          );
          updateCameraOverlay(
            nextOverlay,
            initialAligned,
            initialVisible,
            showCamerasRef.current,
          );
        }
        if (disposed) {
          next.forEach(disposeObjectResources);
          if (nextOverlay) disposeObjectResources(nextOverlay.group);
          return;
        }
        // Add both layers before removing the old pair so a completed batch is
        // committed in one synchronous scene update. A failed fetch never gets
        // here, so the previously displayed geometry and overlay stay intact.
        const previous = objects.current;
        const previousOverlay = cameraOverlay.current;
        next.forEach((points) => world.add(points));
        if (nextOverlay) world.add(nextOverlay.group);
        objects.current = next;
        cameraOverlay.current = nextOverlay;
        previous.forEach((points) => {
          world.remove(points);
          disposeObjectResources(points);
        });
        if (previousOverlay) {
          world.remove(previousOverlay.group);
          disposeObjectResources(previousOverlay.group);
        }
        if (nextOverlay)
          updateCameraOverlay(
            nextOverlay,
            initialAligned,
            initialVisible,
            showCamerasRef.current,
          );
        setCameraReadouts(nextOverlay ? [...nextOverlay.latest.values()] : []);
        if (cameraSceneKind.current !== Boolean(snapshot.live)) {
          cameraSceneKind.current = Boolean(snapshot.live);
          setShowCameras(Boolean(snapshot.live));
        }
        grid.scale.setScalar(size / 10);
        grid.position.y =
          (bounds.isEmpty() ? center.y - size / 2 : bounds.min.y) - 0.01;
        reset.current = () => {
          camera.position
            .copy(center)
            .add(new THREE.Vector3(size * 0.65, size * 0.48, size * 0.7));
          controls.target.copy(center);
          camera.near = size / 10000;
          camera.far = size * 100;
          camera.updateProjectionMatrix();
          controls.update();
        };
        const segment = snapshot.live?.segment || snapshot.id;
        if (displayedSegment.current !== segment) reset.current();
        displayedSegment.current = segment;
        setLoading(false);
      } catch (e) {
        if (!disposed) {
          next.forEach(disposeObjectResources);
          if (nextOverlay) disposeObjectResources(nextOverlay.group);
          setError(e instanceof Error ? e.message : 'Could not load scene.');
          setLoading(false);
        }
      }
    }
    void load();
    return () => {
      disposed = true;
      abort.abort();
    };
  }, [scene.id]);
  useEffect(() => {
    const live = Boolean(scene.live);
    if (cameraSceneKind.current === live) return;
    cameraSceneKind.current = live;
    setShowCameras(live);
  }, [scene.id, scene.live]);
  useEffect(() => {
    for (const points of objects.current) {
      const material = points.material as THREE.PointsMaterial;

      material.vertexColors = !colorBySource;
      material.color.set(
        colorBySource
          ? points.userData.source === 'A'
            ? '#55c6aa'
            : '#e3a871'
          : '#ffffff',
      );
      material.needsUpdate = true;
      points.visible = visible[points.userData.source] !== false;
      setTransform(
        points,
        aligned ? points.userData.transform : new THREE.Matrix4(),
      );
      material.size =
        points.userData.spacing *
        pointSize *
        Math.cbrt(Math.abs(points.matrix.determinant()));
    }
    if (cameraOverlay.current)
      updateCameraOverlay(cameraOverlay.current, aligned, visible, showCameras);
  }, [aligned, visible, pointSize, colorBySource, loading, showCameras]);
  return (
    <>
      <div
        ref={host}
        style={{ position: 'absolute', inset: 0 }}
        aria-label="Interactive point-cloud viewer"
      />
      <div className="canvas-label">
        <ScanLine size={16} />
        {pickSource
          ? `Pick a point in source ${pickSource}`
          : scene.live
            ? `Live batch ${scene.live.batch} · ${scene.live.continuity.status === 'accepted' ? 'continuous preview' : 'new segment'}`
            : aligned
              ? 'Registered scene'
              : scene.reconstruction?.method?.startsWith('joint_')
                ? 'Joint reconstruction · shared coordinates'
                : 'Independent coordinate frames'}
      </div>
      <div className="scene-controls">
        {(scene.live || cameraReadouts.length > 0) && (
          <Button
            variant="outline"
            aria-pressed={showCameras}
            aria-label={
              showCameras ? 'Hide camera locations' : 'Show camera locations'
            }
            onClick={() => setShowCameras((current) => !current)}
            title={
              showCameras ? 'Hide camera locations' : 'Show camera locations'
            }
          >
            <CameraIcon size={15} />
            {showCameras ? 'Cameras on' : 'Cameras off'}
          </Button>
        )}
        <Button
          variant="outline"
          onClick={() => reset.current()}
          aria-label="Reset camera"
        >
          <RotateCcw size={15} />
        </Button>
      </div>
      {(loading || error) && (
        <div
          className={objects.current.length ? 'canvas-label' : 'empty-scene'}
          style={objects.current.length ? { top: 54 } : undefined}
        >
          <p role={error ? 'alert' : 'status'}>
            {error || 'Loading scene geometry…'}
          </p>
        </div>
      )}
      {showCameras && cameraReadouts.length > 0 && (
        <output
          aria-label="Latest camera locations in arbitrary units"
          style={{
            position: 'absolute',
            top: 68,
            left: 20,
            zIndex: 2,
            maxWidth: 'calc(100% - 40px)',
            padding: '9px 11px',
            borderRadius: 6,
            background: '#152530e8',
            color: '#d7e5e8',
            fontSize: 12,
            lineHeight: 1.55,
            pointerEvents: 'none',
          }}
        >
          <span style={{ color: '#9eb1bb', marginBottom: 3, display: 'block' }}>
            Camera locations · arbitrary units
          </span>
          {[...cameraReadouts]
            .filter((sample) => visible[sample.source] !== false)
            .sort((a, b) => a.source.localeCompare(b.source))
            .map((sample) => (
              <span key={sample.source} style={{ display: 'block' }}>
                <span
                  style={{
                    color: `#${sourceColor(sample.source).toString(16).padStart(6, '0')}`,
                  }}
                >
                  Source {sample.source}
                </span>{' '}
                x {formatCoordinate(sample.position[0])} · y{' '}
                {formatCoordinate(sample.position[1])} · z{' '}
                {formatCoordinate(sample.position[2])}
              </span>
            ))}
        </output>
      )}
      <div className="canvas-footer">
        <span>
          {scene.clouds.map((cloud, index) => (
            <span
              key={cloud.source}
              style={{
                marginLeft: index ? 15 : 0,
                opacity: visible[cloud.source] === false ? 0.45 : 1,
              }}
            >
              <i className={`dot ${cloud.source === 'B' ? 'b' : ''}`} />
              Source {cloud.source}
              {visible[cloud.source] === false ? ' (hidden)' : ''}
            </span>
          ))}
        </span>
        <span>Drag to orbit · scroll to zoom</span>
      </div>
    </>
  );
}
