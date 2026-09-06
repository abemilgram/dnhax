'use client';

import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type {
  TacticalSnapshot,
  TrackSnapshot,
  TrajectoryCandidate,
  Vec3,
} from './types';
import { covarianceEllipse95 } from './state.mjs';

function point([x, y, z]: Vec3, lift = 0): THREE.Vector3 {
  return new THREE.Vector3(x, y + lift, z);
}

function disposeGroup(group: THREE.Group) {
  group.traverse((object) => {
    if (
      object instanceof THREE.Mesh ||
      object instanceof THREE.Line ||
      object instanceof THREE.LineSegments
    ) {
      object.geometry.dispose();
      const materials = Array.isArray(object.material)
        ? object.material
        : [object.material];
      materials.forEach((material) => material.dispose());
    }
  });
  group.clear();
}

function addTrack(
  group: THREE.Group,
  track: TrackSnapshot,
  ghost: boolean,
) {
  const conflicting = track.evidence === 'conflicting';
  const geometry = conflicting
    ? new THREE.OctahedronGeometry(0.42, 0)
    : new THREE.SphereGeometry(0.38, 12, 8);
  const material = new THREE.MeshStandardMaterial({
    color: ghost ? '#9aa7af' : conflicting ? '#f5b942' : '#f0f4f5',
    wireframe: ghost,
    transparent: ghost,
    opacity: ghost ? 0.62 : 1,
    roughness: 0.72,
  });
  const marker = new THREE.Mesh(geometry, material);
  marker.position.copy(point(track.xyz, 0.42));
  marker.scale.set(1, 1.45, 1);
  group.add(marker);

  const ellipse = covarianceEllipse95(track.covariance);
  const points: THREE.Vector3[] = [];
  for (let index = 0; index <= 64; index += 1) {
    const theta = (index / 64) * Math.PI * 2;
    const localX = ellipse.major * Math.cos(theta);
    const localZ = ellipse.minor * Math.sin(theta);
    const cosine = Math.cos(ellipse.angle);
    const sine = Math.sin(ellipse.angle);
    points.push(
      new THREE.Vector3(
        track.xyz[0] + localX * cosine - localZ * sine,
        0.065,
        track.xyz[2] + localX * sine + localZ * cosine,
      ),
    );
  }
  const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
  const lineMaterial = ghost
    ? new THREE.LineDashedMaterial({
        color: '#b7c0c6',
        dashSize: 0.24,
        gapSize: 0.18,
        transparent: true,
        opacity: 0.72,
      })
    : new THREE.LineBasicMaterial({
        color: conflicting ? '#f5b942' : '#f0f4f5',
        transparent: true,
        opacity: 0.86,
      });
  const line = new THREE.Line(lineGeometry, lineMaterial);
  if (ghost) line.computeLineDistances();
  group.add(line);
}

function addRoute(
  group: THREE.Group,
  candidate: TrajectoryCandidate,
  rank: number,
) {
  const route = candidate.route.map((entry) => point(entry.xyz, 0.14 + rank * 0.025));
  if (route.length < 2) return;
  const colors = ['#48d8bd', '#f5c763', '#ae9ef5'];
  const color = colors[rank] ?? '#cbd5da';
  const distance = route[0].distanceTo(route[route.length - 1]);
  if (rank === 0 && distance > 0.01) {
    const curve = new THREE.CatmullRomCurve3(route);
    const tube = new THREE.Mesh(
      new THREE.TubeGeometry(curve, Math.max(16, route.length * 2), 0.095, 6),
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: candidate.valid ? 0.94 : 0.42,
      }),
    );
    group.add(tube);
    return;
  }
  const geometry = new THREE.BufferGeometry().setFromPoints(route);
  const material = new THREE.LineDashedMaterial({
    color,
    dashSize: rank === 1 ? 0.45 : 0.12,
    gapSize: rank === 1 ? 0.22 : 0.2,
    transparent: true,
    opacity: candidate.valid ? 0.84 : 0.38,
  });
  const line = new THREE.Line(geometry, material);
  line.computeLineDistances();
  group.add(line);
}

export default function TacticalViewer({
  snapshot,
}: {
  snapshot: TacticalSnapshot;
}) {
  const host = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const dynamicRef = useRef<THREE.Group | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const lastRevision = useRef(-1);

  useEffect(() => {
    const container = host.current;
    if (!container) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch {
      container.textContent = 'WebGL is unavailable in this browser.';
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor('#071116');
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog('#071116', 34, 65);
    scene.add(new THREE.HemisphereLight('#b9e7df', '#10191d', 1.25));
    const key = new THREE.DirectionalLight('#ffffff', 1.1);
    key.position.set(8, 18, -8);
    scene.add(key);

    const camera = new THREE.PerspectiveCamera(46, 1, 0.05, 100);
    camera.position.set(23, 24, -25);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.enableDamping = true;
    controls.maxPolarAngle = Math.PI / 2.04;
    controls.minDistance = 10;
    controls.maxDistance = 55;
    controls.update();

    sceneRef.current = scene;
    cameraRef.current = camera;
    controlsRef.current = controls;
    const resize = new ResizeObserver(() => {
      const { width, height } = container.getBoundingClientRect();
      renderer.setSize(Math.max(width, 1), Math.max(height, 1), false);
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
    });
    resize.observe(container);

    let frame = 0;
    let disposed = false;
    const draw = () => {
      if (disposed) return;
      controls.update();
      renderer.render(scene, camera);
      frame = requestAnimationFrame(draw);
    };
    draw();
    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      resize.disconnect();
      controls.dispose();
      if (dynamicRef.current) disposeGroup(dynamicRef.current);
      scene.clear();
      renderer.dispose();
      renderer.domElement.remove();
      sceneRef.current = null;
      dynamicRef.current = null;
      cameraRef.current = null;
      controlsRef.current = null;
    };
  }, []);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || snapshot.revision <= lastRevision.current) return;
    const revision = snapshot.revision;
    let cancelled = false;
    const frame = requestAnimationFrame(() => {
      if (cancelled || revision <= lastRevision.current) return;
      lastRevision.current = revision;
      if (dynamicRef.current) {
        scene.remove(dynamicRef.current);
        disposeGroup(dynamicRef.current);
      }
      const group = new THREE.Group();
      dynamicRef.current = group;

      const floor = new THREE.Mesh(
        new THREE.PlaneGeometry(30, 24),
        new THREE.MeshStandardMaterial({
          color: '#101d22',
          roughness: 0.95,
          metalness: 0.05,
        }),
      );
      floor.rotation.x = -Math.PI / 2;
      floor.position.y = -0.025;
      group.add(floor);

      for (const zone of snapshot.map.zones) {
        const width = zone.max[0] - zone.min[0];
        const depth = zone.max[2] - zone.min[2];
        const zoneMesh = new THREE.Mesh(
          new THREE.PlaneGeometry(width, depth),
          new THREE.MeshBasicMaterial({
            color: '#1d6e68',
            transparent: true,
            opacity: 0.08,
            depthWrite: false,
          }),
        );
        zoneMesh.rotation.x = -Math.PI / 2;
        zoneMesh.position.set(
          (zone.min[0] + zone.max[0]) / 2,
          0.012,
          (zone.min[2] + zone.max[2]) / 2,
        );
        group.add(zoneMesh);
      }

      for (const obstacle of snapshot.map.obstacles) {
        const size = new THREE.Vector3(
          obstacle.max[0] - obstacle.min[0],
          obstacle.max[1] - obstacle.min[1],
          obstacle.max[2] - obstacle.min[2],
        );
        const mesh = new THREE.Mesh(
          new THREE.BoxGeometry(size.x, size.y, size.z),
          new THREE.MeshStandardMaterial({
            color: '#2c3d43',
            roughness: 0.86,
            flatShading: true,
          }),
        );
        mesh.position.set(
          (obstacle.min[0] + obstacle.max[0]) / 2,
          (obstacle.min[1] + obstacle.max[1]) / 2,
          (obstacle.min[2] + obstacle.max[2]) / 2,
        );
        group.add(mesh);
        const edges = new THREE.LineSegments(
          new THREE.EdgesGeometry(mesh.geometry),
          new THREE.LineBasicMaterial({
            color: '#587078',
            transparent: true,
            opacity: 0.48,
          }),
        );
        edges.position.copy(mesh.position);
        group.add(edges);
      }

      const nodes = new Map(snapshot.map.nodes.map((node) => [node.id, node]));
      for (const edge of snapshot.map.edges) {
        const start = nodes.get(edge.from);
        const end = nodes.get(edge.to);
        if (!start || !end) continue;
        const geometry = new THREE.BufferGeometry().setFromPoints([
          point(start.xyz, 0.035),
          point(end.xyz, 0.035),
        ]);
        const line = new THREE.Line(
          geometry,
          new THREE.LineBasicMaterial({
            color: '#4d7776',
            transparent: true,
            opacity: 0.34,
          }),
        );
        group.add(line);
      }

      const actor = new THREE.Mesh(
        new THREE.CylinderGeometry(0.35, 0.45, 0.85, 10),
        new THREE.MeshStandardMaterial({
          color: '#4bd9be',
          roughness: 0.5,
          flatShading: true,
        }),
      );
      actor.position.copy(point(snapshot.actor.xyz, 0.43));
      group.add(actor);

      snapshot.tracks.forEach((track) => addTrack(group, track, false));
      snapshot.ghosts.forEach((track) => addTrack(group, track, true));
      snapshot.ranking?.candidates
        .slice(0, 3)
        .forEach((candidate, index) => addRoute(group, candidate, index));
      scene.add(group);
    });
    return () => {
      cancelled = true;
      cancelAnimationFrame(frame);
    };
  }, [snapshot]);

  const setCamera = (top: boolean) => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;
    camera.position.set(top ? 0.01 : 23, top ? 34 : 24, top ? 0.01 : -25);
    controls.target.set(0, 0, 0);
    controls.update();
  };

  return (
    <div className="tactical-viewer-frame">
      <figure
        ref={host}
        className="tactical-canvas"
        aria-label="Interactive metric twin showing route futures and belief uncertainty"
      />
      <div className="tactical-camera-controls" aria-label="Camera views">
        <button type="button" onClick={() => setCamera(true)}>
          Top
        </button>
        <button type="button" onClick={() => setCamera(false)}>
          Oblique
        </button>
      </div>
      <div className="tactical-route-key" aria-label="Route line styles">
        <span><i className="route-solid" />1 · strongest · solid ribbon</span>
        <span><i className="route-dashed" />2 · alternate · dashed</span>
        <span><i className="route-dotted" />3 · alternate · dotted</span>
      </div>
    </div>
  );
}
