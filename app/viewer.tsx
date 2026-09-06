'use client';
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Button } from '@/components/ui/button';
import { RotateCcw, ScanLine } from 'lucide-react';
import type { Scene } from './types';

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
      world.traverse((o) => {
        if (o instanceof THREE.Points || o instanceof THREE.LineSegments) {
          o.geometry.dispose();
          const mats = Array.isArray(o.material) ? o.material : [o.material];
          mats.forEach((m) => m.dispose());
        }
      });
      context.current = null;
      renderer.dispose();
      renderer.domElement.remove();
      objects.current = [];
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
        const next: THREE.Points[] = [];
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
          points.userData = {
            source: cloud.source,
            transform: cloud.transform,
            spacing,
          };
          points.matrixAutoUpdate = false;
          points.matrix.set(
            ...(cloud.transform.flat() as [
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
          if (!initialAligned) points.matrix.identity();
          points.visible = initialVisible[cloud.source] !== false;
          next.push(points);
        }
        const previous = objects.current;
        next.forEach((points) => world.add(points));
        objects.current = next;
        previous.forEach((points) => {
          world.remove(points);
          points.geometry.dispose();
          const materials = Array.isArray(points.material)
            ? points.material
            : [points.material];
          materials.forEach((material) => material.dispose());
        });
        const bounds = new THREE.Box3();
        objects.current.forEach((o) => {
          o.updateMatrixWorld(true);
          bounds.union(new THREE.Box3().setFromObject(o));
        });
        const center = bounds.getCenter(new THREE.Vector3());
        const size = Math.max(bounds.getSize(new THREE.Vector3()).length(), 1);
        grid.scale.setScalar(size / 10);
        grid.position.y = bounds.min.y - 0.01;
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
      points.matrix.identity();
      if (aligned)
        points.matrix.set(
          ...(points.userData.transform.flat() as [
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
      material.size =
        points.userData.spacing *
        pointSize *
        Math.cbrt(Math.abs(points.matrix.determinant()));
      points.matrixWorldNeedsUpdate = true;
    }
  }, [aligned, visible, pointSize, colorBySource, loading]);
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
              : scene.reconstruction?.method === 'joint_vggt'
                ? 'Joint reconstruction · shared coordinates'
                : 'Independent coordinate frames'}
      </div>
      <div className="scene-controls">
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
