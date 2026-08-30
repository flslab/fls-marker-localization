import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { Crosshair, Maximize2, RotateCcw } from 'lucide-react';
import { asVector } from '../lib/logModel.js';
import { AXIS_COLORS, gridCellPosition, rawRotationQuaternion, rawToScene } from '../lib/sceneMath.js';

function lineMaterial(color, opacity = 1) {
  return new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity });
}

function textSprite(text, size) {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  context.font = '600 22px ui-monospace, monospace';
  canvas.width = Math.ceil(context.measureText(text).width + 20);
  canvas.height = 42;
  context.fillStyle = 'rgba(7, 17, 15, 0.88)';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.font = '600 22px ui-monospace, monospace';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillStyle = '#d9fff0';
  context.fillText(text, canvas.width / 2, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false, depthWrite: false }));
  sprite.scale.set(size * 0.38 * (canvas.width / canvas.height), size * 0.38, 1);
  sprite.renderOrder = 10;
  return sprite;
}

function buildCoordinateFrame(label, size) {
  const group = new THREE.Group();
  for (const [axis, direction] of Object.entries({ x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] })) {
    group.add(new THREE.ArrowHelper(rawToScene(direction), new THREE.Vector3(), size, AXIS_COLORS[axis], size * 0.2, size * 0.09));
  }
  const labelSprite = textSprite(label, size);
  labelSprite.position.copy(rawToScene([0, 0, size * 1.3]));
  group.add(labelSprite);
  return group;
}

function buildCamera(rpy, color = 0xff9f66) {
  const group = new THREE.Group();
  const vertices = [
    [0, 0, 0], [-0.07, -0.045, 0.13],
    [0, 0, 0], [0.07, -0.045, 0.13],
    [0, 0, 0], [0.07, 0.045, 0.13],
    [0, 0, 0], [-0.07, 0.045, 0.13],
    [-0.07, -0.045, 0.13], [0.07, -0.045, 0.13],
    [0.07, -0.045, 0.13], [0.07, 0.045, 0.13],
    [0.07, 0.045, 0.13], [-0.07, 0.045, 0.13],
    [-0.07, 0.045, 0.13], [-0.07, -0.045, 0.13],
  ].flatMap((point) => rawToScene(point).toArray());
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  group.add(new THREE.LineSegments(geometry, lineMaterial(color)));
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(0.085, 0.07, 0.055),
    new THREE.MeshStandardMaterial({ color: 0x13231e, emissive: color, emissiveIntensity: 0.24, roughness: 0.5 }),
  );
  group.add(body, buildCoordinateFrame('camera', 0.11));
  group.quaternion.copy(rawRotationQuaternion(rpy));
  return group;
}

function addGrid(group, grid) {
  const rows = Number.isInteger(grid?.rows) ? grid.rows : 0;
  const cols = Number.isInteger(grid?.cols) ? grid.cols : 0;
  const spacing = typeof grid?.cell_spacing === 'number' ? grid.cell_spacing : 0.1;
  const origin = asVector(grid?.grid_origin) || [0, 0, 0];
  if (!rows || !cols) return;
  const positions = [];
  for (let row = 0; row < rows; row += 1) {
    positions.push(...rawToScene(gridCellPosition(origin, spacing, row, 0)).toArray());
    positions.push(...rawToScene(gridCellPosition(origin, spacing, row, cols - 1)).toArray());
  }
  for (let col = 0; col < cols; col += 1) {
    positions.push(...rawToScene(gridCellPosition(origin, spacing, 0, col)).toArray());
    positions.push(...rawToScene(gridCellPosition(origin, spacing, rows - 1, col)).toArray());
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  group.add(new THREE.LineSegments(geometry, lineMaterial(0x315b4d, 0.85)));
}

function addTrajectory(group, model, useFiltered) {
  const points = [];
  const segmentEnds = [];
  for (let index = 1; index < model.worldCameraPath.length; index += 1) {
    const previous = model.worldCameraPath[index - 1];
    const current = model.worldCameraPath[index];
    // A missing pose remains a visible break instead of being bridged.
    if (current.frameIndex !== previous.frameIndex + 1) continue;
    points.push(rawToScene((useFiltered && previous.filteredPosition) || previous.position));
    points.push(rawToScene((useFiltered && current.filteredPosition) || current.position));
    segmentEnds.push(current.frameIndex);
  }
  if (!points.length) return null;
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const line = new THREE.LineSegments(geometry, lineMaterial(useFiltered ? 0x9df7c7 : 0x61d9f4, 0.9));
  group.add(line);
  return { line, segmentEnds };
}

function addMarker(group, marker, highlighted, frameLabel = null) {
  const color = marker.kind === 'aruco' ? 0x61d9f4 : 0x9df7c7;
  const markerGroup = new THREE.Group();
  markerGroup.position.copy(rawToScene(marker.position));
  markerGroup.quaternion.copy(rawRotationQuaternion(marker.orientation));
  const mesh = new THREE.Mesh(
    marker.kind === 'aruco' ? new THREE.BoxGeometry(0.036, 0.006, 0.036) : new THREE.SphereGeometry(highlighted ? 0.012 : 0.008, 18, 12),
    new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: highlighted ? 0.75 : 0.25, roughness: 0.32 }),
  );
  markerGroup.add(mesh);
  if (frameLabel) markerGroup.add(buildCoordinateFrame(frameLabel, 0.075));
  if (highlighted) {
    const halo = new THREE.Mesh(
      new THREE.RingGeometry(0.016, 0.021, 24),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.6, side: THREE.DoubleSide }),
    );
    halo.rotation.x = -Math.PI / 2;
    markerGroup.add(halo);
  }
  group.add(markerGroup);
}

function disposeGroup(group) {
  group.traverse((object) => {
    // ArrowHelper owns its materials, but Three shares its line/cone geometry.
    if (object.geometry && object.parent?.type !== 'ArrowHelper') object.geometry.dispose();
    if (object.material) {
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      materials.forEach((material) => { material.map?.dispose(); material.dispose(); });
    }
  });
  group.clear();
}

function fitDisplay(state, viewDirection = new THREE.Vector3(1, 0.65, 1)) {
  const box = new THREE.Box3().setFromObject(state.display);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(0.15, box.getBoundingSphere(new THREE.Sphere()).radius);
  const verticalFov = THREE.MathUtils.degToRad(state.camera.fov);
  const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * state.camera.aspect);
  const distance = radius / Math.sin(Math.min(verticalFov, horizontalFov) / 2) * 1.1;
  const direction = viewDirection.clone().normalize();
  state.controls.target.copy(center);
  state.camera.position.copy(center).addScaledVector(direction, distance);
  state.controls.update();
}

export default function WorldScene({ model, frameIndex, useFiltered, markerId = null }) {
  const mountRef = useRef(null);
  const sceneState = useRef(null);
  const fittedModel = useRef(null);
  const frame = model.frames[frameIndex];
  const currentMatchedKeys = useMemo(() => {
    const keys = new Set((frame?.grid?.matched_markers || []).map((marker) => `grid:${marker.map_row ?? '?'}:${marker.map_col ?? '?'}:${marker.id ?? '?'}`));
    for (const pose of frame?.poseRecords || []) {
      if (pose?.marker_pose === true) keys.add(`aruco:${pose.marker_id ?? '?'}`);
      for (const marker of Array.isArray(pose?.marker_poses) ? pose.marker_poses : []) keys.add(`aruco:${marker.marker_id ?? '?'}`);
    }
    return keys;
  }, [frame]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return undefined;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x081512);
    const camera = new THREE.PerspectiveCamera(40, 1, 0.005, 100);
    camera.position.set(1.05, 0.92, 1.05);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    mount.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0);
    controls.minDistance = 0.12;
    scene.add(new THREE.HemisphereLight(0xd9fff0, 0x13241f, 2.1));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.5);
    keyLight.position.set(1.5, 2, 1);
    scene.add(keyLight);
    const display = new THREE.Group();
    const staticGroup = new THREE.Group();
    const frameGroup = new THREE.Group();
    display.add(staticGroup, frameGroup);
    scene.add(display);

    const resize = () => {
      const width = Math.max(1, mount.clientWidth);
      const height = Math.max(1, mount.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(mount);
    resize();
    let animationFrame;
    const render = () => {
      controls.update();
      renderer.render(scene, camera);
      animationFrame = requestAnimationFrame(render);
    };
    render();
    fittedModel.current = null;
    sceneState.current = { scene, camera, renderer, controls, display, staticGroup, frameGroup, trajectory: null };
    return () => {
      cancelAnimationFrame(animationFrame);
      observer.disconnect();
      controls.dispose();
      disposeGroup(staticGroup);
      disposeGroup(frameGroup);
      renderer.dispose();
      renderer.domElement.remove();
      sceneState.current = null;
    };
  }, []);

  useEffect(() => {
    const state = sceneState.current;
    if (!state) return;
    disposeGroup(state.staticGroup);
    addGrid(state.staticGroup, model.config?.marker_grid);
    if (model.config?.marker_grid || model.worldCameraPath.length || model.worldMarkers.length) {
      state.staticGroup.add(buildCoordinateFrame(model.config?.marker_grid ? 'world / grid' : 'world', 0.16));
    }
    for (const marker of model.worldMarkers) {
      addMarker(state.staticGroup, marker, false, marker.kind === 'aruco' ? `marker ${marker.id ?? '?'}` : null);
    }
    state.trajectory = addTrajectory(state.staticGroup, model, useFiltered);
  }, [model, useFiltered]);

  useEffect(() => {
    const state = sceneState.current;
    if (!state) return;
    disposeGroup(state.frameGroup);
    if (state.trajectory) {
      const visibleSegments = state.trajectory.segmentEnds.filter((endFrame) => endFrame <= frameIndex).length;
      state.trajectory.line.geometry.setDrawRange(0, visibleSegments * 2);
    }
    for (const marker of model.worldMarkers) {
      if (currentMatchedKeys.has(marker.key)) addMarker(state.frameGroup, marker, true);
    }

    const selectedLegacyPose = markerId === null ? null : frame?.poses.find((pose) => (pose.kind === 'legacy' || pose.kind === 'historical-marker') && pose.markerId === markerId);
    // Once an ID is selected, an absent record is a real gap. Never substitute
    // another marker's pose just to keep the 3D object moving.
    const primary = markerId === null ? frame?.primary : (selectedLegacyPose || null);
    const cameraFrameMarkers = (frame?.poses || []).filter((pose) => (
      ((pose.kind === 'legacy' && pose.entity === 'marker') || pose.kind === 'historical-marker')
      && pose.position
    ));
    if (primary?.kind === 'camera-world' && primary.position) {
      const cameraObject = buildCamera(primary.orientation);
      cameraObject.position.copy(rawToScene((useFiltered && primary.filteredPosition) || primary.position));
      state.frameGroup.add(cameraObject);
    } else if (cameraFrameMarkers.length) {
      // These records place markers in the camera frame, so all markers share
      // one origin and can be shown together even when the selected ID is absent.
      state.frameGroup.add(buildCamera(null));
      for (const pose of cameraFrameMarkers) {
        addMarker(
          state.frameGroup,
          { kind: 'legacy', position: (useFiltered && pose.filteredPosition) || pose.position, orientation: pose.orientation },
          pose === primary,
          `marker ${pose.markerId ?? '?'}`,
        );
      }
    } else if (primary?.kind === 'legacy' && primary.position) {
      if (primary.entity === 'camera') {
        const cameraObject = buildCamera(primary.orientation, 0x61d9f4);
        cameraObject.position.copy(rawToScene((useFiltered && primary.filteredPosition) || primary.position));
        state.frameGroup.add(cameraObject);
        addMarker(state.frameGroup, { kind: 'legacy', position: [0, 0, 0], orientation: null }, true, `marker ${primary.markerId ?? '?'}`);
      }
    }
    if (fittedModel.current !== model) {
      fitDisplay(state);
      fittedModel.current = model;
    }
  }, [model, frameIndex, useFiltered, markerId, currentMatchedKeys]);

  const resetView = () => {
    const state = sceneState.current;
    if (!state) return;
    fitDisplay(state);
  };
  const topView = () => {
    const state = sceneState.current;
    if (!state) return;
    fitDisplay(state, new THREE.Vector3(0, 1, 0.001));
  };
  const fitView = () => {
    const state = sceneState.current;
    if (!state) return;
    fitDisplay(state);
  };

  return (
    <div className="world-scene-wrap">
      <div ref={mountRef} className="world-scene" role="img" aria-label="Interactive 3D spatial view with labeled world, camera, and marker coordinate frames; X axes are red, Y axes green, and Z axes blue" />
      <div className="scene-controls" aria-label="3D view controls">
        <button onClick={resetView} title="Reset view"><RotateCcw size={14} /><span>Reset</span></button>
        <button onClick={fitView} title="Fit all objects"><Maximize2 size={14} /><span>Fit</span></button>
        <button onClick={topView} title="Top view"><Crosshair size={14} /><span>Top</span></button>
      </div>
      <div className="scene-legend">
        <span><i className="legend-camera" />camera</span>
        <span><i className="legend-path" />trajectory</span>
        <span><i className="legend-marker" />markers</span>
      </div>
      <p className="scale-note">glyphs schematic · not to scale</p>
      <p className="coordinate-note">frames <b className="axis-x">X</b> <b className="axis-y">Y</b> <b className="axis-z">Z</b> · right-handed +Z-up view · values unchanged</p>
      {!model.hasPoseData && !model.worldMarkers.length && <div className="spatial-empty"><strong>No 3D pose data</strong><span>Image detections and raw frame diagnostics remain available.</span></div>}
    </div>
  );
}
