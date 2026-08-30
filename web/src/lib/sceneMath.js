import * as THREE from 'three';

export const AXIS_COLORS = Object.freeze({ x: 0xff0000, y: 0x00ff00, z: 0x0000ff });

// Keep the logged right-handed frame, but present it in Three's Y-up scene:
// raw +X -> scene +X, raw +Y -> scene -Z, raw +Z -> scene +Y.
const rawToSceneBasis = new THREE.Matrix3().set(
  1, 0, 0,
  0, 0, 1,
  0, -1, 0,
);

export const rawToScene = ([x, y, z]) => new THREE.Vector3(x, z, -y);

export function rawRotationQuaternion(rpy) {
  if (!rpy) return new THREE.Quaternion();
  const [roll, pitch, yaw] = rpy;
  const cx = Math.cos(roll); const sx = Math.sin(roll);
  const cy = Math.cos(pitch); const sy = Math.sin(pitch);
  const cz = Math.cos(yaw); const sz = Math.sin(yaw);
  const raw = new THREE.Matrix3().set(
    cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
    sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
    -sy, cy * sx, cy * cx,
  );
  const mapped = rawToSceneBasis.clone().multiply(raw).multiply(rawToSceneBasis.clone().transpose());
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().setFromMatrix3(mapped));
}

export function gridCellPosition(origin, spacing, row, col) {
  return [origin[0] - row * spacing, origin[1] - col * spacing, origin[2]];
}
