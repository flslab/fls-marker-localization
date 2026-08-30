import test from 'node:test';
import assert from 'node:assert/strict';
import { classifyPose, createLogModel, flattenObject, formatRawTime, timeToSeconds } from './logModel.js';
import { AXIS_COLORS, gridCellPosition, rawRotationQuaternion, rawToScene } from './sceneMath.js';

test('maps the defined right-handed frames into a Z-up RGB scene', () => {
  assert.deepEqual(rawToScene([1, 2, 3]).toArray(), [1, 3, -2]);
  assert.deepEqual(AXIS_COLORS, { x: 0xff0000, y: 0x00ff00, z: 0x0000ff });
  const rotatedX = rawToScene([1, 0, 0]).applyQuaternion(rawRotationQuaternion([0, 0, Math.PI / 2]));
  assert.ok(rotatedX.distanceTo(rawToScene([0, 1, 0])) < 1e-12);
});

test('maps grid rows to world -X and columns to world -Y', () => {
  const origin = [0.2, 0.4, 0.5];
  assert.deepEqual(gridCellPosition(origin, 0.1, 0, 0), origin);
  assert.deepEqual(gridCellPosition(origin, 0.1, 2, 3), [0, 0.09999999999999998, 0.5]);
});

test('models current blob-grid camera records without losing diagnostics', () => {
  const raw = {
    args: { aruco_mode: false },
    config: { blob_grid_localization_enabled: true, marker_grid: { rows: 2, cols: 2 } },
    frames: [{
      time: 4,
      frame_id: 7,
      blobs: [{ id: 3, x: 12, y: 14 }],
      poses: [{ camera_pose: true, source: 'blob_grid', camera_position: [1, 2, -3], camera_orientation: [0.1, 0.2, 0.3], markers_used: 4 }],
      blob_grid_localization: { status: 'success', pose_valid: true, accepted_marker_count: 4, matched_markers: [] },
    }],
  };
  const model = createLogModel(raw, 'grid.json');
  assert.equal(model.mode, 'Blob grid');
  assert.equal(model.frames[0].frameId, 7);
  assert.deepEqual(model.frames[0].primary.position, [1, 2, -3]);
  assert.equal(model.frames[0].status, 'success');
  assert.equal(model.frames[0].counts.accepted, 4);
});

test('recognizes ArUco marker and camera pose variants', () => {
  assert.equal(classifyPose({ camera_pose: true, camera_position: [0, 0, 1], camera_orientation: [0, 0, 0] }).source, 'aruco');
  const marker = classifyPose({ marker_pose: true, marker_id: 5, marker_position: [1, 2, 3], marker_orientation: [0, 0, 1] });
  assert.equal(marker.kind, 'marker-world');
  assert.equal(marker.markerId, 5);
});

test('keeps legacy coordinate frames explicit', () => {
  const pose = classifyPose({
    marker_id: 2,
    camera_position: [1, 0, 0], camera_orientation: [0, 0, 0],
    marker_position: [-1, 0, 0], marker_orientation: [0, 0, 0],
    marker_position_filtered: [-0.9, 0, 0],
  });
  assert.equal(pose.kind, 'legacy');
  assert.equal(pose.frameLabel, 'camera frame');
  assert.deepEqual(pose.cameraPosition, [1, 0, 0]);
  assert.deepEqual(pose.filteredPosition, [-0.9, 0, 0]);
});

test('supports historical milliseconds and yaw-pitch-roll ordering', () => {
  const model = createLogModel({
    args: {}, config: {},
    frames: [
      { frame_id: 0, time: 1700000000000, tvec: [1, 2, 3], yaw_pitch_roll: [0.3, 0.2, 0.1] },
      { frame_id: 1, time: 1700000000010, tvec: [2, 3, 4], yaw_pitch_roll: [0.4, 0.2, 0.2] },
    ],
  });
  assert.equal(timeToSeconds(1700000000000), 1700000000);
  assert.equal(model.frames[1].t, 0.009999990463256836);
  assert.deepEqual(model.frames[0].primary.orientation, [0.1, 0.2, 0.3]);
  assert.ok(model.fps > 99 && model.fps < 101);
  assert.equal(formatRawTime(1700000000010), '1700000000010');
});

test('supports producer-shaped historical ArUco camera records', () => {
  const model = createLogModel({
    args: {}, config: {},
    frames: [{
      frame_id: 8,
      time: 1700000000100,
      poses: [{
        camera_pose: true,
        tvec: [1, 2, 3],
        tvec_filtered: [1.1, 2.1, 3.1],
        yaw_pitch_roll: [0.3, 0.2, 0.1],
      }],
    }],
  }, 'old-aruco.json');
  assert.equal(model.mode, 'ArUco');
  assert.deepEqual(model.frames[0].primary.position, [1, 2, 3]);
  assert.deepEqual(model.frames[0].primary.filteredPosition, [1.1, 2.1, 3.1]);
  assert.deepEqual(model.frames[0].primary.orientation, [0.1, 0.2, 0.3]);
  assert.equal(model.worldMarkers.length, 0);
});

test('converts nested historical ArUco marker YPR into RPY', () => {
  const model = createLogModel({
    args: { aruco_mode: true }, config: {},
    frames: [{ poses: [{
      camera_pose: true,
      tvec: [1, 2, 3],
      yaw_pitch_roll: [0.3, 0.2, 0.1],
      marker_poses: [{ marker_id: 4, marker_position: [4, 5, 6], marker_orientation: [0.6, 0.5, 0.4] }],
    }] }],
  });
  assert.deepEqual(model.worldMarkers[0].orientation, [0.4, 0.5, 0.6]);
});

test('uses explicit filtered camera positions from intermediate legacy logs', () => {
  const pose = classifyPose({
    marker_id: 2,
    camera_position: [1, 2, 3], camera_position_filtered: [1.1, 2.1, 3.1], camera_orientation: [0.1, 0.2, 0.3],
    marker_position: [-1, -2, -3], marker_orientation: [-0.1, -0.2, -0.3],
  });
  assert.equal(pose.entity, 'camera');
  assert.equal(pose.frameLabel, 'marker frame');
  assert.deepEqual(pose.position, [1, 2, 3]);
  assert.deepEqual(pose.filteredPosition, [1.1, 2.1, 3.1]);
  assert.deepEqual(pose.orientation, [0.1, 0.2, 0.3]);
});

test('collects current marker-world records without changing roll-pitch-yaw order', () => {
  const model = createLogModel({
    args: { aruco_mode: true }, config: {},
    frames: [{ poses: [{ marker_pose: true, marker_id: 7, marker_position: [4, 5, 6], marker_orientation: [0.4, 0.5, 0.6] }] }],
  });
  assert.deepEqual(model.worldMarkers[0].position, [4, 5, 6]);
  assert.deepEqual(model.worldMarkers[0].orientation, [0.4, 0.5, 0.6]);
  assert.equal(model.worldMarkers[0].id, 7);
});

test('does not fabricate unavailable diagnostics or reinterpret direct tvec as a camera world pose', () => {
  const model = createLogModel({ args: {}, config: {}, frames: [{ time: 1, tvec: [1, 2, 3], marker_id: 9 }] });
  assert.equal(model.frames[0].primary.kind, 'historical-marker');
  assert.equal(model.frames[0].primary.frameLabel, 'camera frame');
  assert.equal(model.frames[0].counts.blobs, null);
  assert.equal(model.frames[0].counts.decoded, null);
  assert.equal(model.frames[0].counts.matched, null);
  assert.equal(model.frames[0].counts.markersUsed, null);
});

test('preserves metadata when frames are absent and flattens false/null/empty values', () => {
  const model = createLogModel({ args: { enabled: false, missing: null, path: '' }, config: {} });
  assert.equal(model.frames.length, 0);
  assert.match(model.warnings[0], /no frames array/i);
  const fields = flattenObject({ enabled: false, missing: null, empty: [] });
  assert.deepEqual(fields.map((field) => field.type), ['boolean', 'null', 'empty array']);
});
