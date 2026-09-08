import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildMarkerGridFile, canonicalRing, generateRingSignatures, hasRotationalSymmetry, hypergridMarkersForTile, rotateRing,
} from './markerGrid.js';

test('cyclic signatures ignore their starting marker', () => {
  const ring = [0, 0, 1, 0, 1, 1];
  assert.deepEqual(canonicalRing(ring), canonicalRing(rotateRing(ring, 4)));
});

test('generated four-marker MyGrid signatures are asymmetric and unique under rotation and reflection', () => {
  const signatures = generateRingSignatures(10, 4, 4);
  assert.equal(signatures.length, 10);
  assert.ok(signatures.every((signature) => !hasRotationalSymmetry(signature)));
  assert.equal(new Set(signatures.map((signature) => canonicalRing(signature, true).join(','))).size, signatures.length);
});

test('exports an infinite HyperGrid and negative MyGrid coordinates', () => {
  const output = buildMarkerGridFile({
    hypergridSpacing: 0.2,
    origin: [1, -2, 0.5],
    maxIds: 9,
    mygridSpacing: 0.04,
    blinkFrequencyHz: 40,
  }, [{ i: -2, j: 3, events: [
    { time: 2, action: 'blinking' }, { time: 5, action: 'off' },
    { time: 6, action: 'on' }, { time: 8, action: 'blinking' },
  ] }]);

  assert.equal(output.hypergrid.infinite, true);
  assert.equal(output.hypergrid.tile_size, 0.4);
  assert.equal(output.hypergrid.markers_per_tile, 4);
  assert.equal(output.hypergrid.markers_have_ids, false);
  assert.deepEqual(output.mygrid.tiles[0].center, [0.2, -0.8, 0.5]);
  assert.equal(output.mygrid.markers_per_tile, 4);
  assert.equal(output.mygrid.tiles[0].markers.length, 4);
  assert.deepEqual(output.mygrid.tiles[0].markers.map((marker) => marker.global_position), [
    [0.18, -0.78, 0.5], [0.22, -0.78, 0.5],
    [0.22, -0.82, 0.5], [0.18, -0.82, 0.5],
  ]);
  assert.equal(output.encoding.payload_bits, 4);
  assert.equal(output.encoding.delimiter_bits, 5);
  assert.equal(output.encoding.delimiter_pattern, '111110');
  assert.deepEqual(output.mygrid.tiles[0].blender_animation.events, [
    { time_s: 2, state: 'blinking' }, { time_s: 5, state: 'off' },
    { time_s: 6, state: 'on' }, { time_s: 8, state: 'blinking' },
  ]);
});

test('each tile owns four centered HyperGrid markers with grid and global coordinates', () => {
  const markers = hypergridMarkersForTile({ hypergridSpacing: 0.2, origin: [1, -2, 0.5] }, -2, 3);
  assert.equal(markers.length, 4);
  assert.deepEqual(markers.map((marker) => marker.grid_coordinates), [[-5, 6], [-4, 6], [-4, 5], [-5, 5]]);
  assert.deepEqual(markers.map((marker) => marker.global_position), [
    [0.1, -0.7, 0.5], [0.3, -0.7, 0.5],
    [0.3, -0.9, 0.5], [0.1, -0.9, 0.5],
  ]);
});

test('omits animation for default-off MyGrids and rejects ambiguous event lists', () => {
  const output = buildMarkerGridFile({}, [{ i: 0, j: 0 }]);
  assert.equal(output.mygrid.tiles[0].blender_animation, undefined);
  assert.deepEqual(
    buildMarkerGridFile({}, [{ i: 0, j: 0, events: [{ time: 1, action: 'on' }] }]).mygrid.tiles[0].blender_animation.events,
    [{ time_s: 1, state: 'on' }],
  );
  assert.throws(() => buildMarkerGridFile({}, [{ i: 0, j: 0, events: [{ time: 1, action: 'pulse' }] }]), /invalid action/);
  assert.throws(() => buildMarkerGridFile({}, [{ i: 0, j: 0, events: [{ time: 1, action: 'off' }, { time: 1, action: 'static' }] }]), /two events/);
  assert.throws(() => buildMarkerGridFile({}, [{ i: 0, j: 0 }, { i: 0, j: 0 }]), /more than once/);
});
