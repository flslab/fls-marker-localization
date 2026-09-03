import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { load as loadYaml } from 'js-yaml';
import {
  allocateTakeoffs, buildFlightModel, footprintForPose, normalizeFlight, normalizeGrid, solveMinimumWindowCover, stateAt,
} from './flightModel.js';

const gridFixture = {
  rows: 8,
  cols: 8,
  num_ids: 64,
  cell_spacing: 0.5,
  marker_size: 0.05,
  window_size: 2,
  grid_origin: [1.75, 1.75, 0],
  working_range: { min_distance: 0.2, max_distance: 4 },
  range_model: {
    focal_length: 1,
    sensor_width: 4,
    sensor_height: 4,
    usable_width_fraction: 1,
    usable_height_fraction: 1,
    resolution_width: 100,
    resolution_height: 100,
  },
  grid: Array.from({ length: 8 }, (_, row) => Array.from({ length: 8 }, (_, col) => row * 8 + col)),
};
const ZERO_CAMERA_OFFSET = [0, 0, 0];

test('models vertical takeoff, target transit, and waypoint dt on one timeline', () => {
  const mission = {
    name: 'path timing',
    takeoff_speed: 1,
    drones: {
      lb1: {
        target: [1, 0, 1, Math.PI / 2],
        waypoints: [[1, 0, 1, Math.PI / 2, 0], [2, 0, 1, Math.PI / 2, 4]],
        delta_t: 0,
        iterations: 1,
        params: { linear: true, relative: false },
      },
    },
  };
  const model = buildFlightModel(mission, gridFixture, { lb1: [0, 0] }, ZERO_CAMERA_OFFSET);
  const drone = model.drones[0];
  assert.equal(model.duration, 16.1);
  assert.deepEqual(stateAt(drone, 0.5).position, [0, 0, 0.5]);
  assert.equal(stateAt(drone, 0.5).yaw, 0);
  assert.deepEqual(stateAt(drone, 4).position, [0.5, 0, 1]);
  assert.ok(Math.abs(stateAt(drone, 4).yaw - Math.PI / 4) < 1e-12);
  assert.deepEqual(stateAt(drone, 9).position, [1.5, 0, 1]);
  assert.equal(stateAt(drone, 11).phase, 'waypoint 1');
  assert.ok(model.coverage.belowRangeSamples > 0);
});

test('rotates the rectangular camera footprint with waypoint yaw', () => {
  const grid = normalizeGrid({
    ...gridFixture,
    range_model: { ...gridFixture.range_model, sensor_width: 2, sensor_height: 1 },
  });
  const axisAligned = footprintForPose([0, 0, 1], 0, grid).corners;
  const rotated = footprintForPose([0, 0, 1], Math.PI / 2, grid).corners;
  const span = (points, axis) => Math.max(...points.map((point) => point[axis])) - Math.min(...points.map((point) => point[axis]));
  assert.ok(Math.abs(span(axisAligned, 0) - 1) < 1e-9);
  assert.ok(Math.abs(span(axisAligned, 1) - 2) < 1e-9);
  assert.ok(Math.abs(span(rotated, 0) - 2) < 1e-9);
  assert.ok(Math.abs(span(rotated, 1) - 1) < 1e-9);
});

test('rotates the drone-frame camera offset into each footprint centre', () => {
  const mission = {
    takeoff_speed: 1,
    drones: {
      d: { target: [1, 0, 1, Math.PI / 2], waypoints: [], delta_t: 0, iterations: 1 },
    },
  };
  const model = buildFlightModel(mission, gridFixture, { d: [0, 0] }, [1, 0, -0.1]);
  const atTarget = model.frames.find((frame) => Math.abs(frame.time - 7) < 1e-12).drones[0];
  assert.ok(Math.abs(atTarget.cameraPosition[0] - 1) < 1e-12);
  assert.ok(Math.abs(atTarget.cameraPosition[1] - 1) < 1e-12);
  assert.ok(Math.abs(atTarget.cameraPosition[2] - 0.9) < 1e-12);
  assert.deepEqual(atTarget.footprint.center, atTarget.cameraPosition.slice(0, 2));
});

test('includes cells crossed only by a yawing camera offset in the accumulated trace', () => {
  const angle = Math.PI / 72;
  const grid = {
    rows: 2,
    cols: 2,
    num_ids: 4,
    cell_spacing: 0.1,
    marker_size: 0.001,
    window_size: 2,
    grid_origin: [Math.cos(angle), Math.sin(angle), 0],
    working_range: { min_distance: 0.5, max_distance: 2 },
    range_model: {
      focal_length: 1, sensor_width: 0.002, sensor_height: 0.002,
      usable_width_fraction: 1, usable_height_fraction: 1,
      resolution_width: 100, resolution_height: 100,
    },
    grid: [[0, 1], [2, 3]],
  };
  const mission = {
    takeoff_speed: 1,
    drones: {
      d: {
        target: [0, 0, 1, 0],
        waypoints: [[0, 0, 1, 0, 0], [0, 0, 1, Math.PI / 2, 1]],
        delta_t: 0,
        iterations: 1,
        params: { linear: true, relative: false },
      },
    },
  };
  const model = buildFlightModel(mission, grid, { d: [0, 0] }, [1, 0, 0]);
  const yawEndsAt = model.drones[0].segments.find((segment) => segment.phase === 'waypoint 1').t1;
  assert.equal(model.frames.filter((frame) => frame.time <= yawEndsAt + 1e-12).some((frame) => frame.drones[0].observedCellKeys.includes('0:0')), false);
  assert.ok(model.observedCellKeys.includes('0:0'));
  assert.ok(model.observedCellTimes['0:0'] > 1 && model.observedCellTimes['0:0'] < 1.1);
});

test('preserves SFL yaw radians and includes the controller loop synchronization', () => {
  const raw = {
    takeoff_speed: 1,
    drones: {
      lb1: {
        target: [0, 0, 1, Math.PI / 2],
        waypoints: [[0, 0, 1, Math.PI / 2, 0], [0, 0, 1, Math.PI, 1]],
        delta_t: 0,
        iterations: 2,
        params: { linear: true, relative: false },
      },
    },
  };
  const mission = normalizeFlight(raw);
  assert.ok(Math.abs(mission.drones[0].targetYaw - Math.PI / 2) < 1e-9);
  assert.ok(Math.abs(mission.drones[0].waypoints[1].yaw - Math.PI) < 1e-9);
  const model = buildFlightModel(raw, gridFixture, { lb1: [0, 0] }, ZERO_CAMERA_OFFSET);
  assert.ok(Math.abs(model.duration - 4.3) < 1e-9);
});

test('uses the controller seventh-degree trajectory when linear is false', () => {
  const mission = {
    takeoff_speed: 1,
    drones: {
      lb1: {
        target: [0, 0, 1, 0],
        waypoints: [[0, 0, 1, 0, 0], [1, 0, 1, 0, 4]],
        delta_t: 0,
        iterations: 1,
        params: { linear: false, relative: false },
      },
    },
  };
  const model = buildFlightModel(mission, gridFixture, { lb1: [0, 0] }, ZERO_CAMERA_OFFSET);
  assert.ok(Math.abs(stateAt(model.drones[0], 2).position[0] - 0.070556640625) < 1e-12);
});

test('uses target altitude for the controller landing duration', () => {
  const mission = {
    takeoff_speed: 1,
    drones: {
      d: {
        target: [0, 0, 1, 0],
        waypoints: [[0, 0, 1, 0, 0], [0, 0, 2, 0, 1]],
        delta_t: 0,
        iterations: 1,
        params: { linear: true, relative: false },
      },
    },
  };
  const model = buildFlightModel(mission, gridFixture, { d: [0, 0] }, ZERO_CAMERA_OFFSET);
  assert.equal(model.drones[0].segments.at(-1).phase, 'landing');
  assert.equal(model.drones[0].segments.at(-1).duration, 1);
});

test('derives working range directly from pixel intrinsics', () => {
  const grid = normalizeGrid({
    rows: 2,
    cols: 2,
    num_ids: 4,
    cell_spacing: 0.1,
    marker_size: 0.01,
    window_size: 2,
    grid_origin: [0.05, 0.05, 0],
    grid: [[0, 1], [2, 3]],
    image_resolution_pixels: [100, 100],
    focal_length_pixels: [1000, 1000],
  });
  assert.ok(Math.abs(grid.workingRange.min - 1.375) < 1e-9);
  assert.ok(Math.abs(grid.workingRange.max - 11 / 3) < 1e-9);
});

test('rejects grids with no IDs or no feasible optical range', () => {
  assert.throws(() => normalizeGrid({
    rows: 2, cols: 2, num_ids: 4, cell_spacing: 0.1, marker_size: 0.01,
  }), /complete grid ID matrix/);
  assert.throws(() => normalizeGrid({
    ...gridFixture,
    working_range: undefined,
    range_model: { ...gridFixture.range_model, min_marker_px: 1000, min_bbox_px: 3000 },
  }), /no feasible working range/);
  assert.throws(() => normalizeGrid({ ...gridFixture, window_size: 1 }), /at least 2/);
  assert.throws(() => normalizeGrid({ ...gridFixture, num_ids: 1 }), /integer in \[0, num_ids\)/);
});

test('requires each target to be above the configured marker plane', () => {
  const mission = {
    drones: {
      d: { target: [0, 0, 0.5], waypoints: [], delta_t: 0, iterations: 1 },
    },
  };
  assert.throws(
    () => buildFlightModel(mission, { ...gridFixture, grid_origin: [1.75, 1.75, 1] }, { d: [0, 0] }, ZERO_CAMERA_OFFSET),
    /above the marker plane at z=1/,
  );
  mission.drones.d.target[2] = 1.5;
  mission.drones.d.waypoints = [[0, 0, 1.5, 0, 0], [0, 0, 0.5, 0, 1]];
  assert.throws(
    () => buildFlightModel(mission, { ...gridFixture, grid_origin: [1.75, 1.75, 1] }, { d: [0, 0] }, ZERO_CAMERA_OFFSET),
    /waypoints\[1\]\[2\] must stay above the marker plane/,
  );
});

test('excludes reference-marker entries from relative SFL swarms', () => {
  const mission = normalizeFlight({
    drones: {
      lb1: {
        target: [0, 0, 1, 0],
        waypoints: [[0, 0, 1, 0, 0]],
        delta_t: 0,
        iterations: 1,
        relative_anchor: { id: 'm1', method: 'ekf', source: 'tracker' },
        params: { linear: true, relative: false },
      },
      m1: { target: [0, 0, 0, 0, 0], waypoints: [[0, 0, 0, 0, 0]] },
    },
  });
  assert.deepEqual(mission.drones.map((drone) => drone.id), ['lb1']);
  assert.match(mission.warnings[0], /reference marker entry/);
});

test('requires distinct explicit takeoff positions', () => {
  const mission = {
    drones: {
      a: { target: [0, 0, 1], waypoints: [], delta_t: 0, iterations: 1 },
      b: { target: [1, 0, 1], waypoints: [], delta_t: 0, iterations: 1 },
    },
  };
  assert.throws(() => buildFlightModel(mission, gridFixture, { a: [0, 0], b: [0, 0] }, ZERO_CAMERA_OFFSET), /distinct XY/);
});

test('deduplicates repeated short-range tiles before assigning homes', () => {
  const grid = normalizeGrid({
    ...gridFixture,
    short_range: {
      cell_spacing: 0.1,
      tiles: [{ i: 0, j: 0 }, { i: 0, j: 0 }, { i: 2, j: 2 }],
    },
  });
  const drones = normalizeFlight({
    drones: {
      a: { target: [1.5, 1.5, 1], waypoints: [], delta_t: 0, iterations: 1 },
      b: { target: [1.4, 1.4, 1], waypoints: [], delta_t: 0, iterations: 1 },
    },
  }).drones;
  const homes = allocateTakeoffs(drones, grid, {}, ZERO_CAMERA_OFFSET);
  assert.deepEqual([...homes.values()].map((home) => home.window.key).sort(), ['0:0', '2:2']);
});

test('adaptively samples a short leg whose midpoint loses every complete window', () => {
  const grid = {
    rows: 3,
    cols: 2,
    num_ids: 6,
    cell_spacing: 1,
    marker_size: 0.1,
    window_size: 2,
    grid_origin: [1, 0.5, 0],
    working_range: { min_distance: 0.55, max_distance: 10 },
    range_model: {
      focal_length: 1, sensor_width: 2, sensor_height: 2,
      usable_width_fraction: 1, usable_height_fraction: 1,
      resolution_width: 100, resolution_height: 100,
    },
    grid: [[0, 1], [2, 3], [4, 5]],
  };
  const mission = {
    takeoff_speed: 0.5,
    drones: {
      d: {
        target: [0.5, 0, 0.55, 0],
        waypoints: [[0.5, 0, 0.55, 0, 0], [-0.5, 0, 0.55, 0, 0.001]],
        delta_t: 0,
        iterations: 1,
        params: { linear: true, relative: false },
      },
    },
  };
  const model = buildFlightModel(mission, grid, { d: [0.5, 0] }, ZERO_CAMERA_OFFSET);
  assert.ok(model.coverage.unsupportedSamples > 0);
  assert.ok(model.coverage.supportRate < 1);
});

test('does not exempt below-range horizontal flight after takeoff', () => {
  const mission = {
    takeoff_speed: 1,
    drones: {
      d: {
        target: [0, 0, 0.4, 0],
        waypoints: [[0, 0, 0.4, 0, 0], [0.1, 0, 0.4, 0, 1]],
        delta_t: 0,
        iterations: 1,
        params: { linear: true, relative: false },
      },
    },
  };
  const model = buildFlightModel(
    mission,
    { ...gridFixture, working_range: { min_distance: 0.5, max_distance: 4 } },
    { d: [0, 0] },
    ZERO_CAMERA_OFFSET,
  );
  assert.ok(model.coverage.belowRangeSamples > 0);
  assert.ok(model.coverage.unsupportedBelowRangeSamples > 0);
  assert.ok(model.coverage.unsupportedBelowRangeDuration > 0);
  assert.equal(model.coverage.supportRate, 0);
  assert.equal(model.solution.feasible, false);
});

test('excludes controller-managed vertical takeoff from main-window requirements', () => {
  const grid = {
    rows: 3,
    cols: 2,
    num_ids: 6,
    cell_spacing: 1,
    marker_size: 0.1,
    window_size: 2,
    grid_origin: [1, 0.5, 0],
    working_range: { min_distance: 1.05, max_distance: 10 },
    range_model: {
      focal_length: 1, sensor_width: 2, sensor_height: 2,
      usable_width_fraction: 1, usable_height_fraction: 1,
      resolution_width: 100, resolution_height: 100,
    },
    grid: [[0, 1], [2, 3], [4, 5]],
  };
  const mission = {
    takeoff_speed: 1,
    drones: {
      d: {
        target: [-0.5, 0.25, 1.55, 0],
        waypoints: [[-0.5, 0.25, 1.55, 0, 0], [-0.5, 0.25, 1.05, 0, 1]],
        delta_t: 0,
        iterations: 1,
        params: { linear: true, relative: false },
      },
    },
  };
  const model = buildFlightModel(mission, grid, { d: [0.5, 0.25] }, ZERO_CAMERA_OFFSET);
  assert.ok(model.coverage.shortRangeTakeoffDuration > 1.5);
  assert.ok(model.coverage.shortRangeLandingDuration > 1);
  assert.deepEqual(model.requiredWindows.map((window) => window.index), [0, 1]);
  assert.deepEqual(model.solution.preselected, [0]);
  assert.deepEqual(model.solution.added, [1]);
});

test('certifies between samples and catches an arbitrarily narrow support gap', () => {
  const grid = {
    rows: 3,
    cols: 2,
    num_ids: 6,
    cell_spacing: 1,
    marker_size: 0.1,
    window_size: 2,
    grid_origin: [1, 0.5, 0],
    working_range: { min_distance: 1, max_distance: 10 },
    range_model: {
      focal_length: 1, sensor_width: 2, sensor_height: 2,
      usable_width_fraction: 1, usable_height_fraction: 1,
      resolution_width: 100, resolution_height: 100,
    },
    grid: [[0, 1], [2, 3], [4, 5]],
  };
  const mission = {
    takeoff_speed: 0.5,
    drones: {
      d: {
        target: [0.49, 0, 1.049, 0],
        waypoints: [[0.49, 0, 1.049, 0, 0], [-0.48, 0, 1.049, 0, 0.001]],
        delta_t: 0,
        iterations: 1,
        params: { linear: true, relative: false },
      },
    },
  };
  const model = buildFlightModel(mission, grid, { d: [0.49, 0] }, ZERO_CAMERA_OFFSET);
  assert.equal(model.coverage.unsupportedSamples, 0);
  assert.equal(model.coverage.unsupportedIntervals, 2);
  assert.ok(model.coverage.supportRate > 0.99 && model.coverage.supportRate < 1);
  assert.equal(model.solution.feasible, false);
});

test('does not hide a nanometre-scale real gap behind a window handoff tolerance', () => {
  const height = 1.05 - 5e-9;
  const grid = {
    rows: 3,
    cols: 2,
    num_ids: 6,
    cell_spacing: 1,
    marker_size: 0.1,
    window_size: 2,
    grid_origin: [1, 0.5, 0],
    working_range: { min_distance: 1, max_distance: 10 },
    range_model: {
      focal_length: 1, sensor_width: 2, sensor_height: 2,
      usable_width_fraction: 1, usable_height_fraction: 1,
      resolution_width: 100, resolution_height: 100,
    },
    grid: [[0, 1], [2, 3], [4, 5]],
  };
  const mission = {
    takeoff_speed: 0.5,
    drones: {
      d: {
        target: [0.47, 0, height, 0],
        waypoints: [[0.47, 0, height, 0, 0], [-0.52, 0, height, 0, 0.001]],
        delta_t: 0,
        iterations: 1,
        params: { linear: true, relative: false },
      },
    },
  };
  const model = buildFlightModel(mission, grid, { d: [0.47, 0] }, ZERO_CAMERA_OFFSET);
  assert.equal(model.coverage.unsupportedSamples, 0);
  assert.equal(model.coverage.unsupportedIntervals, 2);
  assert.equal(model.solution.feasible, false);
});

test('certifies a tangent window while the camera moves along its supported axis', () => {
  const grid = {
    rows: 2,
    cols: 2,
    num_ids: 4,
    cell_spacing: 1,
    marker_size: 0.1,
    window_size: 2,
    grid_origin: [0.5, 0.5, 0],
    working_range: { min_distance: 1, max_distance: 10 },
    range_model: {
      focal_length: 1, sensor_width: 2, sensor_height: 1.1,
      usable_width_fraction: 1, usable_height_fraction: 1,
      resolution_width: 100, resolution_height: 100,
    },
    grid: [[0, 1], [2, 3]],
  };
  const mission = {
    takeoff_speed: 1,
    drones: {
      d: {
        target: [0, -0.1, 1, 0],
        waypoints: [[0, -0.1, 1, 0, 0], [0, 0.1, 1, 0, 0.001]],
        delta_t: 0,
        iterations: 1,
        params: { linear: true, relative: false },
      },
    },
  };
  const model = buildFlightModel(mission, grid, { d: [0, -0.1] }, ZERO_CAMERA_OFFSET);
  assert.equal(model.coverage.unsupportedSamples, 0);
  assert.equal(model.coverage.unsupportedIntervals, 0);
  assert.equal(model.coverage.supportRate, 1);
});

test('allows two windows to hand off continuous support at a shared boundary', () => {
  const grid = {
    rows: 3,
    cols: 2,
    num_ids: 6,
    cell_spacing: 1,
    marker_size: 0.1,
    window_size: 2,
    grid_origin: [1, 0.5, 0],
    working_range: { min_distance: 1.05, max_distance: 10 },
    range_model: {
      focal_length: 1, sensor_width: 2, sensor_height: 2,
      usable_width_fraction: 1, usable_height_fraction: 1,
      resolution_width: 100, resolution_height: 100,
    },
    grid: [[0, 1], [2, 3], [4, 5]],
  };
  const mission = {
    takeoff_speed: 0.5,
    drones: {
      d: {
        target: [0.47, 0, 1.05, 0],
        waypoints: [[0.47, 0, 1.05, 0, 0], [-0.52, 0, 1.05, 0, 0.001]],
        delta_t: 0,
        iterations: 1,
        params: { linear: true, relative: false },
      },
    },
  };
  const model = buildFlightModel(mission, grid, { d: [0.47, 0] }, ZERO_CAMERA_OFFSET);
  assert.equal(model.coverage.unsupportedSamples, 0);
  assert.equal(model.coverage.unsupportedIntervals, 0);
  assert.equal(model.coverage.supportRate, 1);
  assert.equal(model.solution.feasible, true);
  assert.deepEqual(model.requiredWindows.map((window) => window.index), [0, 1]);
});

test('keeps a middle window optional when two outer windows hand off support', () => {
  const grid = {
    rows: 4,
    cols: 2,
    num_ids: 8,
    cell_spacing: 1,
    marker_size: 0.1,
    window_size: 2,
    grid_origin: [1.5, 0.5, 0],
    working_range: { min_distance: 1.55, max_distance: 10 },
    range_model: {
      focal_length: 1, sensor_width: 2, sensor_height: 2,
      usable_width_fraction: 1, usable_height_fraction: 1,
      resolution_width: 100, resolution_height: 100,
    },
    grid: [[0, 1], [2, 3], [4, 5], [6, 7]],
  };
  const mission = {
    takeoff_speed: 0.5,
    drones: {
      d: {
        target: [1.21, 0, 1.55, 0],
        waypoints: [[1.21, 0, 1.55, 0, 0], [-1.2, 0, 1.55, 0, 0.001]],
        delta_t: 0,
        iterations: 1,
        params: { linear: true, relative: false },
      },
    },
  };
  const model = buildFlightModel(mission, grid, { d: [1.21, 0] }, ZERO_CAMERA_OFFSET);
  assert.equal(model.coverage.supportRate, 1);
  assert.equal(model.solution.optimal, true);
  assert.deepEqual(model.requiredWindows.map((window) => window.index), [0, 2]);
});

test('proves an exact window cover where greedy alone would choose three', () => {
  const solution = solveMinimumWindowCover([
    [0, 1], [0, 1], [0, 2], [0, 2], [1], [2],
  ], 3, { maxMs: 1000 });
  assert.deepEqual(solution.selected, [1, 2]);
  assert.equal(solution.optimal, true);
});

test('preselected launch windows satisfy route constraints before adding extras', () => {
  const solution = solveMinimumWindowCover([[0, 2], [1, 3]], 4, { preselected: [2, 3] });
  assert.deepEqual(solution.selected, [2, 3]);
  assert.deepEqual(solution.added, []);
  assert.equal(solution.requirementCount, 2);
  assert.equal(solution.residualRequirementCount, 0);
  assert.equal(solution.optimal, true);
});

test('reserves one launch and landing window per la_base drone', () => {
  const mission = loadYaml(readFileSync(new URL('../data/la_base.yaml', import.meta.url), 'utf8'));
  const model = buildFlightModel(mission);
  const homeKeys = model.drones.map((drone) => drone.homeWindowKey);
  assert.deepEqual(homeKeys, ['12:10', '8:10', '10:10', '8:8']);
  assert.equal(new Set(homeKeys).size, model.drones.length);
  assert.equal(model.requiredWindows.length, model.drones.length);
  assert.equal(model.landingSpots.length, model.drones.length);
  assert.deepEqual(model.landingSpots.map((spot) => spot.windowKey), homeKeys);
  assert.ok(model.drones.every((drone) => model.homeWindowIndexes.includes(drone.homeWindowIndex)));
  assert.ok(model.drones.every((drone) => drone.landing.every((value, index) => value === drone.takeoff[index])));
  assert.ok(model.drones.every((drone) => drone.segments.some((segment) => segment.phase === 'return to landing')));
  assert.ok(model.drones.every((drone) => drone.segments.some((segment) => segment.phase === 'landing')));
});

test('marks repeated marker signatures as unsafe for guaranteed relocalization', () => {
  const grid = normalizeGrid({
    rows: 3,
    cols: 3,
    num_ids: 1,
    cell_spacing: 0.2,
    marker_size: 0.01,
    window_size: 2,
    grid_origin: [0.2, 0.2, 0],
    grid: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
  });
  assert.equal(grid.windows.length, 4);
  assert.ok(grid.windows.every((window) => window.unique === false));
});

test('does not report above-range flight as supported', () => {
  const mission = {
    takeoff_speed: 10,
    drones: {
      lb1: {
        target: [0, 0, 5, 0],
        waypoints: [[0, 0, 5, 0, 0], [0, 0, 5, 0, 1]],
        delta_t: 0,
        iterations: 1,
        params: { linear: true, relative: false },
      },
    },
  };
  const model = buildFlightModel(mission, gridFixture, { lb1: [0, 0] }, ZERO_CAMERA_OFFSET);
  assert.ok(model.coverage.aboveRangeSamples > 0);
  assert.ok(model.coverage.supportRate < 1);
  assert.equal(model.solution.feasible, false);
});
