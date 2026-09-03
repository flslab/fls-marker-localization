const EPSILON = 1e-9;

const DEFAULT_OPTICS = Object.freeze({
  focalLength: 0.00285,
  sensorWidth: 0.00384,
  sensorHeight: 0.0024,
  usableWidth: 0.8,
  usableHeight: 0.8,
  resolutionWidth: 640,
  resolutionHeight: 400,
  minMarkerPixels: 1,
  minWindowPixels: 30,
});

export const DEFAULT_CAMERA_OFFSET = Object.freeze([0.04, 0, 0]);

export const DEFAULT_FLIGHT_GRID = Object.freeze({
  rows: 20,
  cols: 20,
  num_ids: 8,
  min_k: 3,
  cell_spacing: 0.155,
  marker_size: 0.01,
  window_size: 2,
  working_range: { min_distance: 0.24492187500000004, max_distance: 2.6125 },
  range_model: {
    model: 'pinhole_marker_window_v1',
    focal_length: 0.00285,
    sensor_width: 0.00384,
    sensor_height: 0.0024,
    resolution_width: 640,
    resolution_height: 400,
    usable_width_fraction: 0.8,
    usable_height_fraction: 0.8,
    min_marker_px: 1,
    min_bbox_px: 30,
  },
  short_range: {
    window_size: 2,
    cell_spacing: 0.024,
    marker_size: 0.006,
    working_range: { min_distance: 0.04453125000000001, max_distance: 0.475 },
    tiles: Array.from({ length: 10 }, (_, row) => (
      Array.from({ length: 10 }, (_, col) => ({ i: row * 2, j: col * 2 }))
    )).flat(),
  },
  grid_origin: [1.4725, 1.4725, 0],
  grid: [
    [0, 6, 5, 3, 3, 6, 0, 5, 1, 0, 4, 7, 5, 6, 5, 6, 4, 1, 6, 3],
    [4, 2, 1, 7, 6, 5, 3, 6, 4, 3, 3, 1, 0, 4, 7, 0, 6, 6, 2, 5],
    [1, 6, 5, 2, 0, 7, 3, 7, 5, 6, 6, 1, 2, 3, 3, 0, 4, 1, 5, 5],
    [7, 5, 2, 7, 3, 2, 7, 2, 0, 3, 6, 1, 3, 1, 5, 3, 2, 1, 4, 5],
    [7, 3, 1, 6, 5, 5, 0, 2, 6, 6, 3, 6, 6, 3, 7, 2, 1, 5, 5, 1],
    [6, 1, 6, 0, 6, 4, 6, 5, 3, 5, 2, 6, 4, 3, 4, 4, 0, 1, 1, 0],
    [3, 5, 5, 5, 6, 4, 7, 6, 4, 5, 0, 4, 0, 4, 6, 2, 4, 0, 2, 3],
    [7, 1, 2, 3, 7, 6, 0, 1, 6, 0, 6, 2, 7, 2, 3, 5, 1, 4, 4, 6],
    [7, 5, 3, 3, 3, 6, 2, 1, 2, 0, 0, 0, 6, 5, 5, 3, 5, 1, 7, 4],
    [7, 1, 3, 5, 3, 3, 1, 3, 1, 2, 5, 5, 4, 2, 7, 0, 2, 0, 2, 7],
    [2, 7, 3, 5, 3, 2, 6, 7, 7, 6, 2, 5, 6, 3, 5, 2, 0, 0, 3, 7],
    [1, 3, 5, 1, 5, 2, 6, 4, 4, 1, 3, 6, 0, 6, 3, 5, 5, 3, 2, 5],
    [1, 4, 0, 5, 4, 0, 6, 3, 6, 0, 1, 3, 1, 3, 3, 1, 5, 0, 1, 4],
    [6, 1, 2, 7, 7, 4, 3, 2, 4, 4, 2, 0, 1, 7, 3, 3, 3, 6, 0, 0],
    [2, 3, 5, 3, 3, 7, 1, 4, 4, 3, 6, 2, 4, 2, 4, 4, 6, 4, 2, 0],
    [2, 6, 0, 7, 4, 1, 2, 4, 2, 0, 5, 5, 0, 2, 1, 5, 5, 5, 7, 6],
    [2, 0, 4, 7, 7, 1, 1, 0, 1, 4, 4, 2, 2, 6, 3, 6, 7, 2, 1, 7],
    [5, 2, 4, 6, 7, 2, 7, 7, 1, 1, 4, 0, 2, 3, 7, 7, 1, 7, 0, 5],
    [6, 7, 1, 7, 0, 4, 3, 2, 0, 1, 2, 5, 3, 2, 7, 0, 5, 5, 5, 2],
    [0, 7, 7, 1, 0, 0, 0, 6, 5, 1, 5, 1, 2, 4, 5, 6, 0, 7, 0, 2],
  ],
});

const DRONE_COLORS = Object.freeze([
  '#9df7c7', '#61d9f4', '#ff9f66', '#a889d8', '#ff7777',
  '#ffd166', '#7dd3fc', '#c4b5fd', '#86efac', '#fda4af',
]);

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function finiteNumber(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${label} must be a finite number.`);
  return value;
}

function positiveNumber(value, label) {
  const number = finiteNumber(value, label);
  if (number <= 0) throw new Error(`${label} must be greater than zero.`);
  return number;
}

function vector(value, length, label) {
  if (!Array.isArray(value) || value.length < length) throw new Error(`${label} must contain at least ${length} numbers.`);
  return value.slice(0, length).map((item, index) => finiteNumber(item, `${label}[${index}]`));
}

function withOffset(position, offset) {
  return position.map((value, index) => (index < 3 ? value + offset[index] : value));
}

export function normalizeFlight(raw) {
  if (!isObject(raw)) throw new Error('SFL root must be a mapping.');
  if (!isObject(raw.drones) || !Object.keys(raw.drones).length) throw new Error('SFL must define at least one drone.');
  const warnings = [];
  const takeoffSpeed = positiveNumber(raw.takeoff_speed ?? 0.5, 'takeoff_speed');
  const anchorIds = new Set(Object.values(raw.drones).map((settings) => settings?.relative_anchor?.id).filter(Boolean));
  const entries = Object.entries(raw.drones).filter(([id, settings]) => !(
    anchorIds.has(id)
    && isObject(settings)
    && settings.delta_t === undefined
    && settings.iterations === undefined
    && settings.params === undefined
  ));
  const referenceCount = Object.keys(raw.drones).length - entries.length;
  if (referenceCount) warnings.push(`${referenceCount} relative-anchor reference marker entr${referenceCount === 1 ? 'y was' : 'ies were'} excluded from the flying swarm.`);
  if (!entries.length) throw new Error('SFL does not define any flying drones.');
  const drones = entries.map(([id, settings], index) => {
    if (!isObject(settings)) throw new Error(`drones.${id} must be a mapping.`);
    if (!Array.isArray(settings.target) || settings.target.length < 3 || settings.target.length > 5) {
      throw new Error(`drones.${id}.target must be [x, y, z] with optional yaw and unused fifth value.`);
    }
    const offset = settings.position_offset === undefined
      ? [0, 0, 0]
      : vector(settings.position_offset, 3, `drones.${id}.position_offset`);
    const targetValues = vector(settings.target, 3, `drones.${id}.target`);
    const targetYaw = settings.target.length >= 4 ? finiteNumber(settings.target[3], `drones.${id}.target[3]`) : 0;
    const target = withOffset([...targetValues, targetYaw], offset);
    const deltaT = finiteNumber(settings.delta_t ?? raw.delta_t ?? 0, `drones.${id}.delta_t`);
    if (deltaT < 0) throw new Error(`drones.${id}.delta_t cannot be negative.`);
    const rawWaypoints = settings.waypoints ?? [];
    if (!Array.isArray(rawWaypoints)) throw new Error(`drones.${id}.waypoints must be a list.`);
    const waypoints = rawWaypoints.map((waypoint, waypointIndex) => {
      if (!Array.isArray(waypoint) || (waypoint.length !== 4 && waypoint.length !== 5)) {
        throw new Error(`drones.${id}.waypoints[${waypointIndex}] must be [x, y, z, yaw] or [x, y, z, yaw, dt].`);
      }
      const values = vector(waypoint, 4, `drones.${id}.waypoints[${waypointIndex}]`);
      const duration = waypoint.length === 5
        ? finiteNumber(waypoint[4], `drones.${id}.waypoints[${waypointIndex}][4]`)
        : deltaT;
      if (duration < 0) throw new Error(`drones.${id}.waypoints[${waypointIndex}] duration cannot be negative.`);
      return { position: withOffset(values, offset).slice(0, 3), yaw: values[3], duration };
    });
    const iterations = settings.iterations ?? 1;
    if (!Number.isInteger(iterations) || iterations < 1) throw new Error(`drones.${id}.iterations must be a positive integer.`);
    if (waypoints.length && Math.hypot(
      waypoints[0].position[0] - target[0],
      waypoints[0].position[1] - target[1],
      waypoints[0].position[2] - target[2],
    ) > 1e-6) {
      warnings.push(`${id}: waypoint 0 differs from target; it is treated as the controller's initial animation pose, not a commanded segment.`);
    }
    return {
      id,
      index,
      color: DRONE_COLORS[index % DRONE_COLORS.length],
      target: target.slice(0, 3),
      targetYaw: target[3],
      waypoints,
      deltaT,
      iterations,
      linear: settings.params?.linear === true,
      relative: settings.params?.relative === true,
    };
  });
  return { name: typeof raw.name === 'string' && raw.name.trim() ? raw.name : 'Untitled flight', takeoffSpeed, drones, warnings };
}

function inferWorkingRange(grid, optics, markerSize, windowSize, spacing) {
  const configured = grid.working_range;
  if (isObject(configured)) {
    const min = finiteNumber(configured.min_distance, 'working_range.min_distance');
    const max = finiteNumber(configured.max_distance, 'working_range.max_distance');
    if (min < 0 || max < min) throw new Error('working_range must satisfy 0 <= min_distance <= max_distance.');
    return { min, max, source: 'grid' };
  }
  const bbox = (windowSize - 1) * spacing + markerSize;
  let min;
  let max;
  if (optics.fx) {
    min = Math.max(
      bbox * optics.fx / (optics.resolutionWidth * optics.usableWidth),
      bbox * optics.fy / (optics.resolutionHeight * optics.usableHeight),
    );
    const markerLimit = Math.min(
      markerSize * optics.fx / optics.minMarkerPixels,
      markerSize * optics.fy / optics.minMarkerPixels,
    );
    const bboxLimit = Math.max(bbox * optics.fx, bbox * optics.fy) / optics.minWindowPixels;
    max = Math.min(markerLimit, bboxLimit);
  } else {
    const pixelPitch = optics.sensorWidth / optics.resolutionWidth;
    min = Math.max(
      bbox * optics.focalLength / (optics.sensorWidth * optics.usableWidth),
      bbox * optics.focalLength / (optics.sensorHeight * optics.usableHeight),
    );
    max = Math.min(
      markerSize * optics.focalLength / (optics.minMarkerPixels * pixelPitch),
      bbox * optics.focalLength / (optics.minWindowPixels * pixelPitch),
    );
  }
  if (max < min - EPSILON) throw new Error(`Marker window has no feasible working range: maximum ${max.toFixed(3)} m is below minimum ${min.toFixed(3)} m.`);
  return { min, max, source: 'derived' };
}

export function normalizeGrid(input) {
  const raw = input?.config?.marker_grid ?? input?.marker_grid ?? input;
  if (!isObject(raw)) throw new Error('Marker grid must be a JSON object.');
  const rows = raw.rows;
  const cols = raw.cols;
  const numIds = raw.num_ids;
  const windowSize = raw.window_size ?? 2;
  if (!Number.isInteger(rows) || rows < 1 || !Number.isInteger(cols) || cols < 1) throw new Error('Marker grid rows and cols must be positive integers.');
  if (!Number.isInteger(numIds) || numIds < 1) throw new Error('Marker grid num_ids must be a positive integer.');
  if (!Number.isInteger(windowSize) || windowSize < 2 || windowSize > Math.min(rows, cols)) throw new Error('Marker grid window_size must be at least 2 and fit inside the grid.');
  const spacing = positiveNumber(raw.cell_spacing, 'cell_spacing');
  const markerSize = positiveNumber(raw.marker_size ?? 0.01, 'marker_size');
  const origin = raw.grid_origin === undefined
    ? [(rows - 1) * spacing / 2, (cols - 1) * spacing / 2, 0]
    : vector(raw.grid_origin, 3, 'grid_origin');
  if (!Array.isArray(raw.grid) || raw.grid.length !== rows || raw.grid.some((row) => !Array.isArray(row) || row.length !== cols)) {
    throw new Error('Marker grid must include a complete grid ID matrix matching rows and cols.');
  }
  const ids = raw.grid.map((row, rowIndex) => row.map((id, colIndex) => {
    const number = finiteNumber(id, `grid[${rowIndex}][${colIndex}]`);
    if (!Number.isInteger(number) || number < 0 || number >= numIds) throw new Error(`grid[${rowIndex}][${colIndex}] must be an integer in [0, num_ids).`);
    return number;
  }));
  const rangeModel = isObject(raw.range_model) ? raw.range_model : {};
  const imageResolution = raw.image_resolution_pixels;
  const focalPixels = raw.focal_length_pixels;
  let optics;
  if (Array.isArray(imageResolution) && Array.isArray(focalPixels) && imageResolution.length >= 2 && focalPixels.length >= 2) {
    const width = positiveNumber(imageResolution[0], 'image_resolution_pixels[0]');
    const height = positiveNumber(imageResolution[1], 'image_resolution_pixels[1]');
    const fx = positiveNumber(focalPixels[0], 'focal_length_pixels[0]');
    const fy = positiveNumber(focalPixels[1], 'focal_length_pixels[1]');
    const usableWidth = positiveNumber(rangeModel.usable_width_fraction ?? DEFAULT_OPTICS.usableWidth, 'range_model.usable_width_fraction');
    const usableHeight = positiveNumber(rangeModel.usable_height_fraction ?? DEFAULT_OPTICS.usableHeight, 'range_model.usable_height_fraction');
    if (usableWidth > 1 || usableHeight > 1) throw new Error('Usable sensor fractions cannot exceed one.');
    optics = {
      fx,
      fy,
      usableWidth,
      usableHeight,
      resolutionWidth: width,
      resolutionHeight: height,
      minMarkerPixels: positiveNumber(rangeModel.min_marker_px ?? DEFAULT_OPTICS.minMarkerPixels, 'range_model.min_marker_px'),
      minWindowPixels: positiveNumber(rangeModel.min_bbox_px ?? DEFAULT_OPTICS.minWindowPixels, 'range_model.min_bbox_px'),
      halfXFactor: (height / (2 * fy)) * usableHeight,
      halfYFactor: (width / (2 * fx)) * usableWidth,
      source: 'pixel intrinsics',
    };
  } else {
    const focalLength = positiveNumber(rangeModel.focal_length ?? DEFAULT_OPTICS.focalLength, 'range_model.focal_length');
    const sensorWidth = positiveNumber(rangeModel.sensor_width ?? DEFAULT_OPTICS.sensorWidth, 'range_model.sensor_width');
    const sensorHeight = positiveNumber(rangeModel.sensor_height ?? DEFAULT_OPTICS.sensorHeight, 'range_model.sensor_height');
    const usableWidth = positiveNumber(rangeModel.usable_width_fraction ?? DEFAULT_OPTICS.usableWidth, 'range_model.usable_width_fraction');
    const usableHeight = positiveNumber(rangeModel.usable_height_fraction ?? DEFAULT_OPTICS.usableHeight, 'range_model.usable_height_fraction');
    if (usableWidth > 1 || usableHeight > 1) throw new Error('Usable sensor fractions cannot exceed one.');
    optics = {
      focalLength,
      sensorWidth,
      sensorHeight,
      usableWidth,
      usableHeight,
      resolutionWidth: positiveNumber(rangeModel.resolution_width ?? DEFAULT_OPTICS.resolutionWidth, 'range_model.resolution_width'),
      resolutionHeight: positiveNumber(rangeModel.resolution_height ?? DEFAULT_OPTICS.resolutionHeight, 'range_model.resolution_height'),
      minMarkerPixels: positiveNumber(rangeModel.min_marker_px ?? DEFAULT_OPTICS.minMarkerPixels, 'range_model.min_marker_px'),
      minWindowPixels: positiveNumber(rangeModel.min_bbox_px ?? DEFAULT_OPTICS.minWindowPixels, 'range_model.min_bbox_px'),
      halfXFactor: sensorHeight * usableHeight / (2 * focalLength),
      halfYFactor: sensorWidth * usableWidth / (2 * focalLength),
      source: Object.keys(rangeModel).length ? 'physical intrinsics' : 'default physical intrinsics',
    };
  }
  const workingRange = inferWorkingRange(raw, optics, markerSize, windowSize, spacing);
  const cells = [];
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      cells.push({ key: `${row}:${col}`, row, col, id: ids?.[row][col] ?? null, position: [origin[0] - row * spacing, origin[1] - col * spacing, origin[2]] });
    }
  }
  const grid = { rows, cols, numIds, windowSize, spacing, markerSize, origin, ids, optics, workingRange, cells, shortRange: raw.short_range ?? null };
  grid.windows = buildWindows(grid);
  const halfMarker = markerSize / 2;
  grid.bounds = {
    minX: origin[0] - (rows - 1) * spacing - halfMarker,
    maxX: origin[0] + halfMarker,
    minY: origin[1] - (cols - 1) * spacing - halfMarker,
    maxY: origin[1] + halfMarker,
  };
  return grid;
}

function buildWindows(grid) {
  const windows = [];
  const signatureCounts = new Map();
  for (let row = 0; row <= grid.rows - grid.windowSize; row += 1) {
    for (let col = 0; col <= grid.cols - grid.windowSize; col += 1) {
      const cells = [];
      const signature = [];
      for (let localRow = 0; localRow < grid.windowSize; localRow += 1) {
        for (let localCol = 0; localCol < grid.windowSize; localCol += 1) {
          const cell = grid.cells[(row + localRow) * grid.cols + col + localCol];
          cells.push(cell);
          if (grid.ids) signature.push(cell.id);
        }
      }
      const signatureKey = grid.ids ? signature.join(',') : null;
      if (signatureKey !== null) signatureCounts.set(signatureKey, (signatureCounts.get(signatureKey) ?? 0) + 1);
      const highX = grid.origin[0] - row * grid.spacing + grid.markerSize / 2;
      const lowX = grid.origin[0] - (row + grid.windowSize - 1) * grid.spacing - grid.markerSize / 2;
      const highY = grid.origin[1] - col * grid.spacing + grid.markerSize / 2;
      const lowY = grid.origin[1] - (col + grid.windowSize - 1) * grid.spacing - grid.markerSize / 2;
      windows.push({
        index: windows.length,
        key: `${row}:${col}`,
        row,
        col,
        cells,
        signature: grid.ids ? signature : null,
        signatureKey,
        bounds: { minX: lowX, maxX: highX, minY: lowY, maxY: highY },
        center: [(lowX + highX) / 2, (lowY + highY) / 2, grid.origin[2]],
      });
    }
  }
  return windows.map((window) => ({ ...window, unique: window.signatureKey === null ? null : signatureCounts.get(window.signatureKey) === 1 }));
}

function takeoffCandidates(grid) {
  const tiles = Array.isArray(grid.shortRange?.tiles) ? grid.shortRange.tiles : [];
  if (tiles.length) {
    const candidates = tiles.map((tile) => {
      const markers = Array.isArray(tile.markers) ? tile.markers : [];
      const x = markers.length ? markers.reduce((sum, marker) => sum + Number(marker.global_x), 0) / markers.length : grid.origin[0] - (tile.i + (grid.windowSize - 1) / 2) * grid.spacing;
      const y = markers.length ? markers.reduce((sum, marker) => sum + Number(marker.global_y), 0) / markers.length : grid.origin[1] - (tile.j + (grid.windowSize - 1) / 2) * grid.spacing;
      const window = grid.windows.find((candidate) => candidate.row === tile.i && candidate.col === tile.j);
      return { row: tile.i, col: tile.j, position: [x, y], window, tile };
    }).filter((candidate) => candidate.window && candidate.position.every(Number.isFinite));
    return [...new Map(candidates.map((candidate) => [candidate.window.index, candidate])).values()];
  }
  const rowStarts = [];
  const colStarts = [];
  for (let row = 0; row <= grid.rows - grid.windowSize; row += grid.windowSize) rowStarts.push(row);
  for (let col = 0; col <= grid.cols - grid.windowSize; col += grid.windowSize) colStarts.push(col);
  if (rowStarts.at(-1) !== grid.rows - grid.windowSize) rowStarts.push(grid.rows - grid.windowSize);
  if (colStarts.at(-1) !== grid.cols - grid.windowSize) colStarts.push(grid.cols - grid.windowSize);
  return rowStarts.flatMap((row) => colStarts.map((col) => {
    const window = grid.windows[row * (grid.cols - grid.windowSize + 1) + col];
    return { row, col, position: window.center.slice(0, 2), window, tile: null };
  }));
}

export function allocateTakeoffs(drones, grid, overrides = {}, cameraOffset = [0, 0, 0]) {
  const candidates = takeoffCandidates(grid).map((candidate) => ({
    ...candidate,
    bodyPosition: [candidate.position[0] - cameraOffset[0], candidate.position[1] - cameraOffset[1]],
  }));
  const assignment = new Map();
  const usedCandidates = new Set();
  const overridePositions = [];
  for (const drone of drones) {
    const override = overrides[drone.id];
    if (!override) continue;
    const position = vector(override, 2, `takeoff override for ${drone.id}`);
    if (overridePositions.some((other) => Math.hypot(other[0] - position[0], other[1] - position[1]) <= EPSILON)) {
      throw new Error('Takeoff overrides must assign every drone a distinct XY position.');
    }
    overridePositions.push(position);
  }
  const overridePairs = [];
  drones.forEach((drone) => {
    if (!overrides[drone.id]) return;
    const position = vector(overrides[drone.id], 2, `takeoff override for ${drone.id}`);
    candidates.forEach((candidate, candidateIndex) => overridePairs.push({
      drone,
      position,
      candidate,
      candidateIndex,
      distance: Math.hypot(candidate.bodyPosition[0] - position[0], candidate.bodyPosition[1] - position[1]),
    }));
  });
  overridePairs.sort((a, b) => a.distance - b.distance || a.drone.index - b.drone.index || a.candidateIndex - b.candidateIndex);
  for (const pair of overridePairs) {
    if (assignment.has(pair.drone.id) || usedCandidates.has(pair.candidateIndex)) continue;
    assignment.set(pair.drone.id, {
      ...pair.candidate,
      position: pair.position,
      alignmentError: pair.distance,
    });
    usedCandidates.add(pair.candidateIndex);
  }
  const pairs = [];
  drones.forEach((drone) => {
    if (assignment.has(drone.id)) return;
    candidates.forEach((candidate, candidateIndex) => pairs.push({
      drone,
      candidateIndex,
      distance: Math.hypot(candidate.bodyPosition[0] - drone.target[0], candidate.bodyPosition[1] - drone.target[1]),
    }));
  });
  pairs.sort((a, b) => a.distance - b.distance || a.drone.index - b.drone.index || a.candidateIndex - b.candidateIndex);
  for (const pair of pairs) {
    if (assignment.has(pair.drone.id) || usedCandidates.has(pair.candidateIndex)) continue;
    const candidate = candidates[pair.candidateIndex];
    assignment.set(pair.drone.id, {
      ...candidate,
      position: candidate.bodyPosition.slice(),
      alignmentError: 0,
    });
    usedCandidates.add(pair.candidateIndex);
  }
  if (assignment.size !== drones.length) throw new Error('The marker grid does not contain enough distinct launch/landing windows for this swarm.');
  const shortRangeTolerance = Math.max(1e-6, Number(grid.shortRange?.cell_spacing ?? 0) * 0.1);
  for (const [droneId, home] of assignment) {
    if (home.tile && home.alignmentError > shortRangeTolerance) {
      throw new Error(`Takeoff override for ${droneId} must align with a short-range tile centre.`);
    }
  }
  return assignment;
}

function addSegment(segments, start, end, duration, phase, linear, cursor) {
  const segment = { start, end, duration, phase, linear, t0: cursor, t1: cursor + duration };
  segments.push(segment);
  return segment.t1;
}

function buildSegments(mission, takeoffs, planeZ) {
  const drones = mission.drones.map((drone) => {
    const home = takeoffs.get(drone.id);
    const takeoff = home.position;
    const ground = [takeoff[0], takeoff[1], planeZ, 0];
    const airborne = [takeoff[0], takeoff[1], drone.target[2], 0];
    const target = [...drone.target, drone.targetYaw];
    const segments = [];
    let cursor = 0;
    cursor = addSegment(segments, ground, airborne, Math.abs(drone.target[2] - planeZ) / mission.takeoffSpeed, 'takeoff', false, cursor);
    cursor = addSegment(segments, airborne, target, Math.hypot(target[0] - airborne[0], target[1] - airborne[1]) * 6, 'to target', false, cursor);
    return {
      ...drone,
      takeoff,
      landing: takeoff.slice(),
      homeWindowIndex: home.window.index,
      homeWindowKey: home.window.key,
      homeTileKey: home.tile ? `${home.tile.i}:${home.tile.j}` : null,
      homeAlignmentError: home.alignmentError,
      segments,
      preflightDuration: cursor,
      current: target,
    };
  });
  const waypointStart = Math.max(...drones.map((drone) => drone.preflightDuration));
  for (const drone of drones) {
    let cursor = drone.preflightDuration;
    if (cursor < waypointStart - EPSILON) cursor = addSegment(drone.segments, drone.current, drone.current, waypointStart - cursor, 'synchronize', true, cursor);
    let current = drone.current;
    if (drone.waypoints.length <= 1) {
      const duration = drone.waypoints[0]?.duration ?? drone.deltaT;
      if (duration > 0) cursor = addSegment(drone.segments, current, current, duration, 'waypoint hold', drone.linear, cursor);
    } else {
      for (let iteration = 0; iteration < drone.iterations; iteration += 1) {
        if (drone.iterations > 1) cursor = addSegment(drone.segments, current, current, 0.1, 'loop sync', true, cursor);
        for (let index = 1; index < drone.waypoints.length; index += 1) {
          const waypoint = drone.waypoints[index];
          const destination = drone.relative
            ? [current[0] + waypoint.position[0], current[1] + waypoint.position[1], current[2] + waypoint.position[2], current[3] + waypoint.yaw]
            : [...waypoint.position, waypoint.yaw];
          if (destination[2] <= planeZ + EPSILON) {
            throw new Error(`drones.${drone.id}.waypoints[${index}][2] must stay above the marker plane at z=${planeZ}.`);
          }
          cursor = addSegment(drone.segments, current, destination, waypoint.duration, `waypoint ${index}${drone.iterations > 1 ? ` · loop ${iteration + 1}` : ''}`, drone.linear, cursor);
          current = destination;
        }
      }
    }
    const landingPose = [drone.landing[0], drone.landing[1], current[2], 0];
    const returnDistance = Math.hypot(landingPose[0] - current[0], landingPose[1] - current[1]);
    cursor = addSegment(drone.segments, current, landingPose, returnDistance * 2 + 0.1, 'return to landing', false, cursor);
    const ground = [landingPose[0], landingPose[1], planeZ, 0];
    cursor = addSegment(drone.segments, landingPose, ground, Math.abs(drone.target[2] - planeZ) / mission.takeoffSpeed, 'landing', false, cursor);
    drone.current = ground;
    drone.duration = cursor;
    drone.path = [drone.segments[0].start, ...drone.segments.map((segment) => segment.end)];
  }
  return drones;
}

function interpolateYaw(start, end, amount) {
  const delta = Math.atan2(Math.sin(end - start), Math.cos(end - start));
  return start + delta * amount;
}

function trajectoryAmount(rawAmount, linear) {
  return linear
    ? rawAmount
    : rawAmount ** 4 * (35 - 84 * rawAmount + 70 * rawAmount ** 2 - 20 * rawAmount ** 3);
}

export function stateAt(drone, time) {
  let last = drone.segments[0]?.start ?? [...drone.target, drone.targetYaw];
  for (const segment of drone.segments) {
    if (segment.duration <= EPSILON) {
      if (time + EPSILON >= segment.t1) last = segment.end;
      continue;
    }
    if (time <= segment.t1 + EPSILON) {
      const rawAmount = Math.max(0, Math.min(1, (time - segment.t0) / segment.duration));
      const amount = trajectoryAmount(rawAmount, segment.linear);
      return {
        position: segment.start.slice(0, 3).map((value, index) => value + (segment.end[index] - value) * amount),
        yaw: interpolateYaw(segment.start[3], segment.end[3], amount),
        phase: segment.phase,
      };
    }
    last = segment.end;
  }
  return { position: last.slice(0, 3), yaw: last[3], phase: 'complete' };
}

export function footprintForPose(position, yaw, grid) {
  const height = Math.max(0, position[2] - grid.origin[2]);
  const halfX = height * grid.optics.halfXFactor;
  const halfY = height * grid.optics.halfYFactor;
  const cos = Math.cos(yaw);
  const sin = Math.sin(yaw);
  const corners = [[-halfX, -halfY], [halfX, -halfY], [halfX, halfY], [-halfX, halfY]].map(([forward, left]) => [
    position[0] + forward * cos - left * sin,
    position[1] + forward * sin + left * cos,
  ]);
  return { center: position.slice(0, 2), yaw, halfX, halfY, corners };
}

function footprintContains(footprint, x, y) {
  const dx = x - footprint.center[0];
  const dy = y - footprint.center[1];
  const cos = Math.cos(footprint.yaw);
  const sin = Math.sin(footprint.yaw);
  const forward = dx * cos + dy * sin;
  const left = -dx * sin + dy * cos;
  return Math.abs(forward) <= footprint.halfX + EPSILON && Math.abs(left) <= footprint.halfY + EPSILON;
}

function windowFits(window, footprint) {
  return windowMargin(window, footprint) >= -EPSILON;
}

function windowMargin(window, footprint) {
  const { minX, maxX, minY, maxY } = window.bounds;
  let margin = Infinity;
  for (const [x, y] of [[minX, minY], [minX, maxY], [maxX, minY], [maxX, maxY]]) {
    const dx = x - footprint.center[0];
    const dy = y - footprint.center[1];
    const cos = Math.cos(footprint.yaw);
    const sin = Math.sin(footprint.yaw);
    const forward = dx * cos + dy * sin;
    const left = -dx * sin + dy * cos;
    margin = Math.min(margin, footprint.halfX - Math.abs(forward), footprint.halfY - Math.abs(left));
  }
  return margin;
}

function maxAbsLocalProjection(dx, dy, startYaw, endYaw, axis) {
  const radius = Math.hypot(dx, dy);
  if (radius <= EPSILON) return 0;
  const yawDelta = Math.atan2(Math.sin(endYaw - startYaw), Math.cos(endYaw - startYaw));
  const finishYaw = startYaw + yawDelta;
  const project = axis === 'forward'
    ? (yaw) => Math.abs(dx * Math.cos(yaw) + dy * Math.sin(yaw))
    : (yaw) => Math.abs(-dx * Math.sin(yaw) + dy * Math.cos(yaw));
  const low = Math.min(startYaw, finishYaw);
  const high = Math.max(startYaw, finishYaw);
  const vectorYaw = Math.atan2(dy, dx);
  const criticalBase = axis === 'forward' ? vectorYaw : vectorYaw - Math.PI / 2;
  const critical = criticalBase + Math.ceil((low - criticalBase) / Math.PI) * Math.PI;
  return critical <= high + EPSILON ? radius : Math.max(project(startYaw), project(finishYaw));
}

function intervalMotionAcross(start, middle, end, cameraOffsetRadius) {
  const endpointMotion = [start, end].map((endpoint) => {
    const dx = endpoint.position[0] - middle.position[0];
    const dy = endpoint.position[1] - middle.position[1];
    const yaw = Math.abs(Math.atan2(
      Math.sin(endpoint.yaw - middle.yaw),
      Math.cos(endpoint.yaw - middle.yaw),
    ));
    const offsetMove = 2 * Math.sin(yaw / 2) * cameraOffsetRadius;
    return {
      forward: maxAbsLocalProjection(dx, dy, middle.yaw, endpoint.yaw, 'forward') + offsetMove,
      left: maxAbsLocalProjection(dx, dy, middle.yaw, endpoint.yaw, 'left') + offsetMove,
      yaw,
      halfX: Math.abs(endpoint.footprint.halfX - middle.footprint.halfX),
      halfY: Math.abs(endpoint.footprint.halfY - middle.footprint.halfY),
      center: Math.hypot(dx, dy) + offsetMove,
    };
  });
  return {
    endpointMotion,
    centerReach: Math.max(...endpointMotion.map((motion) => motion.center)),
    footprintRadius: Math.max(...[start, middle, end].map((state) => Math.hypot(state.footprint.halfX, state.footprint.halfY))),
  };
}

function intervalGeometryReach(intervalMotion) {
  return Math.max(...intervalMotion.endpointMotion.map((motion) => (
    motion.center
    + 2 * Math.sin(motion.yaw / 2) * intervalMotion.footprintRadius
    + Math.hypot(motion.halfX, motion.halfY)
  )));
}

function windowStatusAcross(window, middle, intervalMotion) {
  const { endpointMotion } = intervalMotion;
  if (Math.hypot(
    window.center[0] - middle.footprint.center[0],
    window.center[1] - middle.footprint.center[1],
  ) > intervalMotion.centerReach + intervalMotion.footprintRadius + EPSILON) return 'invisible';
  const { minX, maxX, minY, maxY } = window.bounds;
  const cos = Math.cos(middle.yaw);
  const sin = Math.sin(middle.yaw);
  let alwaysVisible = true;
  for (const [x, y] of [[minX, minY], [minX, maxY], [maxX, minY], [maxX, maxY]]) {
    const dx = x - middle.footprint.center[0];
    const dy = y - middle.footprint.center[1];
    const forward = dx * cos + dy * sin;
    const left = -dx * sin + dy * cos;
    const radius = Math.hypot(dx, dy);
    const forwardMove = Math.max(...endpointMotion.map((motion) => motion.forward + 2 * Math.sin(motion.yaw / 2) * radius));
    const leftMove = Math.max(...endpointMotion.map((motion) => motion.left + 2 * Math.sin(motion.yaw / 2) * radius));
    const halfXMove = Math.max(...endpointMotion.map((motion) => motion.halfX));
    const halfYMove = Math.max(...endpointMotion.map((motion) => motion.halfY));
    const lowerX = middle.footprint.halfX - halfXMove - Math.abs(forward) - forwardMove;
    const lowerY = middle.footprint.halfY - halfYMove - Math.abs(left) - leftMove;
    const upperX = middle.footprint.halfX + halfXMove - Math.max(0, Math.abs(forward) - forwardMove);
    const upperY = middle.footprint.halfY + halfYMove - Math.max(0, Math.abs(left) - leftMove);
    if (upperX < -EPSILON || upperY < -EPSILON) return 'invisible';
    if (lowerX < -EPSILON || lowerY < -EPSILON) alwaysVisible = false;
  }
  return alwaysVisible ? 'visible' : 'uncertain';
}

function cellStatusAcross(cell, middle, intervalMotion) {
  const dx = cell.position[0] - middle.footprint.center[0];
  const dy = cell.position[1] - middle.footprint.center[1];
  if (Math.hypot(dx, dy) > intervalMotion.centerReach + intervalMotion.footprintRadius + EPSILON) return 'invisible';
  const cos = Math.cos(middle.yaw);
  const sin = Math.sin(middle.yaw);
  const forward = dx * cos + dy * sin;
  const left = -dx * sin + dy * cos;
  const radius = Math.hypot(dx, dy);
  const forwardMove = Math.max(...intervalMotion.endpointMotion.map((motion) => motion.forward + 2 * Math.sin(motion.yaw / 2) * radius));
  const leftMove = Math.max(...intervalMotion.endpointMotion.map((motion) => motion.left + 2 * Math.sin(motion.yaw / 2) * radius));
  const halfXMove = Math.max(...intervalMotion.endpointMotion.map((motion) => motion.halfX));
  const halfYMove = Math.max(...intervalMotion.endpointMotion.map((motion) => motion.halfY));
  const minForward = Math.max(0, Math.abs(forward) - forwardMove);
  const minLeft = Math.max(0, Math.abs(left) - leftMove);
  if (minForward > middle.footprint.halfX + halfXMove + EPSILON
    || minLeft > middle.footprint.halfY + halfYMove + EPSILON) return 'invisible';
  if (Math.abs(forward) + forwardMove <= middle.footprint.halfX - halfXMove + EPSILON
    && Math.abs(left) + leftMove <= middle.footprint.halfY - halfYMove + EPSILON) return 'visible';
  return 'uncertain';
}

function bitCount(value) {
  let count = 0;
  for (let bits = value; bits; bits &= bits - 1n) count += 1;
  return count;
}

function removeRedundant(selected, masks, universe) {
  const kept = selected.slice();
  for (let index = kept.length - 1; index >= 0; index -= 1) {
    let coverage = 0n;
    kept.forEach((candidate, candidateIndex) => { if (candidateIndex !== index) coverage |= masks[candidate]; });
    if (coverage === universe) kept.splice(index, 1);
  }
  return kept;
}

export function solveMinimumWindowCover(requirements, candidateCount, { maxNodes = 100000, maxMs = 120, preselected = [] } = {}) {
  const allRequirements = [...new Map(requirements.map((requirement) => {
    const normalized = [...new Set(requirement)].filter((candidate) => Number.isInteger(candidate) && candidate >= 0 && candidate < candidateCount).sort((a, b) => a - b);
    return [normalized.join(','), normalized];
  })).values()].filter((requirement) => requirement.length);
  const required = [...new Set(preselected)]
    .filter((candidate) => Number.isInteger(candidate) && candidate >= 0 && candidate < candidateCount)
    .sort((a, b) => a - b);
  const requiredSet = new Set(required);
  const uniqueRequirements = allRequirements.filter((requirement) => !requirement.some((candidate) => requiredSet.has(candidate)));
  if (!uniqueRequirements.length) {
    return {
      selected: required,
      added: [],
      preselected: required,
      optimal: true,
      nodes: 0,
      requirementCount: allRequirements.length,
      residualRequirementCount: 0,
    };
  }
  const universe = (1n << BigInt(uniqueRequirements.length)) - 1n;
  const masks = Array(candidateCount).fill(0n);
  uniqueRequirements.forEach((requirement, requirementIndex) => {
    const bit = 1n << BigInt(requirementIndex);
    requirement.forEach((candidate) => { masks[candidate] |= bit; });
  });
  let candidates = masks.map((mask, index) => ({ index, mask })).filter((candidate) => candidate.mask);
  candidates = candidates.filter((candidate, index) => !candidates.some((other, otherIndex) => (
    otherIndex !== index
    && (candidate.mask | other.mask) === other.mask
    && (candidate.mask !== other.mask || other.index < candidate.index)
  )));
  let covered = 0n;
  let greedy = [];
  while (covered !== universe) {
    const best = candidates.reduce((winner, candidate) => {
      const gain = bitCount(candidate.mask & ~covered);
      return !winner || gain > winner.gain || (gain === winner.gain && candidate.index < winner.candidate.index)
        ? { candidate, gain }
        : winner;
    }, null);
    if (!best?.gain) break;
    greedy.push(best.candidate.index);
    covered |= best.candidate.mask;
  }
  let best = removeRedundant(greedy, masks, universe);
  const started = Date.now();
  let nodes = 0;
  let aborted = false;
  const memo = new Map();
  const visit = (currentCoverage, selected) => {
    nodes += 1;
    if (nodes > maxNodes || Date.now() - started > maxMs) { aborted = true; return; }
    if (currentCoverage === universe) {
      if (selected.length < best.length) best = selected.slice();
      return;
    }
    if (selected.length >= best.length) return;
    const seenDepth = memo.get(currentCoverage);
    if (seenDepth !== undefined && seenDepth <= selected.length) return;
    memo.set(currentCoverage, selected.length);
    const uncovered = universe & ~currentCoverage;
    let maxGain = 0;
    for (const candidate of candidates) maxGain = Math.max(maxGain, bitCount(candidate.mask & uncovered));
    if (!maxGain || selected.length + Math.ceil(bitCount(uncovered) / maxGain) >= best.length) return;
    let options = null;
    for (let requirementIndex = 0; requirementIndex < uniqueRequirements.length; requirementIndex += 1) {
      const bit = 1n << BigInt(requirementIndex);
      if (!(uncovered & bit)) continue;
      const nextOptions = candidates.filter((candidate) => candidate.mask & bit);
      if (!options || nextOptions.length < options.length) options = nextOptions;
    }
    options.sort((a, b) => bitCount(b.mask & uncovered) - bitCount(a.mask & uncovered) || a.index - b.index);
    for (const candidate of options) {
      visit(currentCoverage | candidate.mask, [...selected, candidate.index]);
      if (aborted) return;
    }
  };
  visit(0n, []);
  const added = best.sort((a, b) => a - b);
  return {
    selected: [...new Set([...required, ...added])].sort((a, b) => a - b),
    added,
    preselected: required,
    optimal: !aborted,
    nodes,
    requirementCount: allRequirements.length,
    residualRequirementCount: uniqueRequirements.length,
  };
}

function evaluatePoseGeometry(state, grid, cameraOffset) {
  const cos = Math.cos(state.yaw);
  const sin = Math.sin(state.yaw);
  const cameraPosition = [
    state.position[0] + cameraOffset[0] * cos - cameraOffset[1] * sin,
    state.position[1] + cameraOffset[0] * sin + cameraOffset[1] * cos,
    state.position[2] + cameraOffset[2],
  ];
  const footprint = footprintForPose(cameraPosition, state.yaw, grid);
  const height = cameraPosition[2] - grid.origin[2];
  const rangeStatus = height < grid.workingRange.min - EPSILON ? 'below main range' : (height > grid.workingRange.max + EPSILON ? 'above main range' : 'main range');
  return { ...state, cameraPosition, footprint, rangeStatus };
}

function evaluatePose(state, grid, cameraOffset) {
  const geometry = evaluatePoseGeometry(state, grid, cameraOffset);
  const observedCellKeys = grid.cells.filter((cell) => footprintContains(geometry.footprint, cell.position[0], cell.position[1])).map((cell) => cell.key);
  const visibleWindowIndexes = geometry.rangeStatus === 'main range'
    ? grid.windows.filter((window) => window.unique !== false && windowFits(window, geometry.footprint)).map((window) => window.index)
    : [];
  return { ...geometry, observedCellKeys, visibleWindowIndexes };
}

function sampleTimes(drones, duration, grid, cameraOffset) {
  const maxSamples = 6000;
  const times = new Set([0, duration]);
  const certificationTimes = new Set([0, duration]);
  const segments = [];
  for (const drone of drones) {
    for (const segment of drone.segments) {
      times.add(segment.t0);
      times.add(segment.t1);
      certificationTimes.add(segment.t0);
      certificationTimes.add(segment.t1);
      segments.push(segment);
    }
  }
  const rangeHeights = [
    grid.origin[2] + grid.workingRange.min - cameraOffset[2],
    grid.origin[2] + grid.workingRange.max - cameraOffset[2],
  ];
  for (const segment of segments) {
    if (segment.duration <= EPSILON || Math.abs(segment.end[2] - segment.start[2]) <= EPSILON) continue;
    for (const height of rangeHeights) {
      if ((segment.start[2] - height) * (segment.end[2] - height) >= 0) continue;
      let low = 0;
      let high = 1;
      const ascending = segment.end[2] > segment.start[2];
      for (let iteration = 0; iteration < 48; iteration += 1) {
        const middle = (low + high) / 2;
        const z = segment.start[2] + (segment.end[2] - segment.start[2]) * trajectoryAmount(middle, segment.linear);
        if ((z < height) === ascending) low = middle;
        else high = middle;
      }
      const crossingTime = segment.t0 + segment.duration * (low + high) / 2;
      times.add(crossingTime);
      certificationTimes.add(crossingTime);
    }
  }
  if (times.size > maxSamples) throw new Error(`Flight has more than ${maxSamples} segment boundaries; split it into smaller missions.`);
  const rate = duration > EPSILON ? Math.min(12, 3000 / duration) : 1;
  for (let index = 0; index <= Math.ceil(duration * rate); index += 1) times.add(Math.min(duration, index / rate));
  for (const segment of segments) {
    if (segment.duration <= EPSILON) continue;
    const distance = Math.hypot(
      segment.end[0] - segment.start[0],
      segment.end[1] - segment.start[1],
      segment.end[2] - segment.start[2],
    );
    const yawDistance = Math.abs(Math.atan2(
      Math.sin(segment.end[3] - segment.start[3]),
      Math.cos(segment.end[3] - segment.start[3]),
    ));
    const maxHeight = Math.max(segment.start[2], segment.end[2]) + cameraOffset[2] - grid.origin[2];
    const footprintRadius = Math.max(0, maxHeight) * Math.hypot(grid.optics.halfXFactor, grid.optics.halfYFactor);
    const cameraOffsetRadius = Math.hypot(cameraOffset[0], cameraOffset[1]);
    const easingFactor = segment.linear ? 1 : 2.2;
    const sweptDistance = easingFactor * (distance + yawDistance * (footprintRadius + cameraOffsetRadius));
    const steps = Math.max(
      1,
      Math.ceil(sweptDistance / (grid.spacing * 0.05)),
      Math.ceil(yawDistance / (Math.PI / 36)),
    );
    if (steps > maxSamples) throw new Error(`One flight leg needs more than ${maxSamples} coverage samples; split the mission or use a coarser grid.`);
    for (let step = 1; step < steps; step += 1) times.add(segment.t0 + segment.duration * step / steps);
    if (times.size > maxSamples) throw new Error(`Flight needs more than ${maxSamples} adaptive coverage samples; split it into smaller missions.`);
  }
  return {
    rate,
    times: [...times].sort((a, b) => a - b),
    certificationTimes: [...certificationTimes].sort((a, b) => a - b),
  };
}

export function buildFlightModel(rawMission, rawGrid = DEFAULT_FLIGHT_GRID, takeoffOverrides = {}, rawCameraOffset = DEFAULT_CAMERA_OFFSET) {
  const mission = normalizeFlight(rawMission);
  const grid = normalizeGrid(rawGrid);
  const cameraOffset = vector(rawCameraOffset, 3, 'camera offset');
  for (const drone of mission.drones) {
    if (drone.target[2] <= grid.origin[2] + EPSILON) {
      throw new Error(`drones.${drone.id}.target[2] must be above the marker plane at z=${grid.origin[2]}.`);
    }
  }
  const takeoffs = allocateTakeoffs(mission.drones, grid, takeoffOverrides, cameraOffset);
  const drones = buildSegments(mission, takeoffs, grid.origin[2]);
  const duration = Math.max(...drones.map((drone) => drone.duration), 0);
  const sampled = sampleTimes(drones, duration, grid, cameraOffset);
  const poseCache = new Map();
  const evaluate = (state) => {
    const key = [...state.position, state.yaw].join(':');
    if (!poseCache.has(key)) poseCache.set(key, evaluatePose(state, grid, cameraOffset));
    return { ...poseCache.get(key), phase: state.phase };
  };
  const evaluateGeometry = (state) => evaluatePoseGeometry(state, grid, cameraOffset);
  const frames = sampled.times.map((time) => ({
    time,
    drones: drones.map((drone) => ({ id: drone.id, color: drone.color, ...evaluate(stateAt(drone, time)) })),
  }));
  const requirements = new Map();
  const observedCells = new Set();
  const observedCellTimes = new Map();
  const markObserved = (key, time) => {
    observedCells.add(key);
    observedCellTimes.set(key, Math.min(time, observedCellTimes.get(key) ?? Infinity));
  };
  const coverage = {
    belowRangeSamples: 0,
    unsupportedBelowRangeSamples: 0,
    aboveRangeSamples: 0,
    eligibleSamples: 0,
    supportedSamples: 0,
    unsupportedSamples: 0,
    eligibleIntervals: 0,
    supportedIntervals: 0,
    unsupportedIntervals: 0,
    boundaryIntervals: 0,
    shortRangeTakeoffDuration: 0,
    shortRangeLandingDuration: 0,
    unsupportedBelowRangeDuration: 0,
    aboveRangeDuration: 0,
    eligibleDuration: 0,
    supportedDuration: 0,
    intervalGaps: [],
  };
  const recordRequirement = (indexes) => requirements.set(indexes.join(','), indexes);
  for (const frame of frames) {
    for (const state of frame.drones) {
      state.observedCellKeys.forEach((key) => markObserved(key, frame.time));
      if (state.phase === 'takeoff' || state.phase === 'landing' || state.phase === 'complete') {
        if (state.rangeStatus === 'below main range') coverage.belowRangeSamples += 1;
        continue;
      }
      if (state.rangeStatus === 'below main range') {
        coverage.unsupportedBelowRangeSamples += 1;
        continue;
      }
      if (state.rangeStatus === 'above main range') { coverage.aboveRangeSamples += 1; continue; }
      coverage.eligibleSamples += 1;
      if (!state.visibleWindowIndexes.length) { coverage.unsupportedSamples += 1; continue; }
      coverage.supportedSamples += 1;
      recordRequirement(state.visibleWindowIndexes);
    }
  }
  const candidateWindows = grid.windows.filter((window) => window.unique !== false);
  const cameraOffsetRadius = Math.hypot(cameraOffset[0], cameraOffset[1]);
  const observeSweptCells = (drone, startTime, endTime, candidates = grid.cells, depth = 0) => {
    if (!(endTime > startTime) || !candidates.length) return;
    const middleTime = (startTime + endTime) / 2;
    const start = evaluate(stateAt(drone, startTime));
    const middle = evaluate(stateAt(drone, middleTime));
    const end = evaluate(stateAt(drone, endTime));
    [[start, startTime], [middle, middleTime], [end, endTime]].forEach(([state, time]) => {
      state.observedCellKeys.forEach((key) => markObserved(key, time));
    });
    const unresolved = candidates.filter((cell) => (observedCellTimes.get(cell.key) ?? Infinity) > startTime + EPSILON);
    if (!unresolved.length) return;
    const intervalMotion = intervalMotionAcross(start, middle, end, cameraOffsetRadius);
    const uncertain = [];
    for (const cell of unresolved) {
      const status = cellStatusAcross(cell, middle, intervalMotion);
      if (status === 'visible') markObserved(cell.key, startTime);
      else if (status === 'uncertain') uncertain.push(cell);
    }
    if (!uncertain.length || depth >= 40 || !(middleTime > startTime && middleTime < endTime)) return;
    observeSweptCells(drone, startTime, middleTime, uncertain, depth + 1);
    observeSweptCells(drone, middleTime, endTime, uncertain, depth + 1);
  };
  let intervalNodes = 0;
  const certifyInterval = (drone, startTime, endTime, depth = 0) => {
    if (!(endTime > startTime)) return;
    const span = endTime - startTime;
    intervalNodes += 1;
    if (intervalNodes > 200000) throw new Error('Continuous coverage certification exceeded its safety limit; split the mission into shorter flights.');
    const start = evaluateGeometry(stateAt(drone, startTime));
    const end = evaluateGeometry(stateAt(drone, endTime));
    const startHeight = start.cameraPosition[2] - grid.origin[2];
    const endHeight = end.cameraPosition[2] - grid.origin[2];
    const minHeight = Math.min(startHeight, endHeight);
    const maxHeight = Math.max(startHeight, endHeight);
    const middleTime = (startTime + endTime) / 2;
    const middle = evaluateGeometry(stateAt(drone, middleTime));
    if (middle.phase === 'takeoff' || middle.phase === 'landing' || middle.phase === 'complete') {
      if (middle.phase === 'takeoff') coverage.shortRangeTakeoffDuration += span;
      if (middle.phase === 'landing') coverage.shortRangeLandingDuration += span;
      return;
    }
    const entirelyBelowRange = maxHeight < grid.workingRange.min - EPSILON
      || (maxHeight <= grid.workingRange.min + EPSILON && minHeight < grid.workingRange.min - EPSILON);
    if (entirelyBelowRange) {
      coverage.unsupportedBelowRangeDuration += span;
      return;
    }
    const entirelyAboveRange = minHeight > grid.workingRange.max + EPSILON
      || (minHeight >= grid.workingRange.max - EPSILON && maxHeight > grid.workingRange.max + EPSILON);
    if (entirelyAboveRange) {
      coverage.aboveRangeDuration += span;
      return;
    }
    const entirelyInRange = minHeight >= grid.workingRange.min - EPSILON && maxHeight <= grid.workingRange.max + EPSILON;
    if (!entirelyInRange) {
      if (depth < 16) {
        certifyInterval(drone, startTime, middleTime, depth + 1);
        certifyInterval(drone, middleTime, endTime, depth + 1);
      } else if (middle.rangeStatus === 'below main range') {
        coverage.unsupportedBelowRangeDuration += span;
      } else if (middle.rangeStatus === 'above main range') {
        coverage.aboveRangeDuration += span;
      } else {
        coverage.eligibleIntervals += 1;
        coverage.unsupportedIntervals += 1;
        coverage.eligibleDuration += span;
        coverage.intervalGaps.push({ droneId: drone.id, startTime, endTime });
      }
      return;
    }
    const intervalMotion = intervalMotionAcross(start, middle, end, cameraOffsetRadius);
    const statuses = candidateWindows.map((window) => [window.index, windowStatusAcross(window, middle, intervalMotion)]);
    const visible = statuses.filter(([, status]) => status === 'visible').map(([index]) => index);
    const hasUncertain = statuses.some(([, status]) => status === 'uncertain');
    if (!hasUncertain) {
      coverage.eligibleIntervals += 1;
      coverage.eligibleDuration += span;
      if (!visible.length) {
        coverage.unsupportedIntervals += 1;
        coverage.intervalGaps.push({ droneId: drone.id, startTime, endTime });
        return;
      }
      coverage.supportedIntervals += 1;
      coverage.supportedDuration += span;
      recordRequirement(visible);
      return;
    }
    const geometryReach = intervalGeometryReach(intervalMotion);
    if (geometryReach > EPSILON && depth < 40 && middleTime > startTime && middleTime < endTime) {
      certifyInterval(drone, startTime, middleTime, depth + 1);
      certifyInterval(drone, middleTime, endTime, depth + 1);
      return;
    }
    coverage.eligibleIntervals += 1;
    coverage.eligibleDuration += span;
    coverage.boundaryIntervals += 1;
    const boundarySets = [start, middle, end].map((state) => evaluate(state).visibleWindowIndexes);
    if (geometryReach <= EPSILON && boundarySets.every((indexes) => indexes.length)) {
      coverage.supportedIntervals += 1;
      coverage.supportedDuration += span;
      boundarySets.forEach(recordRequirement);
      return;
    }
    coverage.unsupportedIntervals += 1;
    coverage.intervalGaps.push({ droneId: drone.id, startTime, endTime });
  };
  for (let timeIndex = 1; timeIndex < sampled.certificationTimes.length; timeIndex += 1) {
    for (const drone of drones) {
      const startTime = sampled.certificationTimes[timeIndex - 1];
      const endTime = sampled.certificationTimes[timeIndex];
      certifyInterval(drone, startTime, endTime);
      observeSweptCells(drone, startTime, endTime);
    }
  }
  coverage.intervalGaps.sort((a, b) => a.droneId.localeCompare(b.droneId) || a.startTime - b.startTime);
  coverage.intervalGaps = coverage.intervalGaps.reduce((merged, gap) => {
    const previous = merged.at(-1);
    if (previous?.droneId === gap.droneId && gap.startTime <= previous.endTime + EPSILON) {
      previous.endTime = Math.max(previous.endTime, gap.endTime);
    } else {
      merged.push({ ...gap });
    }
    return merged;
  }, []);
  coverage.unsupportedIntervals = coverage.intervalGaps.length;
  coverage.certificationNodes = intervalNodes;
  const supportDuration = coverage.eligibleDuration + coverage.aboveRangeDuration + coverage.unsupportedBelowRangeDuration;
  coverage.supportRate = supportDuration > EPSILON ? coverage.supportedDuration / supportDuration : 1;
  const homeWindowIndexes = drones.map((drone) => drone.homeWindowIndex);
  if (new Set(homeWindowIndexes).size !== drones.length) {
    throw new Error('Every drone must have a distinct launch/landing window.');
  }
  const solution = solveMinimumWindowCover([...requirements.values()], grid.windows.length, { preselected: homeWindowIndexes });
  solution.feasible = coverage.unsupportedSamples === 0
    && coverage.unsupportedIntervals === 0
    && coverage.unsupportedBelowRangeSamples === 0
    && coverage.aboveRangeSamples === 0
    && coverage.unsupportedBelowRangeDuration <= EPSILON
    && coverage.aboveRangeDuration <= EPSILON;
  const homeDronesByWindow = new Map();
  for (const drone of drones) {
    const owners = homeDronesByWindow.get(drone.homeWindowIndex) ?? [];
    owners.push(drone.id);
    homeDronesByWindow.set(drone.homeWindowIndex, owners);
  }
  const routeWindowIndexes = new Set(solution.added);
  for (const requirement of requirements.values()) {
    for (const index of requirement) {
      if (homeDronesByWindow.has(index)) routeWindowIndexes.add(index);
    }
  }
  const requiredWindows = solution.selected.map((index) => ({
    ...grid.windows[index],
    homeDroneIds: homeDronesByWindow.get(index) ?? [],
    supportsFlight: routeWindowIndexes.has(index),
  }));
  const requiredCellKeys = [...new Set(requiredWindows.flatMap((window) => window.cells.map((cell) => cell.key)))];
  const bounds = { ...grid.bounds };
  for (const frame of frames) {
    for (const state of frame.drones) {
      for (const [x, y] of state.footprint.corners) {
        bounds.minX = Math.min(bounds.minX, x); bounds.maxX = Math.max(bounds.maxX, x);
        bounds.minY = Math.min(bounds.minY, y); bounds.maxY = Math.max(bounds.maxY, y);
      }
    }
  }
  const ambiguousWindowCount = grid.windows.filter((window) => window.unique === false).length;
  const unalignedFallbackHomes = drones.filter((drone) => !drone.homeTileKey && drone.homeAlignmentError > EPSILON);
  return {
    name: mission.name,
    grid,
    drones,
    duration,
    sampleRate: sampled.rate,
    frames,
    requiredWindows,
    requiredCellKeys,
    observedCellKeys: [...observedCells],
    observedCellTimes: Object.fromEntries(observedCellTimes),
    coverage,
    solution,
    homeWindowIndexes,
    landingSpots: drones.map((drone) => ({
      droneId: drone.id,
      position: drone.landing.slice(),
      windowIndex: drone.homeWindowIndex,
      windowKey: drone.homeWindowKey,
      tileKey: drone.homeTileKey,
    })),
    cameraOffset,
    bounds,
    warnings: [
      ...mission.warnings,
      ...(ambiguousWindowCount ? [`${ambiguousWindowCount} repeated-signature windows were excluded from guaranteed relocalization.`] : []),
      ...(unalignedFallbackHomes.length ? [`${unalignedFallbackHomes.map((drone) => drone.id).join(', ')}: explicit launch XY is not centred on its nearest main-grid home window; vertical support is only a staging assumption.`] : []),
    ],
  };
}
