const isObject = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);

export const isFiniteNumber = (value) => typeof value === 'number' && Number.isFinite(value);

export function asVector(value, length = 3) {
  return Array.isArray(value) && value.length >= length && value.slice(0, length).every(isFiniteNumber)
    ? value.slice(0, length)
    : null;
}

export function timeToSeconds(value) {
  if (!isFiniteNumber(value)) return null;
  return Math.abs(value) > 1e11 ? value / 1000 : value;
}

function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function orientationFrom(record, kind) {
  // Historical writers sometimes duplicated the same YPR vector into the
  // newer-looking orientation key, so the explicit legacy key must win.
  const historical = asVector(record.yaw_pitch_roll);
  if (historical) return [historical[2], historical[1], historical[0]];
  const direct = kind === 'legacy'
    ? (asVector(record.marker_orientation) || asVector(record.camera_orientation))
    : asVector(record.camera_orientation);
  if (direct) return direct;
  return null;
}

export function classifyPose(record) {
  if (!isObject(record)) return null;

  if (record.marker_pose === true) {
    return {
      kind: 'marker-world',
      label: `Marker ${record.marker_id ?? '?'}`,
      frameLabel: 'world frame',
      position: asVector(record.marker_position),
      filteredPosition: null,
      orientation: asVector(record.marker_orientation),
      markerId: record.marker_id,
      source: 'aruco-marker',
      raw: record,
    };
  }

  if (record.camera_pose === true) {
    return {
      kind: 'camera-world',
      label: 'Camera',
      frameLabel: 'world frame',
      position: asVector(record.camera_position) || asVector(record.tvec),
      filteredPosition: asVector(record.camera_position_filtered) || asVector(record.tvec_filtered),
      orientation: orientationFrom(record, 'camera-world'),
      markerId: null,
      source: record.source === 'blob_grid' ? 'blob-grid' : ('tvec' in record ? 'historical-aruco' : 'aruco'),
      raw: record,
    };
  }

  if ('marker_position' in record || ('camera_position' in record && 'marker_id' in record)) {
    const historicalShape = 'tvec' in record || 'yaw_pitch_roll' in record;
    const primaryIsCamera = (historicalShape || asVector(record.camera_position_filtered))
      && !asVector(record.marker_position_filtered);
    const position = primaryIsCamera
      ? (asVector(record.camera_position) || asVector(record.tvec))
      : (asVector(record.marker_position) || asVector(record.tvec));
    const filteredPosition = primaryIsCamera
      ? (asVector(record.camera_position_filtered) || asVector(record.tvec_filtered))
      : asVector(record.marker_position_filtered);
    const orientation = primaryIsCamera
      ? orientationFrom(record, 'camera-world')
      : (asVector(record.marker_orientation) || orientationFrom(record, 'legacy'));
    return {
      kind: 'legacy',
      entity: primaryIsCamera ? 'camera' : 'marker',
      label: `${primaryIsCamera ? 'Camera · marker' : 'Marker'} ${record.marker_id ?? '?'}`,
      frameLabel: primaryIsCamera ? 'marker frame' : 'camera frame',
      position,
      filteredPosition,
      orientation,
      cameraPosition: asVector(record.camera_position),
      cameraOrientation: asVector(record.camera_orientation),
      markerPosition: asVector(record.marker_position),
      markerOrientation: asVector(record.marker_orientation),
      markerId: record.marker_id,
      source: 'legacy-blob',
      raw: record,
    };
  }

  if ('tvec' in record) {
    return {
      kind: 'historical-marker',
      entity: 'marker',
      label: 'Marker',
      frameLabel: 'camera frame',
      position: asVector(record.tvec),
      filteredPosition: asVector(record.tvec_filtered),
      orientation: orientationFrom(record, 'historical'),
      markerId: record.marker_id,
      source: 'historical',
      raw: record,
    };
  }

  return {
    kind: 'unknown',
    label: 'Unknown pose',
    frameLabel: 'unknown frame',
    position: null,
    filteredPosition: null,
    orientation: null,
    markerId: record.marker_id,
    source: 'unknown',
    raw: record,
  };
}

function inferMode(raw, frames) {
  if (raw?.config?.aruco_mode === true || raw?.args?.aruco_mode === true) return 'ArUco';
  if (raw?.config?.blob_grid_localization_enabled === true || frames.some((frame) => isObject(frame?.blob_grid_localization))) return 'Blob grid';
  if (frames.some((frame) => Array.isArray(frame?.poses) && frame.poses.some((pose) => pose?.camera_pose === true && pose?.source !== 'blob_grid'))) return 'ArUco';
  if (frames.some((frame) => 'tvec' in (frame || {}) || frame?.yaw_pitch_roll
    || (Array.isArray(frame?.poses) && frame.poses.some((pose) => 'tvec' in (pose || {}) || pose?.yaw_pitch_roll)))) return 'Historical';
  if (frames.some((frame) => Object.prototype.hasOwnProperty.call(frame || {}, 'blobs')
    || (Array.isArray(frame?.poses) && frame.poses.some((pose) => 'marker_position' in (pose || {}))))) return 'Legacy blob';
  return 'Unknown';
}

function frameStatus(frame, primary) {
  const grid = isObject(frame.blob_grid_localization) ? frame.blob_grid_localization : null;
  if (grid?.status) return String(grid.status);
  if (primary?.position) return 'pose';
  if (Array.isArray(frame.poses) && frame.poses.length) return 'unrecognized_pose';
  return 'no_pose';
}

function collectWorldMarkers(rawFrames) {
  const markers = new Map();
  for (const frame of rawFrames) {
    const grid = isObject(frame?.blob_grid_localization) ? frame.blob_grid_localization : null;
    for (const marker of Array.isArray(grid?.matched_markers) ? grid.matched_markers : []) {
      const position = asVector(marker?.global_position);
      if (!position) continue;
      const key = `grid:${marker.map_row ?? '?'}:${marker.map_col ?? '?'}:${marker.id ?? '?'}`;
      markers.set(key, {
        key,
        kind: 'grid',
        id: marker.id,
        row: marker.map_row,
        col: marker.map_col,
        position,
        orientation: null,
      });
    }
    for (const pose of Array.isArray(frame?.poses) ? frame.poses : []) {
      if (pose?.marker_pose !== true) continue;
      const position = asVector(pose.marker_position);
      if (!position) continue;
      const key = `aruco:${pose.marker_id ?? '?'}`;
      markers.set(key, {
        key,
        kind: 'aruco',
        id: pose.marker_id,
        row: null,
        col: null,
        position,
        orientation: asVector(pose.marker_orientation),
      });
    }
    for (const pose of Array.isArray(frame?.poses) ? frame.poses : []) {
      for (const marker of Array.isArray(pose?.marker_poses) ? pose.marker_poses : []) {
        const position = asVector(marker?.marker_position);
        if (!position) continue;
        const directOrientation = asVector(marker.marker_orientation);
        // The short-lived nested ArUco schema stored marker_orientation as YPR;
        // its parent yaw_pitch_roll key identifies that historical contract.
        const orientation = directOrientation && asVector(pose.yaw_pitch_roll)
          ? [directOrientation[2], directOrientation[1], directOrientation[0]]
          : directOrientation;
        const key = `aruco:${marker.marker_id ?? '?'}`;
        markers.set(key, { key, kind: 'aruco', id: marker.marker_id, row: null, col: null, position, orientation });
      }
    }
  }
  return [...markers.values()];
}

function collectMarkerIds(rawFrames) {
  const ids = new Set();
  const add = (value) => {
    if (Number.isInteger(value) && value >= 0) ids.add(value);
  };
  for (const frame of rawFrames) {
    for (const blob of Array.isArray(frame?.blobs) ? frame.blobs : []) add(blob?.id);
    for (const pose of Array.isArray(frame?.poses) ? frame.poses : []) {
      add(pose?.marker_id);
      for (const id of Array.isArray(pose?.detected_ids) ? pose.detected_ids : []) add(id);
      for (const id of Array.isArray(pose?.used_marker_ids) ? pose.used_marker_ids : []) add(id);
    }
    const grid = frame?.blob_grid_localization;
    for (const track of Array.isArray(grid?.decoded_tracks) ? grid.decoded_tracks : []) add(track?.id);
  }
  return [...ids].sort((a, b) => a - b);
}

export function createLogModel(raw, fileName = 'log.json') {
  if (!isObject(raw)) throw new Error('The log root must be a JSON object.');

  const warnings = [];
  const rawFrames = Array.isArray(raw.frames) ? raw.frames : [];
  if (!Array.isArray(raw.frames)) warnings.push('This log has no frames array. Run metadata and raw JSON are still available.');
  if (!isObject(raw.args)) warnings.push('Run arguments are missing or are not an object.');
  if (!isObject(raw.config)) warnings.push('Run configuration is missing or is not an object.');

  const firstFiniteTime = rawFrames.map((frame) => timeToSeconds(frame?.time)).find((value) => value !== null) ?? 0;
  let invalidFrameCount = 0;
  const frames = rawFrames.map((frame, index) => {
    const safeFrame = isObject(frame) ? frame : {};
    if (!isObject(frame)) invalidFrameCount += 1;
    const poseRecords = Array.isArray(safeFrame.poses)
      ? safeFrame.poses
      : ('tvec' in safeFrame ? [safeFrame] : []);
    const poses = poseRecords.map(classifyPose).filter(Boolean);
    const primary = poses.find((pose) => pose.kind === 'camera-world')
      || poses.find((pose) => pose.kind === 'legacy' || pose.kind === 'historical-marker')
      || poses.find((pose) => pose.position)
      || null;
    const seconds = timeToSeconds(safeFrame.time);
    const grid = isObject(safeFrame.blob_grid_localization) ? safeFrame.blob_grid_localization : null;
    const reprojectionError = isFiniteNumber(primary?.raw?.reprojection_error)
      ? primary.raw.reprojection_error
      : (isFiniteNumber(grid?.reprojection_error) ? grid.reprojection_error : null);
    const status = frameStatus(safeFrame, primary);
    return {
      index,
      raw: frame,
      safeRaw: safeFrame,
      frameId: Number.isInteger(safeFrame.frame_id) ? safeFrame.frame_id : index,
      rawTime: safeFrame.time,
      seconds,
      t: seconds === null ? null : seconds - firstFiniteTime,
      poseRecords,
      poses,
      primary,
      blobs: Array.isArray(safeFrame.blobs) ? safeFrame.blobs : [],
      grid,
      status,
      poseValid: grid ? grid.pose_valid === true : Boolean(primary?.position),
      reprojectionError,
      counts: {
        poses: poseRecords.length,
        blobs: Array.isArray(safeFrame.blobs) ? safeFrame.blobs.length : null,
        tracked: isFiniteNumber(grid?.tracked_decoded_marker_count) ? grid.tracked_decoded_marker_count : null,
        decoded: isFiniteNumber(grid?.decoded_marker_count) ? grid.decoded_marker_count : null,
        accepted: isFiniteNumber(grid?.accepted_marker_count) ? grid.accepted_marker_count : null,
        required: isFiniteNumber(grid?.required_marker_count) ? grid.required_marker_count : null,
        candidates: isFiniteNumber(grid?.candidate_count) ? grid.candidate_count : null,
        matched: Array.isArray(grid?.matched_markers) ? grid.matched_markers.length : null,
        markersUsed: isFiniteNumber(primary?.raw?.markers_used) ? primary.raw.markers_used : null,
      },
    };
  });

  if (invalidFrameCount) warnings.push(`${invalidFrameCount} frame record${invalidFrameCount === 1 ? ' is' : 's are'} malformed; valid fields were kept.`);

  const ids = frames.map((frame) => frame.frameId);
  const duplicateIds = ids.filter((id, index) => ids.indexOf(id) !== index);
  if (duplicateIds.length) warnings.push(`${new Set(duplicateIds).size} duplicate frame ID${new Set(duplicateIds).size === 1 ? '' : 's'} detected.`);

  const finiteTimes = frames.map((frame) => frame.seconds).filter((value) => value !== null);
  const deltas = finiteTimes.slice(1).map((value, index) => value - finiteTimes[index]).filter((value) => value > 0);
  const nonMonotonic = finiteTimes.slice(1).some((value, index) => value < finiteTimes[index]);
  if (nonMonotonic) warnings.push('Timestamps are not monotonic; plots preserve the original frame order.');

  const medianDelta = median(deltas);
  const relativeTimes = frames.map((frame) => frame.t).filter(isFiniteNumber);
  const timeMin = relativeTimes.length ? Math.min(...relativeTimes) : 0;
  const timeMax = relativeTimes.length ? Math.max(...relativeTimes) : 0;
  const duration = Math.max(0, timeMax - timeMin);
  const validPoseCount = frames.filter((frame) => frame.poseValid).length;
  const positionFrames = frames.filter((frame) => frame.primary?.position);
  const hasFiltered = frames.some((frame) => frame.primary?.filteredPosition);
  const worldCameraPath = frames
    .filter((frame) => frame.primary?.kind === 'camera-world' && frame.primary.position)
    .map((frame) => ({ frameIndex: frame.index, position: frame.primary.position, filteredPosition: frame.primary.filteredPosition }));
  const poseMarkerIds = [...new Set(frames.flatMap((frame) => frame.poses
    .filter((pose) => (pose.kind === 'legacy' || pose.kind === 'historical-marker') && Number.isInteger(pose.markerId))
    .map((pose) => pose.markerId)))].sort((a, b) => a - b);

  return {
    raw,
    fileName,
    args: isObject(raw.args) ? raw.args : {},
    config: isObject(raw.config) ? raw.config : {},
    frames,
    mode: inferMode(raw, rawFrames),
    warnings,
    duration,
    timeMin,
    timeMax,
    fps: medianDelta ? 1 / medianDelta : 0,
    validPoseCount,
    poseRate: frames.length ? validPoseCount / frames.length : 0,
    markerIds: collectMarkerIds(rawFrames),
    worldMarkers: collectWorldMarkers(rawFrames),
    worldCameraPath,
    poseMarkerIds,
    hasFiltered,
    hasPoseData: positionFrames.length > 0,
    timestampBasis: finiteTimes.some((value) => value > 1e9) ? 'epoch' : 'relative',
  };
}

export function parseLogText(text, fileName = 'log.json') {
  let raw;
  try {
    raw = JSON.parse(text);
  } catch (error) {
    throw new Error(`Could not parse JSON: ${error.message}`);
  }
  return createLogModel(raw, fileName);
}

export function flattenObject(value, prefix = '', output = []) {
  if (Array.isArray(value)) {
    if (!value.length) output.push({ path: prefix || '(root)', value: [], type: 'empty array' });
    value.forEach((item, index) => flattenObject(item, `${prefix}[${index}]`, output));
    return output;
  }
  if (isObject(value)) {
    const entries = Object.entries(value);
    if (!entries.length) output.push({ path: prefix || '(root)', value: {}, type: 'empty object' });
    entries.forEach(([key, item]) => flattenObject(item, prefix ? `${prefix}.${key}` : key, output));
    return output;
  }
  output.push({ path: prefix || '(root)', value, type: value === null ? 'null' : typeof value });
  return output;
}

export function formatNumber(value, digits = 4) {
  if (!isFiniteNumber(value)) return '—';
  if (value === 0) return '0';
  if (Math.abs(value) >= 100000 || Math.abs(value) < 0.0001) return value.toExponential(Math.max(1, Math.min(12, digits)));
  return value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: 0 });
}

export function formatRawTime(value) {
  if (!isFiniteNumber(value)) return value === null ? 'null' : '—';
  return String(value);
}

export function valueToText(value) {
  if (value === null) return 'null';
  if (value === undefined) return '—';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number') return formatNumber(value, 8);
  if (typeof value === 'string') return value === '' ? '(empty string)' : value;
  return JSON.stringify(value);
}
