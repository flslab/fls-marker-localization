const DEFAULTS = Object.freeze({
  hypergridSpacing: 0.155,
  mygridSpacing: 0.024,
  origin: [0, 0, 0],
  maxIds: 16,
  blinkFrequencyHz: 50,
  hypergridMarkerDiameter: 0.01,
  mygridMarkerDiameter: 0.006,
  sceneFps: 120,
});

export const DEFAULT_MARKER_GRID_CONFIG = DEFAULTS;
export const MYGRID_EVENT_ACTIONS = Object.freeze(['on', 'off', 'static', 'blinking']);

const finite = (value, name) => {
  const number = Number(value);
  if (!Number.isFinite(number)) throw new Error(`${name} must be finite.`);
  return number;
};

const positive = (value, name) => {
  const number = finite(value, name);
  if (number <= 0) throw new Error(`${name} must be positive.`);
  return number;
};

const integer = (value, name, minimum, maximum) => {
  const number = finite(value, name);
  if (!Number.isInteger(number) || number < minimum || number > maximum) {
    throw new Error(`${name} must be an integer from ${minimum} to ${maximum}.`);
  }
  return number;
};

const compareRings = (a, b) => {
  for (let index = 0; index < a.length; index += 1) {
    if (a[index] !== b[index]) return a[index] - b[index];
  }
  return 0;
};

export function rotateRing(signature, offset) {
  return signature.map((_, index) => signature[(index + offset) % signature.length]);
}

export function canonicalRing(signature, includeReflection = false) {
  if (!Array.isArray(signature) || signature.length < 2) throw new Error('A ring needs at least two markers.');
  const candidates = signature.map((_, index) => rotateRing(signature, index));
  if (includeReflection) {
    const reversed = [...signature].reverse();
    reversed.forEach((_, index) => candidates.push(rotateRing(reversed, index)));
  }
  return candidates.reduce((best, candidate) => compareRings(candidate, best) < 0 ? candidate : best, candidates[0]);
}

export function hasRotationalSymmetry(signature) {
  return signature.slice(1).some((_, offset) => compareRings(signature, rotateRing(signature, offset + 1)) === 0);
}

function incrementDigits(digits, base) {
  for (let index = digits.length - 1; index >= 0; index -= 1) {
    digits[index] += 1;
    if (digits[index] < base) return true;
    digits[index] = 0;
  }
  return false;
}

export function generateRingSignatures(count, maxIds, markersPerRing = 4) {
  const wanted = integer(count, 'Tile count', 0, 500);
  const alphabetSize = integer(maxIds, 'Maximum IDs', 2, 65536);
  const ringSize = integer(markersPerRing, 'Markers per ring', 4, 16);
  const signatures = [];
  const digits = Array(ringSize).fill(0);
  let hasNext = true;

  while (hasNext && signatures.length < wanted) {
    const canonical = canonicalRing(digits);
    const isCanonical = compareRings(digits, canonical) === 0;
    // Using one representative from each reflection pair is stronger than
    // rotational uniqueness and prevents a mirrored installation being
    // mistaken for a different tile.
    const reflectionCanonical = canonicalRing(digits, true);
    if (isCanonical && compareRings(digits, reflectionCanonical) === 0 && !hasRotationalSymmetry(digits)) {
      signatures.push([...digits]);
    }
    hasNext = incrementDigits(digits, alphabetSize);
  }

  if (signatures.length !== wanted) {
    throw new Error(`Only ${signatures.length} asymmetric ${ringSize}-marker signatures exist with ${alphabetSize} IDs. Increase the maximum ID count.`);
  }
  return signatures;
}

export function normalizeMarkerGridConfig(raw = {}) {
  const origin = Array.isArray(raw.origin) ? raw.origin : DEFAULTS.origin;
  if (origin.length !== 3) throw new Error('Origin must contain X, Y, and Z.');
  return {
    hypergridSpacing: positive(raw.hypergridSpacing ?? DEFAULTS.hypergridSpacing, 'HyperGrid spacing'),
    mygridSpacing: positive(raw.mygridSpacing ?? DEFAULTS.mygridSpacing, 'MyGrid spacing'),
    origin: origin.map((value, index) => finite(value, `Origin ${'XYZ'[index]}`)),
    maxIds: integer(raw.maxIds ?? DEFAULTS.maxIds, 'Maximum IDs', 2, 65536),
    blinkFrequencyHz: positive(raw.blinkFrequencyHz ?? DEFAULTS.blinkFrequencyHz, 'Blink frequency'),
    hypergridMarkerDiameter: positive(raw.hypergridMarkerDiameter ?? DEFAULTS.hypergridMarkerDiameter, 'HyperGrid marker diameter'),
    mygridMarkerDiameter: positive(raw.mygridMarkerDiameter ?? DEFAULTS.mygridMarkerDiameter, 'MyGrid marker diameter'),
    sceneFps: integer(raw.sceneFps ?? DEFAULTS.sceneFps, 'Scene FPS', 1, 1000),
  };
}

const round = (value) => Number(value.toFixed(9));

export function hypergridMarkersForTile(rawConfig, rawI, rawJ) {
  const config = normalizeMarkerGridConfig(rawConfig);
  const i = integer(rawI, 'Tile X', -1000000, 1000000);
  const j = integer(rawJ, 'Tile Y', -1000000, 1000000);
  const tileSize = config.hypergridSpacing * 2;
  const center = [
    config.origin[0] + i * tileSize,
    config.origin[1] + j * tileSize,
    config.origin[2],
  ];
  return [[0, 0], [0, 1], [1, 1], [1, 0]].map(([localRow, localCol]) => {
    const gridX = i * 2 + localCol - 1;
    const gridY = j * 2 - localRow;
    const position = [
      round(center[0] + (localCol - 0.5) * config.hypergridSpacing),
      round(center[1] + (0.5 - localRow) * config.hypergridSpacing),
      round(center[2]),
    ];
    return {
      tile_coordinates: [i, j],
      local_grid_coordinates: [localRow, localCol],
      grid_coordinates: [gridX, gridY],
      global_position: position,
      global_x: position[0],
      global_y: position[1],
      global_z: position[2],
    };
  });
}

export function buildMarkerGridFile(rawConfig, rawTiles = []) {
  const config = normalizeMarkerGridConfig(rawConfig);
  if (!Array.isArray(rawTiles)) throw new Error('Tiles must be an array.');
  const keys = new Set();
  const tiles = rawTiles.map((tile) => {
    const i = integer(tile.i, 'Tile X', -1000000, 1000000);
    const j = integer(tile.j, 'Tile Y', -1000000, 1000000);
    const key = `${i}:${j}`;
    if (keys.has(key)) throw new Error(`Tile (${i}, ${j}) is selected more than once.`);
    keys.add(key);
    return { ...tile, i, j };
  }).sort((a, b) => a.i - b.i || a.j - b.j);

  const tileSize = config.hypergridSpacing * 2;
  if (config.mygridSpacing >= tileSize) throw new Error('MyGrid spacing must be smaller than the tile size.');
  const signatures = generateRingSignatures(tiles.length, config.maxIds, 4);
  const payloadBits = Math.max(1, Math.ceil(Math.log2(config.maxIds)));
  const delimiterBits = payloadBits + 1;

  const mygridTiles = tiles.map((tile, tileIndex) => {
    const center = [
      config.origin[0] + tile.i * tileSize,
      config.origin[1] + tile.j * tileSize,
      config.origin[2],
    ].map(round);
    const signature = signatures[tileIndex];
    if (tile.events !== undefined && !Array.isArray(tile.events)) throw new Error(`Tile (${tile.i}, ${tile.j}) events must be an array.`);
    const events = (tile.events ?? []).map((event, eventIndex) => {
      const time = finite(event.time, `Tile (${tile.i}, ${tile.j}) event ${eventIndex + 1} time`);
      if (time < 0) throw new Error(`Tile (${tile.i}, ${tile.j}) event times cannot be negative.`);
      if (!MYGRID_EVENT_ACTIONS.includes(event.action)) throw new Error(`Tile (${tile.i}, ${tile.j}) event ${eventIndex + 1} has an invalid action.`);
      return { time_s: time, state: event.action };
    }).sort((a, b) => a.time_s - b.time_s);
    for (let index = 1; index < events.length; index += 1) {
      if (events[index].time_s === events[index - 1].time_s) throw new Error(`Tile (${tile.i}, ${tile.j}) cannot have two events at ${events[index].time_s} s.`);
    }

    // Clockwise from top-left. This is a cyclic logical ordering for yaw
    // matching; the physical layout is a centred 2x2 square, not a circle.
    const localCoordinates = [[0, 0], [0, 1], [1, 1], [1, 0]];
    const markers = signature.map((id, signatureIndex) => {
      const [localRow, localCol] = localCoordinates[signatureIndex];
      const offset = [
        round((localCol - 0.5) * config.mygridSpacing),
        round((0.5 - localRow) * config.mygridSpacing),
        0,
      ];
      const position = [round(center[0] + offset[0]), round(center[1] + offset[1]), center[2]];
      return {
        signature_index: signatureIndex,
        local_row: localRow,
        local_col: localCol,
        tile_coordinates: [tile.i, tile.j],
        local_grid_coordinates: [localRow, localCol],
        grid_coordinates: { tile: [tile.i, tile.j], local: [localRow, localCol] },
        id,
        offset,
        global_position: position,
        global_x: position[0],
        global_y: position[1],
        global_z: position[2],
      };
    });

    return {
      i: tile.i,
      j: tile.j,
      center,
      signature,
      signature_is_cyclic: true,
      signature_order: 'clockwise_from_top_left',
      layout: 'centered_2x2',
      marker_spacing: round(config.mygridSpacing),
      markers,
      ...(events.length ? { blender_animation: { initial_state: 'off', events } } : {}),
    };
  });

  return {
    schema: 'fls-marker-grid',
    schema_version: 1,
    units: 'metres',
    coordinate_frame: 'world_FLU',
    grid_origin: config.origin.map(round),
    origin_definition: 'centre of tile (0, 0)',
    hypergrid: {
      infinite: true,
      marker_spacing: round(config.hypergridSpacing),
      tile_size: round(tileSize),
      marker_diameter: round(config.hypergridMarkerDiameter),
      markers_per_tile: 4,
      layout: 'centered_2x2',
      tile_local_offsets: [
        [-0.5, 0.5, 0], [0.5, 0.5, 0],
        [0.5, -0.5, 0], [-0.5, -0.5, 0],
      ],
      offset_units: 'marker_spacing',
      grid_coordinate_position: 'origin + [(grid_x + 0.5) * marker_spacing, (grid_y + 0.5) * marker_spacing, 0]',
      markers_always_on: true,
      markers_have_ids: false,
    },
    mygrid: {
      selected_tile_count: mygridTiles.length,
      markers_per_tile: 4,
      layout: 'centered_2x2',
      marker_spacing: round(config.mygridSpacing),
      marker_diameter: round(config.mygridMarkerDiameter),
      signature_equivalence: 'cyclic_rotation',
      signatures_are_asymmetric: true,
      tiles: mygridTiles,
    },
    encoding: {
      max_ids: config.maxIds,
      id_range: [0, config.maxIds - 1],
      payload_bits: payloadBits,
      delimiter_bits: delimiterBits,
      delimiter_pattern: `${'1'.repeat(delimiterBits)}0`,
      packet_format: `[${payloadBits}-bit ID][${delimiterBits} × 1][0]`,
      payload_bit_order: 'most_significant_first',
      blink_frequency_hz: round(config.blinkFrequencyHz),
      bit_frequency_hz: round(config.blinkFrequencyHz),
      bit_duration_s: round(1 / config.blinkFrequencyHz),
    },
    blender: {
      scene_fps: config.sceneFps,
      hypergrid_marker_diameter: round(config.hypergridMarkerDiameter),
      mygrid_marker_diameter: round(config.mygridMarkerDiameter),
      mygrid_marker_spacing: round(config.mygridSpacing),
      emission_strength: 5,
      min_brightness: 0,
      max_brightness: 1,
      interpolate: 'constant',
      repeat_packets: true,
    },
  };
}

export function markerGridJson(config, tiles) {
  return `${JSON.stringify(buildMarkerGridFile(config, tiles), null, 2)}\n`;
}
