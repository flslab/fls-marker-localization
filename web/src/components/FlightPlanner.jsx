import { useEffect, useMemo, useRef, useState } from 'react';
import { load as loadYaml } from 'js-yaml';
import {
  AlertTriangle, CheckCircle2, FileJson, FolderOpen, Grid3X3,
  Pause, Play, RotateCcw, SkipBack, SkipForward,
} from 'lucide-react';
import { buildFlightModel, DEFAULT_CAMERA_OFFSET, DEFAULT_FLIGHT_GRID, normalizeGrid } from '../lib/flightModel.js';

const format = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const supportPercent = (model) => !model.solution.feasible && model.coverage.supportRate >= 0.9995
  ? '<100%'
  : `${(model.coverage.supportRate * 100).toFixed(1)}%`;
const svgPoints = (points) => points.map(([x, y]) => `${x},${-y}`).join(' ');
const BUILT_IN_GRID = { name: 'marker_grid_20x20_h.json', raw: DEFAULT_FLIGHT_GRID, builtIn: true };

function nearestFrameIndex(frames, time) {
  let low = 0;
  let high = frames.length - 1;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (frames[middle].time < time) low = middle + 1;
    else high = middle;
  }
  if (low === 0) return 0;
  return frames[low].time - time < time - frames[low - 1].time ? low : low - 1;
}

function FlightScene({ model, frameIndex }) {
  const frame = model.frames[frameIndex] ?? model.frames[0];
  const history = useMemo(() => model.frames.slice(0, frameIndex + 1), [model, frameIndex]);
  const observed = useMemo(() => {
    if (model.observedCellTimes) {
      return new Set(Object.entries(model.observedCellTimes)
        .filter(([, firstSeen]) => firstSeen <= frame.time + 1e-9)
        .map(([key]) => key));
    }
    const keys = new Set();
    history.forEach((sample) => sample.drones.forEach((drone) => drone.observedCellKeys.forEach((key) => keys.add(key))));
    return keys;
  }, [model, frame.time, history]);
  const required = useMemo(() => new Set(model.requiredCellKeys), [model]);
  const activeWindows = useMemo(() => {
    const visible = new Set(frame.drones.flatMap((drone) => drone.visibleWindowIndexes));
    return new Set(model.requiredWindows.filter((window) => visible.has(window.index)).map((window) => window.key));
  }, [model, frame]);
  const width = model.bounds.maxX - model.bounds.minX;
  const height = model.bounds.maxY - model.bounds.minY;
  const padding = Math.max(model.grid.spacing, Math.max(width, height) * 0.055);
  const viewBox = `${model.bounds.minX - padding} ${-model.bounds.maxY - padding} ${width + padding * 2} ${height + padding * 2}`;
  const markerRadius = Math.max(model.grid.markerSize / 2, model.grid.spacing * 0.018);
  const cellSize = model.grid.spacing * 0.82;
  const droneRadius = model.grid.spacing * 0.09;
  const traceStride = Math.max(1, Math.ceil(history.length / 180), Math.round(model.sampleRate * 0.35));
  const footprintHistory = history.filter((_, index) => index % traceStride === 0 || index === history.length - 1);
  const pathByDrone = new Map(model.drones.map((drone) => [
    drone.id,
    history.map((sample) => sample.drones.find((state) => state.id === drone.id).position.slice(0, 2)),
  ]));

  return (
    <svg className="flight-scene" viewBox={viewBox} role="img" aria-label={`Top-down swarm flight, launch and landing spots, accumulated camera coverage, marker grid, and ${model.solution.feasible && model.solution.optimal ? 'minimum required' : 'selected support'} windows`}>
      <defs>
        <pattern id="floor-grid" width={model.grid.spacing} height={model.grid.spacing} patternUnits="userSpaceOnUse">
          <path d={`M ${model.grid.spacing} 0 L 0 0 0 ${model.grid.spacing}`} className="flight-floor-grid" />
        </pattern>
      </defs>
      <rect
        x={model.grid.bounds.minX - model.grid.spacing / 2}
        y={-model.grid.bounds.maxY - model.grid.spacing / 2}
        width={model.grid.bounds.maxX - model.grid.bounds.minX + model.grid.spacing}
        height={model.grid.bounds.maxY - model.grid.bounds.minY + model.grid.spacing}
        className="flight-floor"
      />
      <rect
        x={model.grid.bounds.minX - model.grid.spacing / 2}
        y={-model.grid.bounds.maxY - model.grid.spacing / 2}
        width={model.grid.bounds.maxX - model.grid.bounds.minX + model.grid.spacing}
        height={model.grid.bounds.maxY - model.grid.bounds.minY + model.grid.spacing}
        fill="url(#floor-grid)"
      />

      {model.grid.cells.map((cell) => {
        const isObserved = observed.has(cell.key);
        const isRequired = required.has(cell.key);
        return <rect key={`coverage-${cell.key}`} x={cell.position[0] - cellSize / 2} y={-cell.position[1] - cellSize / 2} width={cellSize} height={cellSize} className={`flight-covered-cell${isObserved ? ' observed' : ''}${isRequired ? ' required' : ''}`} />;
      })}

      {footprintHistory.flatMap((sample, sampleIndex) => sample.drones.map((drone) => (
        <polygon key={`sweep-${sampleIndex}-${drone.id}`} points={svgPoints(drone.footprint.corners)} className="flight-footprint-history" style={{ '--drone-color': drone.color }} />
      )))}

      {model.drones.map((drone) => <polyline key={`full-${drone.id}`} points={svgPoints(drone.path.map((point) => point.slice(0, 2)))} className="flight-path planned" style={{ '--drone-color': drone.color }} />)}
      {model.drones.map((drone) => <polyline key={`trace-${drone.id}`} points={svgPoints(pathByDrone.get(drone.id))} className="flight-path traced" style={{ '--drone-color': drone.color }} />)}

      {model.requiredWindows.map((window) => {
        const { minX, maxX, minY, maxY } = window.bounds;
        return (
          <g key={`required-${window.key}`} className={`flight-window${window.homeDroneIds.length ? ' home' : ''}${activeWindows.has(window.key) ? ' active' : ''}`}>
            <rect x={minX} y={-maxY} width={maxX - minX} height={maxY - minY} />
            <text x={window.center[0]} y={-window.center[1]}>{window.row},{window.col}</text>
          </g>
        );
      })}

      {model.grid.cells.map((cell) => <circle key={`marker-${cell.key}`} cx={cell.position[0]} cy={-cell.position[1]} r={markerRadius} className={required.has(cell.key) ? 'flight-marker required' : 'flight-marker'} />)}

      {model.drones.map((drone) => (
        <g key={`ends-${drone.id}`} style={{ '--drone-color': drone.color }}>
          <circle cx={drone.landing[0]} cy={-drone.landing[1]} r={droneRadius * 1.15} className="flight-landing" />
          <rect x={drone.takeoff[0] - droneRadius * 0.65} y={-drone.takeoff[1] - droneRadius * 0.65} width={droneRadius * 1.3} height={droneRadius * 1.3} className="flight-takeoff" transform={`rotate(45 ${drone.takeoff[0]} ${-drone.takeoff[1]})`} />
          <path d={`M ${drone.target[0] - droneRadius} ${-drone.target[1]} H ${drone.target[0] + droneRadius} M ${drone.target[0]} ${-drone.target[1] - droneRadius} V ${-drone.target[1] + droneRadius}`} className="flight-target" />
        </g>
      ))}

      {frame.drones.map((drone) => {
        const [x, y, z] = drone.position;
        const arrowLength = droneRadius * 2.2;
        const arrowX = x + Math.cos(drone.yaw) * arrowLength;
        const arrowY = y + Math.sin(drone.yaw) * arrowLength;
        return (
          <g key={`drone-${drone.id}`} className="flight-drone" style={{ '--drone-color': drone.color }}>
            <polygon points={svgPoints(drone.footprint.corners)} className="flight-footprint-current" />
            <circle cx={x} cy={-y} r={droneRadius} />
            <line x1={x} y1={-y} x2={arrowX} y2={-arrowY} />
            <text x={x + droneRadius * 1.35} y={-y - droneRadius * 1.15}>{drone.id} · {format(z, 2)}m</text>
          </g>
        );
      })}
      <text x={model.grid.bounds.maxX} y={-model.grid.bounds.minY + padding * 0.62} className="flight-axis-label" textAnchor="end">world +X → · world +Y ↑</text>
    </svg>
  );
}

function FlightTransport({ model, frameIndex, setFrameIndex, playing, setPlaying, speed, setSpeed }) {
  const frame = model.frames[frameIndex];
  useEffect(() => {
    if (!playing || model.frames.length < 2) return undefined;
    let last = performance.now();
    let playhead = frame?.time ?? 0;
    let animationFrame;
    const tick = (now) => {
      playhead += Math.min(0.25, (now - last) / 1000) * speed;
      last = now;
      if (playhead >= model.duration) {
        setFrameIndex(model.frames.length - 1);
        setPlaying(false);
        return;
      }
      let low = 0;
      let high = model.frames.length - 1;
      while (low < high) {
        const middle = Math.ceil((low + high) / 2);
        if (model.frames[middle].time <= playhead) low = middle;
        else high = middle - 1;
      }
      setFrameIndex(low);
      animationFrame = requestAnimationFrame(tick);
    };
    animationFrame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animationFrame);
  }, [playing, model, speed]);

  return (
    <div className="transport flight-transport">
      <div className="transport-buttons">
        <button onClick={() => { setPlaying(false); setFrameIndex(0); }} disabled={!frameIndex} title="Start"><SkipBack size={14} /></button>
        <button className="play-button" onClick={() => { if (!playing && frameIndex >= model.frames.length - 1) setFrameIndex(0); setPlaying((value) => !value); }} aria-label={playing ? 'Pause flight' : 'Play flight'}>{playing ? <Pause size={15} fill="currentColor" /> : <Play size={15} fill="currentColor" />}</button>
        <button onClick={() => { setPlaying(false); setFrameIndex(model.frames.length - 1); }} disabled={frameIndex >= model.frames.length - 1} title="End"><SkipForward size={14} /></button>
      </div>
      <div className="timeline-copy"><span>MISSION TIME</span><strong>{format(frame?.time, 2)} / {format(model.duration, 2)} s</strong></div>
      <input
        className="timeline-range"
        type="range"
        min="0"
        max={model.duration}
        step={Math.max(model.duration / 1000, 0.001)}
        value={frame?.time ?? 0}
        onChange={(event) => { setPlaying(false); setFrameIndex(nearestFrameIndex(model.frames, Number(event.target.value))); }}
        aria-label="Flight time"
        aria-valuetext={`${format(frame?.time, 2)} seconds`}
        style={{ '--progress': `${model.duration > 0 ? (frame?.time ?? 0) / model.duration * 100 : 0}%` }}
      />
      <label className="speed-control">speed<select value={speed} onChange={(event) => setSpeed(Number(event.target.value))} aria-label="Flight playback speed">{[0.25, 0.5, 1, 2, 4].map((value) => <option key={value} value={value}>{value}×</option>)}</select></label>
    </div>
  );
}

function FlightInspector({ model, frame, setCameraOffset }) {
  const cameraOffsetInput = (axis) => {
    const displayed = Number(model.cameraOffset[axis].toFixed(4));
    return <input
      key={`camera-${axis}-${displayed}`}
      type="number"
      step="0.005"
      defaultValue={displayed}
      aria-label={`Camera offset ${['X', 'Y', 'Z'][axis]}`}
      onBlur={(event) => {
        if (event.currentTarget.dataset.cancelled) { delete event.currentTarget.dataset.cancelled; return; }
        if (`${event.currentTarget.value}`.trim() === '' || !Number.isFinite(Number(event.currentTarget.value))) {
          event.currentTarget.value = displayed;
          return;
        }
        setCameraOffset((current) => current.map((value, index) => (index === axis ? Number(event.currentTarget.value) : value)));
      }}
      onKeyDown={(event) => {
        if (event.key === 'Enter') event.currentTarget.blur();
        if (event.key === 'Escape') {
          event.preventDefault();
          event.currentTarget.dataset.cancelled = 'true';
          event.currentTarget.value = displayed;
          event.currentTarget.blur();
        }
      }}
    />;
  };
  const supported = model.solution.feasible;
  return (
    <aside className="panel flight-inspector">
      <div className="panel-head flight-inspector-head" aria-live="polite" aria-atomic="true">
        <div><span className="eyebrow">FLIGHT SUPPORT</span><h2>{supported ? 'Main-grid route covered' : 'Main-grid gaps found'}</h2></div>
        <span className={supported ? 'success-pill' : 'failure-pill'}>{supported ? <CheckCircle2 size={12} /> : <AlertTriangle size={12} />}{supportPercent(model)}</span>
      </div>
      <div className="flight-inspector-scroll">
        <section className="flight-current-states">
          <h3>Swarm at {format(frame.time, 2)} s</h3>
          {frame.drones.map((state) => <div key={state.id}><i style={{ background: state.color }} /><b>{state.id}</b><span>{state.phase}</span><code>{format(state.position[0], 2)}, {format(state.position[1], 2)}, {format(state.position[2], 2)}</code></div>)}
        </section>

        <section className="flight-takeoff-editor">
          <div className="flight-section-title"><div><span className="eyebrow">AUTO-PLACED</span><h3>Launch / landing spots</h3></div></div>
          <p>Each drone owns one distinct home window and returns to the same computed body XY after its waypoints. {model.grid.shortRange?.tiles?.length ? 'Its short-range tile supports the vertical phases.' : 'Without configured short-range tiles, that vertical support remains a staging assumption.'}</p>
          <div role="list" aria-label="Computed launch and landing homes">
            {model.drones.map((drone) => <div className="home-row" key={drone.id} role="listitem" aria-label={`${drone.id}, body X ${format(drone.takeoff[0], 3)} metres, Y ${format(drone.takeoff[1], 3)} metres, home window ${drone.homeWindowKey}`}><b style={{ '--drone-color': drone.color }}>{drone.id}</b><code>{format(drone.takeoff[0], 3)}, {format(drone.takeoff[1], 3)}</code><span>↺ window {drone.homeWindowKey}</span></div>)}
          </div>
        </section>

        <section className="flight-camera-offset">
          <div className="flight-section-title"><div><span className="eyebrow">CAMERA EXTRINSIC</span><h3>Camera centre offset</h3></div><button onClick={() => setCameraOffset([...DEFAULT_CAMERA_OFFSET])}><RotateCcw size={12} /> Default</button></div>
          <p>Drone-frame FLU metres, rotated into the world by SFL yaw. Defaults to the current Lightbender orchestrator launch setting.</p>
          <div className="camera-offset-row">{['X', 'Y', 'Z'].map((axis, index) => <label key={axis}>{axis}{cameraOffsetInput(index)}</label>)}</div>
        </section>

        <section className="flight-window-list">
          <div className="flight-section-title"><div><span className="eyebrow">CONSTRAINED SET COVER</span><h3>{supported ? 'Required installation windows' : 'Partial installation windows'}</h3></div><strong>{model.requiredWindows.length}</strong></div>
          <p>{model.solution.feasible
            ? (model.solution.optimal ? 'Minimum certified for these assigned homes' : 'Best cover found before the solver limit')
            : 'No window set supports the full route; showing its home windows and supportable portions'} · {model.solution.preselected.length} home + {model.solution.added.length} additional · {model.solution.requirementCount} flight constraints · {model.requiredCellKeys.length} main-grid markers{model.grid.shortRange?.tiles?.length ? '; short-range tile markers are separate' : ''}.</p>
          <div className="window-rows" role="list" aria-label={supported ? 'Required installation windows' : 'Partial installation windows'}>
            {model.requiredWindows.map((window) => {
              const purpose = window.homeDroneIds.length ? `${window.homeDroneIds.join(', ')} home${window.supportsFlight ? ' and flight' : ''}` : 'flight';
              const signature = window.signature?.join(' · ') ?? 'IDs unavailable';
              return <div key={window.key} className="window-row" role="listitem" aria-label={`Rows ${window.row} through ${window.row + model.grid.windowSize - 1}, columns ${window.col} through ${window.col + model.grid.windowSize - 1}, marker IDs ${window.signature?.join(', ') ?? 'unavailable'}, ${purpose}`}><span>r{window.row}–{window.row + model.grid.windowSize - 1}</span><span>c{window.col}–{window.col + model.grid.windowSize - 1}</span><code title={signature}>{signature}</code><em>{window.homeDroneIds.length ? `${window.homeDroneIds.join(', ')} home${window.supportsFlight ? ' + flight' : ''}` : 'flight'}</em></div>;
            })}
          </div>
        </section>

        {(model.coverage.unsupportedSamples > 0 || model.coverage.unsupportedIntervals > 0 || model.coverage.unsupportedBelowRangeDuration > 0 || model.coverage.aboveRangeDuration > 0 || model.warnings.length > 0) && <section className="flight-warnings">
          {model.coverage.unsupportedSamples > 0 && <p><AlertTriangle size={13} /> {model.coverage.unsupportedSamples} sampled in-range poses cannot see a complete unique window.</p>}
          {model.coverage.unsupportedIntervals > 0 && <p><AlertTriangle size={13} /> {model.coverage.unsupportedIntervals} in-range path gap{model.coverage.unsupportedIntervals === 1 ? '' : 's'} cannot be certified with a continuously visible unique window.</p>}
          {model.coverage.unsupportedBelowRangeDuration > 0 && <p><AlertTriangle size={13} /> Post-takeoff flight spends {format(model.coverage.unsupportedBelowRangeDuration, 3)} drone-seconds below the main-grid working range.</p>}
          {model.coverage.aboveRangeDuration > 0 && <p><AlertTriangle size={13} /> Flight spends {format(model.coverage.aboveRangeDuration, 3)} drone-seconds above the main-grid working range.</p>}
          {model.warnings.map((warning) => <p key={warning}><AlertTriangle size={13} /> {warning}</p>)}
        </section>}
      </div>
    </aside>
  );
}

export default function FlightPlanner({ source, onOpenSfl }) {
  const [gridSource, setGridSource] = useState(BUILT_IN_GRID);
  const [gridError, setGridError] = useState('');
  const [cameraOffset, setCameraOffset] = useState([...DEFAULT_CAMERA_OFFSET]);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const gridInputRef = useRef(null);
  const parsedMission = useMemo(() => {
    try { return { raw: loadYaml(source.text), error: '' }; }
    catch (error) { return { raw: null, error: `Could not parse SFL YAML: ${error.message}` }; }
  }, [source]);
  const result = useMemo(() => {
    if (parsedMission.error) return { model: null, error: parsedMission.error };
    try { return { model: buildFlightModel(parsedMission.raw, gridSource.raw, {}, cameraOffset), error: '' }; }
    catch (error) { return { model: null, error: error.message || 'Could not model this flight.' }; }
  }, [parsedMission, gridSource, cameraOffset]);

  useEffect(() => { setFrameIndex(0); setPlaying(false); }, [result.model]);
  useEffect(() => {
    const keydown = (event) => {
      if (event.defaultPrevented || event.target.closest?.('button, input, select, textarea, summary, a, [contenteditable="true"]') || !result.model) return;
      if (event.code === 'Space') { event.preventDefault(); setPlaying((value) => !value); }
      if (event.key === 'ArrowLeft') { event.preventDefault(); setPlaying(false); setFrameIndex((value) => Math.max(0, value - 1)); }
      if (event.key === 'ArrowRight') { event.preventDefault(); setPlaying(false); setFrameIndex((value) => Math.min(result.model.frames.length - 1, value + 1)); }
    };
    window.addEventListener('keydown', keydown);
    return () => window.removeEventListener('keydown', keydown);
  }, [result.model]);

  const openGrid = async (file) => {
    if (!file) return;
    try {
      const raw = JSON.parse(await file.text());
      normalizeGrid(raw);
      setGridSource({ name: file.name, raw, builtIn: false });
      setGridError('');
    } catch (error) { setGridError(`Grid not loaded: ${error.message}`); }
  };

  if (!result.model) {
    return <section className="flight-page"><div className="panel flight-load-error" role="alert"><AlertTriangle size={22} /><div><span className="eyebrow">FLIGHT MODEL ERROR</span><h1>{source.name}</h1><p>{result.error}</p></div><div className="flight-load-error-actions">{!gridSource.builtIn && <button className="plain-button" onClick={() => setGridSource(BUILT_IN_GRID)}><Grid3X3 size={14} /> Built-in grid</button>}<button className="open-button" onClick={onOpenSfl}><FolderOpen size={15} /> Open another SFL</button></div></div></section>;
  }
  const model = result.model;
  const frame = model.frames[frameIndex] ?? model.frames[0];
  const hasShortRangeTiles = Boolean(model.grid.shortRange?.tiles?.length);
  return (
    <section className="flight-page">
      <div className="flight-runbar">
        <div className="flight-file-title"><FileJson size={18} /><div><strong>{source.name}</strong><span>{source.builtIn ? 'BUILT-IN EXAMPLE' : 'LOCAL SFL'} · {model.name.toUpperCase()}</span></div></div>
        <div className="flight-stats">
          <div><span>Drones</span><strong>{model.drones.length}</strong></div>
          <div><span>Duration</span><strong>{format(model.duration, 2)} s</strong></div>
          <div><span>Observed cells</span><strong>{model.observedCellKeys.length}</strong></div>
          <div><span>{model.solution.feasible ? 'Required windows' : 'Partial windows'}</span><strong>{model.requiredWindows.length}</strong></div>
          <div><span>Main-grid support</span><strong>{supportPercent(model)}</strong></div>
        </div>
        <div className="flight-run-actions">
          <button className="plain-button inline" onClick={() => gridInputRef.current?.click()}><Grid3X3 size={14} /> Grid JSON</button>
          <input ref={gridInputRef} hidden type="file" accept="application/json,.json" onChange={(event) => { openGrid(event.target.files?.[0]); event.target.value = ''; }} />
        </div>
      </div>
      {(gridError || model.coverage.shortRangeTakeoffDuration > 0) && <div className={gridError ? 'error-banner flight-banner' : 'flight-range-banner'} role={gridError ? 'alert' : undefined}>
        {gridError ? <><AlertTriangle size={14} /><span>{gridError}</span><button onClick={() => setGridError('')}>Dismiss</button></> : <><Grid3X3 size={14} /><span>{gridSource.name} · {model.grid.rows}×{model.grid.cols} at {format(model.grid.spacing, 3)} m · {hasShortRangeTiles ? 'vertical launch and landing use assigned short-range tiles' : 'vertical launch and landing are reserved at main-window homes'}; each horizontal return-to-home leg is certified on the main grid.</span></>}
      </div>}
      <div className="flight-workspace">
        <div className="panel flight-stage-panel">
          <div className="panel-head">
            <div><span className="eyebrow">FLOOR OBSERVATION</span><h1>Swarm coverage trace</h1></div>
            <div className="flight-legend"><span><i className="legend-observed" />observed cell</span><span><i className="legend-home-window" />home window</span><span><i className="legend-required-window" />additional window</span><span><i className="legend-takeoff" />launch / landing</span></div>
          </div>
          <div className="flight-stage"><FlightScene model={model} frameIndex={frameIndex} /></div>
          <FlightTransport {...{ model, frameIndex, setFrameIndex, playing, setPlaying, speed, setSpeed }} />
        </div>
        <FlightInspector model={model} frame={frame} setCameraOffset={setCameraOffset} />
      </div>
      <div className="flight-method-note">
        <CheckCircle2 size={14} /> Down-facing pinhole footprint from grid intrinsics · camera offset {model.cameraOffset.map((value) => format(value, 3)).join(', ')} m FLU · {model.frames.length.toLocaleString()} adaptive samples plus continuous between-sample certification · one distinct home window per drone plus solver-selected additional windows for every in-range outbound, waypoint, and return leg · SFL yaw uses controller radians · waypoint 0 is the animation start pose.
      </div>
    </section>
  );
}
