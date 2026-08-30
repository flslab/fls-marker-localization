import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity, AlertTriangle, Braces, Box, ChevronLeft, ChevronRight, FileJson,
  FolderOpen, Gauge, Image as ImageIcon, Info, Layers3, ListFilter, LoaderCircle,
  Pause, Play, RotateCcw, Search, ShieldCheck, SkipBack, SkipForward, SlidersHorizontal,
  Table2, Upload,
} from 'lucide-react';
import { createLogModel, flattenObject, formatNumber, formatRawTime } from './lib/logModel.js';
import { makeDemoLog } from './lib/demoLog.js';
import WorldScene from './components/WorldScene.jsx';
import ImagePlane from './components/ImagePlane.jsx';
import LineChart, { StatusLane } from './components/LineChart.jsx';
import FrameInspector from './components/FrameInspector.jsx';
import { KeyValueView, RawJsonView } from './components/DataViews.jsx';

const tabs = [
  { id: 'explore', label: 'Explore', icon: Activity },
  { id: 'frames', label: 'Frames', icon: Table2 },
  { id: 'run', label: 'Run data', icon: SlidersHorizontal },
  { id: 'raw', label: 'Raw JSON', icon: Braces },
];

function makeDemoModel() {
  return createLogModel(makeDemoLog(), 'marker_grid_demo.json');
}

function RunStats({ model }) {
  const stats = [
    ['Frames', model.frames.length.toLocaleString()],
    ['Duration', `${formatNumber(model.duration, 3)} s`],
    ['Effective FPS', formatNumber(model.fps, 2)],
    ['Valid poses', `${(model.poseRate * 100).toFixed(1)}%`],
    ['Marker IDs', model.markerIds.length.toLocaleString()],
  ];
  return <div className="stats">{stats.map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>;
}

function Transport({ model, selectedIndex, setSelectedIndex, playing, setPlaying, speed, setSpeed }) {
  const frame = model.frames[selectedIndex];
  const jumpPose = (direction) => {
    let index = selectedIndex + direction;
    while (index >= 0 && index < model.frames.length) {
      if (model.frames[index].poseValid) { setSelectedIndex(index); return; }
      index += direction;
    }
  };
  return (
    <div className="transport full-transport">
      <div className="transport-buttons">
        <button onClick={() => setSelectedIndex(0)} disabled={!selectedIndex} title="First frame"><SkipBack size={14} /></button>
        <button onClick={() => jumpPose(-1)} disabled={!selectedIndex} title="Previous valid pose"><ChevronLeft size={14} /></button>
        <button className="play-button" onClick={() => setPlaying((value) => !value)} disabled={model.frames.length < 2} aria-label={playing ? 'Pause playback' : 'Play log'}>{playing ? <Pause size={15} fill="currentColor" /> : <Play size={15} fill="currentColor" />}</button>
        <button onClick={() => jumpPose(1)} disabled={selectedIndex >= model.frames.length - 1} title="Next valid pose"><ChevronRight size={14} /></button>
        <button onClick={() => setSelectedIndex(Math.max(0, model.frames.length - 1))} disabled={selectedIndex >= model.frames.length - 1} title="Last frame"><SkipForward size={14} /></button>
      </div>
      <div className="timeline-copy"><span>FRAME <b>{frame?.frameId ?? '—'}</b> / {Math.max(0, model.frames.length - 1)}</span><strong>{formatNumber(frame?.t, 5)} s</strong></div>
      <input
        className="timeline-range"
        type="range"
        min="0"
        max={Math.max(0, model.frames.length - 1)}
        value={selectedIndex}
        onChange={(event) => setSelectedIndex(Number(event.target.value))}
        aria-label="Selected log frame"
        style={{ '--progress': `${model.frames.length > 1 ? (selectedIndex / (model.frames.length - 1)) * 100 : 0}%` }}
      />
      <label className="speed-control"><Gauge size={13} /><select value={speed} onChange={(event) => setSpeed(Number(event.target.value))} aria-label="Playback speed">{[0.25, 0.5, 1, 2, 4].map((value) => <option key={value} value={value}>{value}×</option>)}</select></label>
    </div>
  );
}

function ExploreView({ model, selectedIndex, setSelectedIndex, playing, setPlaying, speed, setSpeed }) {
  const [spatialView, setSpatialView] = useState('world');
  const [useFiltered, setUseFiltered] = useState(model.hasFiltered);
  const [markerId, setMarkerId] = useState(model.poseMarkerIds[0] ?? null);
  useEffect(() => { setUseFiltered(model.hasFiltered); setMarkerId(model.poseMarkerIds[0] ?? null); }, [model]);
  const poseForFrame = useCallback((frame) => {
    if (!frame) return null;
    return markerId === null
      ? frame.primary
      : (frame.poses.find((pose) => (pose.kind === 'legacy' || pose.kind === 'historical-marker') && pose.markerId === markerId) || null);
  }, [markerId]);
  const positionKey = useFiltered ? 'filteredPosition' : 'position';
  const positionSeries = useMemo(() => [
    { key: 'x', label: 'X', color: '#ff7777', get: (frame) => poseForFrame(frame)?.[positionKey]?.[0] },
    { key: 'y', label: 'Y', color: '#9df7c7', get: (frame) => poseForFrame(frame)?.[positionKey]?.[1] },
    { key: 'z', label: 'Z', color: '#61d9f4', get: (frame) => poseForFrame(frame)?.[positionKey]?.[2] },
  ], [poseForFrame, positionKey]);
  const orientationSeries = useMemo(() => [
    { key: 'roll', label: 'Roll', color: '#ff9f66', get: (frame) => poseForFrame(frame)?.orientation?.[0] },
    { key: 'pitch', label: 'Pitch', color: '#a889d8', get: (frame) => poseForFrame(frame)?.orientation?.[1] },
    { key: 'yaw', label: 'Yaw', color: '#61d9f4', get: (frame) => poseForFrame(frame)?.orientation?.[2] },
  ], [poseForFrame]);
  const qualitySeries = useMemo(() => [
    { key: 'error', label: 'Reproj. px', color: '#ff9f66', get: (frame) => frame.reprojectionError },
    { key: 'used', label: 'Markers used', color: '#9df7c7', dash: '5 4', get: (frame) => frame.counts.markersUsed },
    { key: 'matched', label: 'Map matched', color: '#61d9f4', dash: '2 4', get: (frame) => frame.counts.matched },
  ], []);
  const countSeries = useMemo(() => [
    { key: 'blobs', label: 'Blobs', color: '#60736c', get: (frame) => frame.counts.blobs },
    { key: 'tracked', label: 'Tracked', color: '#ff9f66', dash: '3 3', get: (frame) => frame.counts.tracked },
    { key: 'decoded', label: 'Eligible', color: '#61d9f4', get: (frame) => frame.counts.decoded },
    { key: 'accepted', label: 'Accepted', color: '#9df7c7', get: (frame) => frame.counts.accepted },
    { key: 'candidates', label: 'Candidates', color: '#a889d8', dash: '4 4', get: (frame) => frame.counts.candidates },
  ], []);
  const representativePose = poseForFrame(model.frames.find((frame) => poseForFrame(frame)?.position) || model.frames[0]);
  const positionTitle = representativePose?.entity === 'marker' || representativePose?.kind === 'historical-marker' ? 'Marker position' : 'Camera position';
  const orientationTitle = representativePose?.entity === 'marker' || representativePose?.kind === 'historical-marker' ? 'Marker orientation' : 'Camera orientation';
  const positionFrame = `${representativePose?.frameLabel || 'unknown frame'} · metres`;
  const spatialTitle = !model.hasPoseData
    ? 'No recoverable spatial pose'
    : (representativePose?.frameLabel === 'world frame' ? 'Camera + marker world' : `${positionTitle} · ${representativePose?.frameLabel || 'unknown frame'}`);

  return (
    <>
      <section className="workspace">
        <div className="panel stage-panel">
          <div className="panel-head">
            <div><span className="eyebrow">SPATIAL VIEW</span><h1>{spatialTitle}</h1></div>
            <div className="stage-actions">
              {model.poseMarkerIds.length > 1 && <label className="marker-select">Marker<select value={markerId ?? ''} onChange={(event) => setMarkerId(Number(event.target.value))} aria-label="Legacy pose marker ID">{model.poseMarkerIds.map((id) => <option key={id} value={id}>ID {id}</option>)}</select></label>}
              {model.hasFiltered && <button aria-pressed={useFiltered} className={`data-toggle ${useFiltered ? 'active' : ''}`} onClick={() => setUseFiltered((value) => !value)}>KF {useFiltered ? 'on' : 'off'}</button>}
              <div className="segmented" role="tablist" aria-label="Spatial visualization">
                <button role="tab" aria-selected={spatialView === 'world'} className={spatialView === 'world' ? 'selected' : ''} onClick={() => setSpatialView('world')}><Box size={14} />Spatial 3D</button>
                <button role="tab" aria-selected={spatialView === 'image'} className={spatialView === 'image' ? 'selected' : ''} onClick={() => setSpatialView('image')}><ImageIcon size={14} />Image plane</button>
              </div>
            </div>
          </div>
          <div className="stage live-stage">
            {spatialView === 'world' ? <WorldScene model={model} frameIndex={selectedIndex} useFiltered={useFiltered} markerId={markerId} /> : <ImagePlane model={model} frameIndex={selectedIndex} />}
          </div>
          <Transport {...{ model, selectedIndex, setSelectedIndex, playing, setPlaying, speed, setSpeed }} />
        </div>
        <FrameInspector model={model} frameIndex={selectedIndex} />
      </section>
      <section className="plot-section">
        <div className="plot-heading">
          <div><span className="eyebrow">SYNCHRONIZED SIGNALS</span><h2>Run telemetry</h2></div>
          <p>Hover to inspect · click any plot or status segment to select a frame</p>
        </div>
        <StatusLane frames={model.frames} selectedIndex={selectedIndex} onSelect={setSelectedIndex} />
        <div className="charts-grid">
          <LineChart title={positionTitle} subtitle={`${useFiltered ? 'Kalman-filtered · ' : 'raw · '}${positionFrame}`} frames={model.frames} series={positionSeries} selectedIndex={selectedIndex} onSelect={setSelectedIndex} />
          <LineChart title={orientationTitle} subtitle="roll / pitch / yaw · radians" frames={model.frames} series={orientationSeries} selectedIndex={selectedIndex} onSelect={setSelectedIndex} />
          <LineChart title="Pose quality" subtitle="pixels and marker count" frames={model.frames} series={qualitySeries} selectedIndex={selectedIndex} onSelect={setSelectedIndex} />
          <LineChart title="Detection pipeline" subtitle="records per frame" frames={model.frames} series={countSeries} selectedIndex={selectedIndex} onSelect={setSelectedIndex} />
        </div>
      </section>
    </>
  );
}

function FramesView({ model, selectedIndex, setSelectedIndex, setActiveTab }) {
  const [query, setQuery] = useState('');
  const [poseOnly, setPoseOnly] = useState(false);
  const [page, setPage] = useState(0);
  const pageSize = 100;
  const filtered = useMemo(() => model.frames.filter((frame) => {
    if (poseOnly && !frame.poseValid) return false;
    const needle = query.trim().toLowerCase();
    return !needle || `${frame.frameId} ${frame.status} ${frame.grid?.lookup_status || ''}`.toLowerCase().includes(needle);
  }), [model, query, poseOnly]);
  const maxPage = Math.max(0, Math.ceil(filtered.length / pageSize) - 1);
  const currentPage = Math.min(page, maxPage);
  const visible = filtered.slice(currentPage * pageSize, (currentPage + 1) * pageSize);
  const select = (index) => { setSelectedIndex(index); setActiveTab('explore'); };
  return (
    <section className="table-page panel">
      <div className="page-heading">
        <div><span className="eyebrow">FRAME INDEX</span><h1>Every captured sample</h1><p>{filtered.length.toLocaleString()} of {model.frames.length.toLocaleString()} frames</p></div>
        <div className="table-filters">
          <label className="search-field"><Search size={14} /><input aria-label="Search frames by ID or status" value={query} onChange={(event) => { setQuery(event.target.value); setPage(0); }} placeholder="Frame ID or status" /></label>
          <button aria-pressed={poseOnly} className={poseOnly ? 'filter-button active' : 'filter-button'} onClick={() => { setPoseOnly((value) => !value); setPage(0); }}><ListFilter size={14} />Valid poses</button>
        </div>
      </div>
      <div className="frame-table-wrap">
        <table className="frame-table">
          <thead><tr><th>Index</th><th>Frame ID</th><th>Elapsed</th><th>Raw time</th><th>Status</th><th>Poses</th><th>Blobs</th><th>Eligible</th><th>Accepted</th><th>Reproj. error</th></tr></thead>
          <tbody>{visible.map((frame) => (
            <tr key={frame.index} className={frame.index === selectedIndex ? 'selected' : ''} onClick={() => select(frame.index)} tabIndex="0" onKeyDown={(event) => event.key === 'Enter' && select(frame.index)}>
              <td>{frame.index}</td><td><b>{frame.frameId}</b></td><td>{formatNumber(frame.t, 6)} s</td><td>{formatRawTime(frame.rawTime)}</td>
              <td><span className={`table-status status-${frame.status}`}>{frame.status}</span></td><td>{formatNumber(frame.counts.poses, 0)}</td><td>{formatNumber(frame.counts.blobs, 0)}</td><td>{formatNumber(frame.counts.decoded, 0)}</td><td>{formatNumber(frame.counts.accepted, 0)}</td><td>{formatNumber(frame.reprojectionError, 5)}</td>
            </tr>
          ))}</tbody>
        </table>
        {!visible.length && <div className="table-empty big">No frames match these filters.</div>}
      </div>
      <div className="pagination"><button onClick={() => setPage((value) => Math.max(0, value - 1))} disabled={currentPage <= 0}><ChevronLeft size={14} />Previous</button><span>Page {currentPage + 1} of {maxPage + 1}</span><button onClick={() => setPage((value) => Math.min(maxPage, value + 1))} disabled={currentPage >= maxPage}>Next<ChevronRight size={14} /></button></div>
    </section>
  );
}

function RunDataView({ model }) {
  const argsCount = flattenObject(model.args).length;
  const configCount = flattenObject(model.config).length;
  return (
    <section className="run-data-page">
      <div className="page-heading standalone"><div><span className="eyebrow">RUN METADATA</span><h1>Capture inputs and resolved configuration</h1><p>Arguments and config remain separate so discrepancies stay visible.</p></div></div>
      <div className="metadata-grid">
        <section className="panel metadata-panel"><div className="metadata-head"><div><SlidersHorizontal size={16} /><strong>Arguments</strong></div><span>{argsCount} scalar fields</span></div><KeyValueView data={model.args} prefix="args" /></section>
        <section className="panel metadata-panel"><div className="metadata-head"><div><Layers3 size={16} /><strong>Resolved config</strong></div><span>{configCount} scalar fields</span></div><KeyValueView data={model.config} prefix="config" /></section>
      </div>
      <section className="panel provenance-panel">
        <div><Info size={17} /><strong>What the log does not embed</strong></div>
        <p>The grid ID matrix, ArUco marker size/map, distortion coefficients, and legacy marker geometry are not present in the output log. Positions and orientations use logged values; camera frustums and marker glyphs are schematic and not to scale.</p>
      </section>
    </section>
  );
}

export default function App() {
  const [model, setModel] = useState(makeDemoModel);
  const [isDemo, setIsDemo] = useState(true);
  const [activeTab, setActiveTab] = useState('explore');
  const [selectedIndex, setSelectedIndex] = useState(120);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

  const loadFile = async (file) => {
    if (!file) return;
    setLoading(true); setError(''); setPlaying(false);
    try {
      const text = await file.text();
      let raw;
      try { raw = JSON.parse(text); } catch (parseError) { throw new Error(`Could not parse JSON: ${parseError.message}`); }
      const nextModel = createLogModel(raw, file.name);
      setModel(nextModel);
      setIsDemo(false);
      const firstPose = nextModel.frames.findIndex((frame) => frame.poseValid);
      setSelectedIndex(firstPose >= 0 ? firstPose : 0);
      setActiveTab('explore');
    } catch (nextError) {
      setError(nextError.message || 'Could not open this log.');
    } finally { setLoading(false); }
  };
  const resetDemo = () => {
    const demo = makeDemoModel();
    setModel(demo); setIsDemo(true); setSelectedIndex(120); setPlaying(false); setError(''); setActiveTab('explore');
  };

  useEffect(() => {
    if (!playing || model.frames.length < 2) return undefined;
    let last = performance.now();
    let carry = 0;
    let animationFrame;
    const tick = (now) => {
      const elapsed = Math.min(0.25, (now - last) / 1000);
      last = now;
      carry += elapsed * Math.max(1, model.fps || 30) * speed;
      const steps = Math.floor(carry);
      carry -= steps;
      if (steps) {
        setSelectedIndex((current) => {
          const next = current + steps;
          if (next >= model.frames.length - 1) { setPlaying(false); return model.frames.length - 1; }
          return next;
        });
      }
      animationFrame = requestAnimationFrame(tick);
    };
    animationFrame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animationFrame);
  }, [playing, model, speed]);

  useEffect(() => {
    const keydown = (event) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement || event.target instanceof HTMLTextAreaElement) return;
      if (!model.frames.length) return;
      if (event.code === 'Space' && model.frames.length > 1) { event.preventDefault(); setPlaying((value) => !value); }
      if (event.key === 'ArrowLeft') { event.preventDefault(); setSelectedIndex((value) => Math.max(0, value - 1)); }
      if (event.key === 'ArrowRight') { event.preventDefault(); setSelectedIndex((value) => Math.min(Math.max(0, model.frames.length - 1), value + 1)); }
      if (event.key === 'Home') { event.preventDefault(); setSelectedIndex(0); }
      if (event.key === 'End') { event.preventDefault(); setSelectedIndex(Math.max(0, model.frames.length - 1)); }
    };
    window.addEventListener('keydown', keydown);
    return () => window.removeEventListener('keydown', keydown);
  }, [model.frames.length]);

  const tabContent = activeTab === 'explore'
    ? <ExploreView {...{ model, selectedIndex, setSelectedIndex, playing, setPlaying, speed, setSpeed }} />
    : activeTab === 'frames'
      ? <FramesView {...{ model, selectedIndex, setSelectedIndex, setActiveTab }} />
      : activeTab === 'run'
        ? <RunDataView model={model} />
        : <RawJsonView model={model} selectedFrameIndex={selectedIndex} />;

  return (
    <main
      className="app-shell"
      onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setDragging(false); }}
      onDrop={(event) => { event.preventDefault(); setDragging(false); loadFile(event.dataTransfer.files?.[0]); }}
    >
      <header className="topbar">
        <div className="brand"><span className="brand-mark"><span /></span><div><b>FLS</b><span>POSE SCOPE</span></div></div>
        <nav aria-label="Viewer sections">{tabs.map(({ id, label, icon: Icon }) => <button key={id} className={activeTab === id ? 'nav-active' : ''} onClick={() => setActiveTab(id)}><Icon size={13} />{label}</button>)}</nav>
        <button className="open-button" onClick={() => fileInputRef.current?.click()}><FolderOpen size={16} /> Open log</button>
        <input ref={fileInputRef} className="visually-hidden" type="file" accept="application/json,.json" onChange={(event) => { loadFile(event.target.files?.[0]); event.target.value = ''; }} />
      </header>

      <section className="runbar">
        <div className="file-title"><FileJson size={18} /><div><strong title={model.fileName}>{model.fileName}</strong><span>{isDemo ? 'DEMO DATA' : 'LOCAL LOG'} · {model.mode.toUpperCase()}</span></div></div>
        <RunStats model={model} />
        <div className="run-actions"><span className="local-pill"><ShieldCheck size={13} />stays on device</span><button className="plain-button" onClick={resetDemo}><RotateCcw size={15} /> Demo</button></div>
      </section>

      {model.warnings.length > 0 && <button className="warning-banner" onClick={() => setActiveTab('raw')}><AlertTriangle size={15} /><span>{model.warnings.length} schema warning{model.warnings.length === 1 ? '' : 's'} · {model.warnings[0]}</span><b>Inspect raw JSON</b></button>}
      {error && <div className="error-banner" role="alert"><AlertTriangle size={16} /><div><strong>Log not loaded</strong><span>{error}</span></div><button onClick={() => setError('')}>Dismiss</button></div>}
      {tabContent}

      {dragging && <div className="drop-overlay"><div><Upload size={28} /><strong>Drop output log</strong><span>JSON is parsed locally in this browser</span></div></div>}
      {loading && <div className="loading-overlay" role="status"><LoaderCircle size={26} className="spin" /><strong>Parsing log…</strong><span>Large diagnostic runs can take a moment.</span></div>}
    </main>
  );
}
