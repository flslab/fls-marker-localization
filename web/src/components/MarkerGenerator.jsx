import { useMemo, useRef, useState, useEffect } from 'react';
import {
  Crosshair, Download, Grid3X3, Minus, MousePointer2, Plus, Radio, RotateCcw, Trash2, X,
} from 'lucide-react';
import {
  buildMarkerGridFile, DEFAULT_MARKER_GRID_CONFIG, hypergridMarkersForTile, markerGridJson, MYGRID_EVENT_ACTIONS, normalizeMarkerGridConfig,
} from '../lib/markerGrid.js';

const tileKey = (i, j) => `${i}:${j}`;
const format = (value, digits = 3) => Number.isFinite(value) ? Number(value.toFixed(digits)).toString() : '—';
let nextGridEventId = 1;

function NumberField({ label, value, onChange, step = 'any', min, max, unit }) {
  return (
    <label className="generator-field">
      <span>{label}</span>
      <span className="generator-input-wrap">
        <input type="number" value={value} step={step} min={min} max={max} onChange={(event) => onChange(event.target.value === '' ? '' : Number(event.target.value))} />
        {unit && <small>{unit}</small>}
      </span>
    </label>
  );
}

function GridCanvas({ tiles, config, generatedFile, selectedKey, onToggle }) {
  const wrapRef = useRef(null);
  const dragRef = useRef(null);
  const [size, setSize] = useState({ width: 900, height: 620 });
  const [view, setView] = useState({ x: 0, y: 0, scale: 92 });
  const [selectedMarker, setSelectedMarker] = useState(null);

  useEffect(() => {
    if (!wrapRef.current) return undefined;
    const observer = new ResizeObserver(([entry]) => {
      setSize({ width: Math.max(1, entry.contentRect.width), height: Math.max(1, entry.contentRect.height) });
    });
    observer.observe(wrapRef.current);
    return () => observer.disconnect();
  }, []);

  const screenToTile = (clientX, clientY) => {
    const rect = wrapRef.current.getBoundingClientRect();
    const x = view.x + (clientX - rect.left - size.width / 2) / view.scale;
    const y = view.y - (clientY - rect.top - size.height / 2) / view.scale;
    return { i: Math.floor(x + 0.5), j: Math.floor(y + 0.5) };
  };

  const selected = useMemo(() => new Map(tiles.map((tile) => [tileKey(tile.i, tile.j), tile])), [tiles]);
  const generatedTiles = useMemo(() => new Map((generatedFile?.mygrid.tiles ?? []).map((tile) => [tileKey(tile.i, tile.j), tile])), [generatedFile]);
  const bounds = useMemo(() => ({
    minI: Math.floor(view.x - size.width / view.scale / 2) - 1,
    maxI: Math.ceil(view.x + size.width / view.scale / 2) + 1,
    minJ: Math.floor(view.y - size.height / view.scale / 2) - 1,
    maxJ: Math.ceil(view.y + size.height / view.scale / 2) + 1,
  }), [view, size]);
  const visibleTiles = [];
  for (let i = bounds.minI; i <= bounds.maxI; i += 1) {
    for (let j = bounds.minJ; j <= bounds.maxJ; j += 1) visibleTiles.push({ i, j });
  }
  const originX = size.width / 2 - view.x * view.scale;
  const originY = size.height / 2 + view.y * view.scale;
  const mod = (value, divisor) => ((value % divisor) + divisor) % divisor;

  const markerDetails = useMemo(() => {
    if (!selectedMarker) return null;
    if (selectedMarker.kind === 'hypergrid') {
      const markers = hypergridMarkersForTile(config, selectedMarker.tileI, selectedMarker.tileJ);
      const marker = markers.find((candidate) => candidate.local_grid_coordinates[0] === selectedMarker.localRow && candidate.local_grid_coordinates[1] === selectedMarker.localCol);
      return marker && { ...marker, kind: 'HyperGrid', id: null };
    }
    const tile = generatedTiles.get(tileKey(selectedMarker.tileI, selectedMarker.tileJ));
    const marker = tile?.markers.find((candidate) => candidate.signature_index === selectedMarker.signatureIndex);
    return marker && { ...marker, kind: 'MyGrid' };
  }, [selectedMarker, config, generatedTiles]);

  const chooseMarker = (event, marker) => {
    event.stopPropagation();
    dragRef.current = null;
    setSelectedMarker(marker);
  };

  const zoomAtCenter = (factor) => setView((current) => ({ ...current, scale: Math.min(240, Math.max(24, current.scale * factor)) }));

  return (
    <div className="generator-canvas" ref={wrapRef}>
      <svg
        width={size.width}
        height={size.height}
        role="application"
        aria-label="Infinite HyperGrid tile selector. Drag to pan, use the mouse wheel to zoom, and click a tile to toggle its MyGrid."
        tabIndex="0"
        onWheel={(event) => {
          event.preventDefault();
          const rect = wrapRef.current.getBoundingClientRect();
          const sx = event.clientX - rect.left;
          const sy = event.clientY - rect.top;
          const beforeX = view.x + (sx - size.width / 2) / view.scale;
          const beforeY = view.y - (sy - size.height / 2) / view.scale;
          const scale = Math.min(240, Math.max(24, view.scale * Math.exp(-event.deltaY * 0.0015)));
          setView({
            x: beforeX - (sx - size.width / 2) / scale,
            y: beforeY + (sy - size.height / 2) / scale,
            scale,
          });
        }}
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          dragRef.current = { x: event.clientX, y: event.clientY, startX: event.clientX, startY: event.clientY, moved: false };
        }}
        onPointerMove={(event) => {
          const drag = dragRef.current;
          if (!drag) return;
          const dx = event.clientX - drag.x;
          const dy = event.clientY - drag.y;
          if (Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) > 4) drag.moved = true;
          drag.x = event.clientX;
          drag.y = event.clientY;
          setView((current) => ({ ...current, x: current.x - dx / current.scale, y: current.y + dy / current.scale }));
        }}
        onPointerUp={(event) => {
          const drag = dragRef.current;
          dragRef.current = null;
          if (!drag?.moved) {
            const coordinate = screenToTile(event.clientX, event.clientY);
            onToggle(coordinate.i, coordinate.j);
          }
        }}
        onPointerCancel={() => { dragRef.current = null; }}
      >
        <defs>
          <pattern id="generator-tile-grid" width={view.scale} height={view.scale} x={mod(originX - view.scale / 2, view.scale)} y={mod(originY - view.scale / 2, view.scale)} patternUnits="userSpaceOnUse">
            <path d={`M ${view.scale} 0 L 0 0 0 ${view.scale}`} className="generator-grid-line" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" className="generator-canvas-bg" />
        <rect width="100%" height="100%" fill="url(#generator-tile-grid)" />

        {visibleTiles.map(({ i, j }) => {
          const key = tileKey(i, j);
          const tile = selected.get(key);
          const generatedTile = generatedTiles.get(key);
          const cx = originX + i * view.scale;
          const cy = originY - j * view.scale;
          const hyperMarkers = hypergridMarkersForTile(config, i, j);
          const myHalfStep = Number(config.mygridSpacing) / (Number(config.hypergridSpacing) * 2) * view.scale / 2;
          return (
            <g key={key} className={`generator-tile${tile ? ' has-mygrid' : ''}${selectedKey === key ? ' active' : ''}`}>
              {tile && <rect x={cx - view.scale / 2 + 2} y={cy - view.scale / 2 + 2} width={view.scale - 4} height={view.scale - 4} rx="5" className="generator-mygrid-tile" />}
              {view.scale >= 66 && <text x={cx - view.scale * 0.41} y={cy + view.scale * 0.42} className="generator-coordinate">{i},{j}</text>}
              {hyperMarkers.map((marker) => {
                const markerIdentity = { kind: 'hypergrid', tileI: i, tileJ: j, localRow: marker.local_grid_coordinates[0], localCol: marker.local_grid_coordinates[1] };
                const markerKey = `hyper:${i}:${j}:${marker.grid_coordinates.join(':')}`;
                const isSelected = selectedMarker?.kind === markerIdentity.kind && selectedMarker.tileI === i && selectedMarker.tileJ === j && selectedMarker.localRow === markerIdentity.localRow && selectedMarker.localCol === markerIdentity.localCol;
                return <circle key={markerKey} cx={cx + (markerIdentity.localCol - 0.5) * view.scale / 2} cy={cy - (0.5 - markerIdentity.localRow) * view.scale / 2} r={view.scale > 55 ? 3.2 : 2.2} className={`generator-hyper-marker selectable${isSelected ? ' selected' : ''}`} role="button" tabIndex="0" aria-label={`HyperGrid marker ${marker.grid_coordinates.join(', ')}`} onPointerDown={(event) => event.stopPropagation()} onPointerUp={(event) => chooseMarker(event, markerIdentity)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') chooseMarker(event, markerIdentity); }} />;
              })}
              {tile && generatedTile && <g className="generator-mygrid">
                <path d={`M ${cx - myHalfStep} ${cy - myHalfStep} H ${cx + myHalfStep} V ${cy + myHalfStep} H ${cx - myHalfStep} Z`} className="generator-mygrid-layout" />
                {generatedTile.markers.map((marker) => {
                  const markerIdentity = { kind: 'mygrid', tileI: i, tileJ: j, signatureIndex: marker.signature_index };
                  const isSelected = selectedMarker?.kind === markerIdentity.kind && selectedMarker.tileI === i && selectedMarker.tileJ === j && selectedMarker.signatureIndex === markerIdentity.signatureIndex;
                  return <circle key={`my:${key}:${marker.signature_index}`} cx={cx + marker.offset[0] / (Number(config.hypergridSpacing) * 2) * view.scale} cy={cy - marker.offset[1] / (Number(config.hypergridSpacing) * 2) * view.scale} r={view.scale > 55 ? 3.7 : 2.5} className={`generator-my-marker selectable${isSelected ? ' selected' : ''}`} role="button" tabIndex="0" aria-label={`MyGrid marker ID ${marker.id}, tile ${i}, ${j}, local ${marker.local_row}, ${marker.local_col}`} onPointerDown={(event) => event.stopPropagation()} onPointerUp={(event) => chooseMarker(event, markerIdentity)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') chooseMarker(event, markerIdentity); }} />;
                })}
              </g>}
            </g>
          );
        })}

        <g className="generator-origin" transform={`translate(${originX} ${originY})`}>
          <circle r="5" />
          <line x1="0" y1="0" x2="48" y2="0" className="origin-x" />
          <path d="M48 0 l-7 -4 v8 z" className="origin-x-fill" />
          <text x="54" y="4" className="origin-x-text">X</text>
          <line x1="0" y1="0" x2="0" y2="-48" className="origin-y" />
          <path d="M0 -48 l-4 7 h8 z" className="origin-y-fill" />
          <text x="-4" y="-55" className="origin-y-text">Y</text>
          <line x1="0" y1="0" x2="-28" y2="28" className="origin-z" />
          <path d="M-28 28 l3 -8 5 5 z" className="origin-z-fill" />
          <text x="-41" y="42" className="origin-z-text">Z</text>
        </g>
      </svg>
      <div className="generator-canvas-tools">
        <button title="Zoom out" onClick={() => zoomAtCenter(0.8)}><Minus size={14} /></button>
        <button title="Reset view" onClick={() => setView({ x: 0, y: 0, scale: 92 })}><Crosshair size={14} /></button>
        <button title="Zoom in" onClick={() => zoomAtCenter(1.25)}><Plus size={14} /></button>
      </div>
      {markerDetails && <aside className="generator-marker-inspector" aria-live="polite">
        <button onClick={() => setSelectedMarker(null)} aria-label="Close marker coordinates"><X size={12} /></button>
        <span className="eyebrow">SELECTED MARKER</span>
        <h3>{markerDetails.kind}{markerDetails.id === null ? '' : ` · ID ${markerDetails.id}`}</h3>
        <dl>
          {markerDetails.kind === 'HyperGrid' ? <><dt>Grid</dt><dd>{markerDetails.grid_coordinates.join(', ')}</dd><dt>Tile</dt><dd>{markerDetails.tile_coordinates.join(', ')}</dd></> : <><dt>Tile grid</dt><dd>{markerDetails.tile_coordinates.join(', ')}</dd><dt>Local grid</dt><dd>{markerDetails.local_grid_coordinates.join(', ')}</dd></>}
          <dt>Global XYZ</dt><dd>{markerDetails.global_position.map((value) => format(value, 6)).join(', ')}</dd>
        </dl>
      </aside>}
      <div className="generator-canvas-hint"><MousePointer2 size={12} />Click tile to add/remove · click marker for coordinates · drag to pan</div>
      <div className="generator-axis-legend"><i className="x" />X <i className="y" />Y <i className="z" />Z</div>
    </div>
  );
}

const EVENT_LABELS = { on: 'Turn on', off: 'Turn off', static: 'Show static', blinking: 'Show blinking IDs' };

function AnimationEditor({ tile, updateEvents }) {
  const events = tile.events ?? [];
  const updateEvent = (id, patch) => updateEvents(events.map((event) => event.id === id ? { ...event, ...patch } : event));
  return (
    <section className="generator-timing">
      <div className="generator-section-head"><div><span className="eyebrow">BLENDER ANIMATION</span><h3>MyGrid ({tile.i}, {tile.j})</h3></div><button className="generator-add-event" onClick={() => updateEvents([...events, { id: nextGridEventId++, time: 0, action: 'blinking' }])}><Plus size={12} />Event</button></div>
      <p>MyGrids start off in blinking mode. Turn on resumes the current mode; static or blinking selects a mode and powers on.</p>
      {!events.length && <div className="generator-no-events">No animation · this MyGrid stays off</div>}
      {!!events.length && <div className="generator-event-list">
        {[...events].sort((a, b) => Number(a.time) - Number(b.time)).map((event) => <div key={event.id}>
          <label><span>Time</span><span className="generator-input-wrap"><input type="number" min="0" step="0.1" value={event.time} onChange={(change) => updateEvent(event.id, { time: change.target.value === '' ? '' : Number(change.target.value) })} /><small>s</small></span></label>
          <label><span>Action</span><select value={event.action} onChange={(change) => updateEvent(event.id, { action: change.target.value })}>{MYGRID_EVENT_ACTIONS.map((action) => <option key={action} value={action}>{EVENT_LABELS[action]}</option>)}</select></label>
          <button className="generator-remove-event" onClick={() => updateEvents(events.filter((candidate) => candidate.id !== event.id))} aria-label={`Remove event at ${event.time} seconds`}><Trash2 size={12} /></button>
        </div>)}
      </div>}
    </section>
  );
}

export default function MarkerGenerator() {
  const [config, setConfig] = useState({ ...DEFAULT_MARKER_GRID_CONFIG, origin: [...DEFAULT_MARKER_GRID_CONFIG.origin] });
  const [tiles, setTiles] = useState([]);
  const [selectedKey, setSelectedKey] = useState(null);
  const generated = useMemo(() => {
    try { return { file: buildMarkerGridFile(config, tiles), error: '' }; }
    catch (error) { return { file: null, error: error.message }; }
  }, [config, tiles]);
  const visualConfig = useMemo(() => {
    try { return normalizeMarkerGridConfig(config); }
    catch { return normalizeMarkerGridConfig(DEFAULT_MARKER_GRID_CONFIG); }
  }, [config]);
  const generatedTilesByKey = useMemo(() => new Map((generated.file?.mygrid.tiles ?? []).map((tile) => [tileKey(tile.i, tile.j), tile])), [generated.file]);
  const selectedTile = tiles.find((tile) => tileKey(tile.i, tile.j) === selectedKey) ?? null;
  const payloadBits = generated.file?.encoding.payload_bits ?? '—';
  const delimiterBits = generated.file?.encoding.delimiter_bits ?? '—';

  const updateConfig = (key, value) => setConfig((current) => ({ ...current, [key]: value }));
  const updateOrigin = (index, value) => setConfig((current) => ({ ...current, origin: current.origin.map((item, itemIndex) => itemIndex === index ? value : item) }));
  const toggleTile = (i, j) => {
    const key = tileKey(i, j);
    const exists = tiles.some((tile) => tileKey(tile.i, tile.j) === key);
    if (exists) {
      setTiles((current) => current.filter((tile) => tileKey(tile.i, tile.j) !== key));
      if (selectedKey === key) setSelectedKey(null);
    } else {
      setTiles((current) => [...current, { i, j, events: [] }]);
      setSelectedKey(key);
    }
  };
  const download = () => {
    if (!generated.file) return;
    const blob = new Blob([markerGridJson(config, tiles)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'hypergrid-mygrid.json';
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="generator-page">
      <header className="generator-heading">
        <div><span className="eyebrow">MARKER MAP AUTHORING</span><h1>Infinite grid, precise local identity.</h1><p>Set the global HyperGrid once, then place cyclic MyGrid signatures exactly where short-range localization is needed.</p></div>
        <button className="generator-download" onClick={download} disabled={!generated.file}><Download size={16} />Download grid file</button>
      </header>

      <div className="generator-layout">
        <aside className="panel generator-sidebar">
          <section>
            <div className="generator-section-head"><div><span className="eyebrow">01 · GEOMETRY</span><h2>HyperGrid</h2></div><Grid3X3 size={18} /></div>
            <p>The grid is infinite. Only its marker spacing and world origin are exported. Every tile is exactly twice the spacing.</p>
            <NumberField label="Marker spacing" value={config.hypergridSpacing} min="0.000001" step="0.005" onChange={(value) => updateConfig('hypergridSpacing', value)} unit="m" />
            <div className="generator-derived"><span>Tile size</span><strong>{format(Number(config.hypergridSpacing) * 2)} m</strong></div>
            <span className="generator-subtitle">Origin · centre of tile (0, 0)</span>
            <div className="generator-origin-inputs">{['X', 'Y', 'Z'].map((axis, index) => <NumberField key={axis} label={axis} value={config.origin[index]} step="0.05" onChange={(value) => updateOrigin(index, value)} unit="m" />)}</div>
          </section>

          <section>
            <div className="generator-section-head"><div><span className="eyebrow">02 · IDENTITY</span><h2>MyGrid encoding</h2></div><Radio size={18} /></div>
            <NumberField label="Maximum IDs" value={config.maxIds} min="2" max="65536" step="1" onChange={(value) => updateConfig('maxIds', value)} />
            <NumberField label="Marker spacing" value={config.mygridSpacing} min="0.000001" step="0.001" onChange={(value) => updateConfig('mygridSpacing', value)} unit="m" />
            <NumberField label="Blink frequency" value={config.blinkFrequencyHz} min="0.001" step="1" onChange={(value) => updateConfig('blinkFrequencyHz', value)} unit="Hz" />
            <div className="generator-encoding">
              <div><span>Payload</span><strong>{payloadBits} bits</strong></div>
              <b>+</b>
              <div><span>Delimiter</span><strong>{delimiterBits} × 1, then 0</strong></div>
            </div>
            <p className="generator-note">Delimiter length is payload + 1, so it cannot occur entirely inside an ID payload.</p>
          </section>

          <section>
            <div className="generator-section-head"><div><span className="eyebrow">03 · BLENDER</span><h2>Scene markers</h2></div></div>
            <div className="generator-two-fields">
              <NumberField label="HyperGrid Ø" value={config.hypergridMarkerDiameter} min="0.000001" step="0.001" onChange={(value) => updateConfig('hypergridMarkerDiameter', value)} unit="m" />
              <NumberField label="MyGrid Ø" value={config.mygridMarkerDiameter} min="0.000001" step="0.001" onChange={(value) => updateConfig('mygridMarkerDiameter', value)} unit="m" />
            </div>
            <NumberField label="Scene frame rate" value={config.sceneFps} min="1" max="1000" step="1" onChange={(value) => updateConfig('sceneFps', value)} unit="fps" />
          </section>

          <section className="generator-selection-section">
            <div className="generator-section-head"><div><span className="eyebrow">04 · SELECTED</span><h2>MyGrid tiles</h2></div><strong className="generator-count">{tiles.length}</strong></div>
            {!tiles.length && <p className="generator-empty">Click any tile on the canvas to give it a MyGrid.</p>}
            {!!tiles.length && <div className="generator-tile-list">{[...tiles].sort((a, b) => a.i - b.i || a.j - b.j).map((tile) => {
              const key = tileKey(tile.i, tile.j);
              const signature = generatedTilesByKey.get(key)?.signature;
              const signatureLabel = signature?.join(' · ') ?? 'unavailable';
              return <button key={key} className={selectedKey === key ? 'active' : ''} onClick={() => setSelectedKey(key)} aria-label={`Tile ${tile.i}, ${tile.j}, signature ${signatureLabel}`}><i /><span>Tile</span><b>{tile.i}, {tile.j}</b><code title={`Cyclic signature: ${signatureLabel}`}>{signatureLabel}</code><Trash2 size={12} onClick={(event) => { event.stopPropagation(); toggleTile(tile.i, tile.j); }} /></button>;
            })}</div>}
          </section>

          {selectedTile && <AnimationEditor tile={selectedTile} updateEvents={(events) => setTiles((current) => current.map((tile) => tileKey(tile.i, tile.j) === selectedKey ? { ...tile, events } : tile))} />}
        </aside>

        <section className="panel generator-stage-panel">
          <div className="generator-stage-head">
            <div><span className="eyebrow">TILE CANVAS</span><h2>Place MyGrids</h2></div>
            <div className="generator-stage-meta"><span><i className="hyper" />HyperGrid · 4 per tile</span><span><i className="my" />MyGrid · centered 2×2</span></div>
          </div>
          <GridCanvas tiles={tiles} config={visualConfig} generatedFile={generated.file} selectedKey={selectedKey} onToggle={toggleTile} />
          <footer className="generator-summary">
            <div><span>Origin</span><strong>{config.origin.map((value) => format(Number(value))).join(', ')}</strong></div>
            <div><span>Selected MyGrids</span><strong>{tiles.length}</strong></div>
            <div><span>Signature rule</span><strong>Unique under rotation</strong></div>
            <button onClick={() => { setTiles([]); setSelectedKey(null); }} disabled={!tiles.length}><RotateCcw size={13} />Clear tiles</button>
          </footer>
          {generated.error && <div className="generator-error" role="alert">{generated.error}</div>}
        </section>
      </div>
    </section>
  );
}
