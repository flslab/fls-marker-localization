import { useMemo, useState } from 'react';

function PointLabel({ x, y, text, className, shape = 'circle' }) {
  return (
    <g className={className}>
      {shape === 'square' ? <rect x={x - 5} y={y - 5} width="10" height="10" rx="2" /> : <circle cx={x} cy={y} r="5" />}
      <text x={x + 8} y={y - 7}>{text}</text>
    </g>
  );
}

export default function ImagePlane({ model, frameIndex }) {
  const frame = model.frames[frameIndex];
  const [layers, setLayers] = useState({ blobs: true, decoded: true, relative: false, matched: true });
  const data = useMemo(() => ({
    blobs: frame?.blobs || [],
    decoded: frame?.grid?.decoded_tracks || [],
    relative: frame?.grid?.relative_markers || [],
    matched: frame?.grid?.matched_markers || [],
  }), [frame]);
  const configuredResolution = model.config?.marker_grid?.image_resolution_pixels;
  const argumentResolution = [model.args?.cam_width, model.args?.cam_height];
  const resolution = [configuredResolution, argumentResolution].find((candidate) => (
    Number(candidate?.[0]) > 0 && Number(candidate?.[1]) > 0
  ));
  const hasLoggedResolution = Boolean(resolution);
  const derivedExtent = useMemo(() => {
    const points = [
      ...data.blobs.map((item) => [item.x, item.y]),
      ...data.decoded.map((item) => [item.image_x, item.image_y]),
      ...data.relative.map((item) => [item.image_x, item.image_y]),
      ...data.matched.map((item) => [item.image_x, item.image_y]),
    ].filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
    if (!points.length) return [1, 1];
    return [Math.max(...points.map(([x]) => x)) + 16, Math.max(...points.map(([, y]) => y)) + 16];
  }, [data]);
  const width = hasLoggedResolution ? Number(resolution[0]) : derivedExtent[0];
  const height = hasLoggedResolution ? Number(resolution[1]) : derivedExtent[1];
  const toggle = (key) => setLayers((current) => ({ ...current, [key]: !current[key] }));

  return (
    <div className="image-plane-wrap">
      <div className="layer-controls" aria-label="Image overlay layers">
        {Object.entries(layers).map(([key, enabled]) => (
          <button key={key} aria-pressed={enabled} className={enabled ? 'active' : ''} onClick={() => toggle(key)}>
            <i className={`layer-${key}`} />{key}
          </button>
        ))}
      </div>
      <svg className="image-plane" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={hasLoggedResolution ? `Image plane ${width} by ${height} pixels with detection overlays` : 'Detection overlay using bounds derived from logged image coordinates; sensor resolution was not logged'}>
        <defs>
          <pattern id="sensor-grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#173128" strokeWidth="1" />
          </pattern>
        </defs>
        <rect width={width} height={height} fill="#081512" />
        <rect width={width} height={height} fill="url(#sensor-grid)" />
        {hasLoggedResolution && <>
          <line x1={width / 2} y1="0" x2={width / 2} y2={height} className="sensor-crosshair" />
          <line x1="0" y1={height / 2} x2={width} y2={height / 2} className="sensor-crosshair" />
        </>}
        {layers.blobs && data.blobs.map((blob, index) => (
          <PointLabel key={`b-${index}`} x={blob.x} y={blob.y} text={blob.id === -1 ? '?' : String(blob.id)} className="point-blob" />
        ))}
        {layers.decoded && data.decoded.map((track, index) => (
          <PointLabel key={`d-${index}`} x={track.image_x} y={track.image_y} text={String(track.id)} className={track.visible ? 'point-decoded' : 'point-stale'} />
        ))}
        {layers.relative && data.relative.map((marker, index) => (
          <PointLabel key={`r-${index}`} x={marker.image_x} y={marker.image_y} text={`${marker.relative_row},${marker.relative_col}`} className={marker.accepted ? 'point-relative' : 'point-rejected'} shape="square" />
        ))}
        {layers.matched && data.matched.map((marker, index) => (
          <g key={`m-${index}`} className="point-matched">
            <circle cx={marker.image_x} cy={marker.image_y} r="10" />
            <text x={marker.image_x + 12} y={marker.image_y + 13}>
              {marker.grid_type === 'short_range'
                ? `tile ${marker.tile_i},${marker.tile_j} · local ${marker.local_i},${marker.local_j}`
                : `[${marker.map_row},${marker.map_col}]`}
            </text>
          </g>
        ))}
        {hasLoggedResolution && <>
          <text x="12" y="22" className="sensor-label">0,0</text>
          <text x={width - 12} y={height - 12} textAnchor="end" className="sensor-label">{width} × {height} px</text>
        </>}
      </svg>
      {!hasLoggedResolution && <p className="resolution-note">Viewport derived from logged coordinates · sensor resolution unavailable</p>}
      {!data.blobs.length && !data.decoded.length && !data.matched.length && (
        <div className="plane-empty"><strong>No image-space detections</strong><span>This frame contains no logged blobs or marker tracks.</span></div>
      )}
    </div>
  );
}
