import { useMemo, useRef, useState } from 'react';
import { formatNumber, isFiniteNumber } from '../lib/logModel.js';

const PAD = { left: 48, right: 14, top: 20, bottom: 29 };

function contiguousPaths(values, toX, toY) {
  const paths = [];
  let current = [];
  values.forEach((point) => {
    if (!isFiniteNumber(point.value) || !isFiniteNumber(point.t)) {
      if (current.length) paths.push(current);
      current = [];
      return;
    }
    current.push([toX(point.t), toY(point.value)]);
  });
  if (current.length) paths.push(current);
  return paths.map((points) => points.map(([x, y], index) => `${index ? 'L' : 'M'} ${x.toFixed(2)} ${y.toFixed(2)}`).join(' '));
}

export default function LineChart({ title, subtitle, frames, series, selectedIndex, onSelect, height = 220 }) {
  const [hoverIndex, setHoverIndex] = useState(null);
  const svgRef = useRef(null);
  const width = 720;
  const chartWidth = width - PAD.left - PAD.right;
  const chartHeight = height - PAD.top - PAD.bottom;
  const geometry = useMemo(() => {
    const times = frames.map((frame) => frame.t).filter(isFiniteNumber);
    const tMin = times.length ? Math.min(...times) : 0;
    const tMax = times.length ? Math.max(...times) : 0;
    const duration = Math.max(0.0001, tMax - tMin);
    const values = series.flatMap((entry) => frames.map((frame) => entry.get(frame)).filter(isFiniteNumber));
    let min = values.length ? Math.min(...values) : 0;
    let max = values.length ? Math.max(...values) : 1;
    if (min === max) { min -= 0.5; max += 0.5; }
    const margin = (max - min) * 0.08;
    min -= margin;
    max += margin;
    const toX = (t) => PAD.left + ((Math.max(tMin, Math.min(tMax, t)) - tMin) / duration) * chartWidth;
    const toY = (value) => PAD.top + (1 - (value - min) / (max - min)) * chartHeight;
    const yTicks = Array.from({ length: 4 }, (_, index) => min + ((max - min) * index) / 3);
    const xTicks = Array.from({ length: 5 }, (_, index) => tMin + (duration * index) / 4);
    const paths = Object.fromEntries(series.map((entry) => [
      entry.key,
      contiguousPaths(frames.map((frame) => ({ t: frame.t, value: entry.get(frame) })), toX, toY),
    ]));
    return { tMin, tMax, duration, min, max, toX, toY, yTicks, xTicks, paths };
  }, [frames, series, chartWidth, chartHeight]);
  const { tMin, tMax, duration, toX, toY, yTicks, xTicks, paths } = geometry;
  const activeIndex = hoverIndex ?? selectedIndex;
  const activeFrame = frames[activeIndex];

  const indexFromEvent = (event) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect || !frames.length) return 0;
    const localX = ((event.clientX - rect.left) / rect.width) * width;
    const targetTime = Math.max(tMin, Math.min(tMax, tMin + ((localX - PAD.left) / chartWidth) * duration));
    let best = 0;
    let distance = Infinity;
    frames.forEach((frame, index) => {
      if (!isFiniteNumber(frame.t)) return;
      const nextDistance = Math.abs(frame.t - targetTime);
      if (nextDistance < distance) { distance = nextDistance; best = index; }
    });
    return best;
  };

  return (
    <section className="chart-card">
      <div className="chart-head">
        <div><span className="eyebrow">TIME SERIES</span><h3>{title}</h3></div>
        <span>{subtitle}</span>
      </div>
      <div
        className="chart-canvas"
        role="slider"
        tabIndex="0"
        aria-label={`${title} selected frame`}
        aria-valuemin="0"
        aria-valuemax={Math.max(0, frames.length - 1)}
        aria-valuenow={Math.max(0, selectedIndex)}
        aria-valuetext={activeFrame ? `Frame ${activeFrame.frameId}, ${formatNumber(activeFrame.t, 5)} seconds` : 'No frames'}
        onKeyDown={(event) => {
          if (!frames.length) return;
          if (event.key === 'ArrowLeft') { event.preventDefault(); onSelect(Math.max(0, selectedIndex - 1)); }
          if (event.key === 'ArrowRight') { event.preventDefault(); onSelect(Math.min(frames.length - 1, selectedIndex + 1)); }
          if (event.key === 'Home') { event.preventDefault(); onSelect(0); }
          if (event.key === 'End') { event.preventDefault(); onSelect(frames.length - 1); }
        }}
      >
        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={`${title} over elapsed time`}
          onPointerMove={(event) => setHoverIndex(indexFromEvent(event))}
          onPointerLeave={() => setHoverIndex(null)}
          onClick={(event) => onSelect(indexFromEvent(event))}
        >
          {yTicks.map((tick) => <line key={tick} x1={PAD.left} x2={width - PAD.right} y1={toY(tick)} y2={toY(tick)} className="chart-gridline" />)}
          {xTicks.map((tick) => <line key={tick} y1={PAD.top} y2={height - PAD.bottom} x1={toX(tick)} x2={toX(tick)} className="chart-gridline vertical" />)}
          {series.map((entry) => {
            return paths[entry.key].map((path, index) => (
              <path key={`${entry.key}-${index}`} d={path} fill="none" stroke={entry.color} strokeWidth={entry.width || 1.8} strokeDasharray={entry.dash || undefined} className="chart-line" />
            ));
          })}
          {yTicks.map((tick) => <text key={`y-${tick}`} x={PAD.left - 8} y={toY(tick) + 3} textAnchor="end" className="chart-axis-text">{formatNumber(tick, 3)}</text>)}
          {xTicks.map((tick) => <text key={`x-${tick}`} x={toX(tick)} y={height - 9} textAnchor="middle" className="chart-axis-text">{tick.toFixed(duration < 10 ? 2 : 1)}s</text>)}
          {activeFrame && isFiniteNumber(activeFrame.t) && (
            <g>
              <line x1={toX(activeFrame.t)} x2={toX(activeFrame.t)} y1={PAD.top} y2={height - PAD.bottom} className="chart-cursor" />
              {series.map((entry) => {
                const value = entry.get(activeFrame);
                return isFiniteNumber(value) ? <circle key={entry.key} cx={toX(activeFrame.t)} cy={toY(value)} r="3.5" fill={entry.color} stroke="#07110f" strokeWidth="1.5" /> : null;
              })}
            </g>
          )}
        </svg>
        {activeFrame && (
          <div className="chart-tooltip">
            <b>f{activeFrame.frameId}</b><span>{formatNumber(activeFrame.t, 4)} s</span>
            {series.map((entry) => <span key={entry.key} style={{ '--series-color': entry.color }}><i />{entry.label} {formatNumber(entry.get(activeFrame), 5)}</span>)}
          </div>
        )}
      </div>
      <div className="chart-legend">
        {series.map((entry) => <span key={entry.key}><i style={{ background: entry.color }} />{entry.label}</span>)}
      </div>
    </section>
  );
}

export function StatusLane({ frames, selectedIndex, onSelect }) {
  const colors = {
    success: '#9df7c7', pose: '#61d9f4', unrecognized_pose: '#a889d8',
    no_detections: '#263b34', insufficient_markers: '#dfb866', normalization_failed: '#ff9f66',
    no_complete_window: '#7e8f89', no_map_match: '#a889d8', ambiguous_map_match: '#f4cf61',
    pnp_failed: '#ff7777', no_pose: '#263b34',
  };
  return (
    <div className="status-lane" aria-label="Localization state by frame">
      {frames.map((frame) => (
        <button
          key={frame.index}
          aria-label={`Frame ${frame.frameId}: ${frame.status}`}
          title={`f${frame.frameId} · ${frame.status}`}
          tabIndex={frame.index === selectedIndex ? 0 : -1}
          aria-current={frame.index === selectedIndex ? 'true' : undefined}
          className={frame.index === selectedIndex ? 'selected' : ''}
          style={{ background: colors[frame.status] || '#60736c' }}
          onClick={() => onSelect(frame.index)}
        />
      ))}
    </div>
  );
}
