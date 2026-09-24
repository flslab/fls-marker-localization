import { useMemo, useState } from 'react';
import LineChart from './LineChart.jsx';

const AXES = [
  { key: 'x', label: 'X', color: '#ff7777' },
  { key: 'y', label: 'Y', color: '#9df7c7' },
  { key: 'z', label: 'Z', color: '#61d9f4' },
];

const ANGLES = [
  { key: 'roll', label: 'Roll', color: '#ff9f66' },
  { key: 'pitch', label: 'Pitch', color: '#a889d8' },
  { key: 'yaw', label: 'Yaw', color: '#61d9f4' },
];

const PROPERTIES = {
  dronePosition: {
    label: 'Drone position',
    field: 'dronePosition',
    subtitle: 'world FLU frame · metres',
    axes: AXES,
  },
  droneOrientation: {
    label: 'Drone orientation',
    field: 'droneOrientation',
    subtitle: 'world FLU frame · radians',
    axes: ANGLES,
  },
  cameraPosition: {
    label: 'Camera position',
    field: 'position',
    subtitle: 'world FLU frame · metres',
    axes: AXES,
  },
  cameraOrientation: {
    label: 'Camera orientation',
    field: 'orientation',
    subtitle: 'world FLU frame · radians',
    axes: ANGLES,
  },
  markerPosition: {
    label: 'Marker position',
    field: 'markerPosition',
    subtitle: 'camera frame · metres',
    axes: AXES,
  },
};

const TECHNIQUES = [
  { key: 'pnp', label: 'PnP' },
  { key: 'shared_attitude', label: 'Shared attitude' },
];

function techniquePose(frame, technique) {
  return frame.poses.find((pose) => pose.kind === 'camera-world' && pose.poseTechnique === technique) || null;
}

export default function PoseTechniqueComparison({ frames, selectedIndex, onSelect }) {
  const [propertyKey, setPropertyKey] = useState('dronePosition');
  const property = PROPERTIES[propertyKey];
  const techniqueSeries = useMemo(() => Object.fromEntries(TECHNIQUES.map((technique) => [
    technique.key,
    property.axes.map((axis, axisIndex) => ({
      ...axis,
      key: `${technique.key}-${property.field}-${axis.key}`,
      get: (frame) => techniquePose(frame, technique.key)?.[property.field]?.[axisIndex],
    })),
  ])), [property]);
  const valueDomain = useMemo(() => {
    let min = Infinity;
    let max = -Infinity;
    Object.values(techniqueSeries).forEach((series) => series.forEach((entry) => frames.forEach((frame) => {
      const value = entry.get(frame);
      if (typeof value !== 'number' || !Number.isFinite(value)) return;
      min = Math.min(min, value);
      max = Math.max(max, value);
    })));
    return min === Infinity ? null : [min, max];
  }, [frames, techniqueSeries]);

  return (
    <section className="pose-comparison-panel">
      <div className="pose-comparison-head">
        <div>
          <span className="eyebrow">POSE TECHNIQUE COMPARISON</span>
          <h3>{property.label}</h3>
          <p>Compare the same logged quantity on synchronized time axes.</p>
        </div>
        <label className="plot-property-select">
          <span>Property</span>
          <select value={propertyKey} onChange={(event) => setPropertyKey(event.target.value)}>
            {Object.entries(PROPERTIES).map(([key, option]) => <option key={key} value={key}>{option.label}</option>)}
          </select>
        </label>
      </div>
      <div className="pose-comparison-grid">
        {TECHNIQUES.map((technique) => (
          <LineChart
            key={technique.key}
            title={technique.label}
            subtitle={property.subtitle}
            frames={frames}
            series={techniqueSeries[technique.key]}
            valueDomain={valueDomain}
            selectedIndex={selectedIndex}
            onSelect={onSelect}
          />
        ))}
      </div>
    </section>
  );
}
