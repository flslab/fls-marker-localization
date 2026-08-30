import { AlertTriangle, CheckCircle2, CircleDot, Layers3, ScanLine } from 'lucide-react';
import { AutoDataTable, KeyValueView } from './DataViews.jsx';
import { formatNumber, formatRawTime, valueToText } from '../lib/logModel.js';

function Vector({ value, labels = ['X', 'Y', 'Z'] }) {
  return (
    <div className="vector">
      {labels.map((label, index) => <div key={label}><span>{label}</span><b>{formatNumber(value?.[index], 6)}</b></div>)}
    </div>
  );
}

function Detail({ title, count, children, open = false }) {
  return (
    <details className="inspector-detail" open={open}>
      <summary><span>{title}</span>{count !== undefined && <b>{count}</b>}</summary>
      <div className="detail-content">{children}</div>
    </details>
  );
}

function PoseCard({ pose, index }) {
  const raw = pose.raw || {};
  return (
    <article className="pose-card">
      <div className="pose-title"><CircleDot size={15} /><strong>{pose.label}</strong><span>{pose.source}</span></div>
      {pose.position && <><h4>Position <em>m · {pose.frameLabel}</em></h4><Vector value={pose.position} /></>}
      {pose.filteredPosition && <><h4>Filtered position <em>m</em></h4><Vector value={pose.filteredPosition} /></>}
      {pose.orientation && <><h4>Orientation <em>rad · roll/pitch/yaw</em></h4><Vector value={pose.orientation} labels={['R', 'P', 'Y']} /></>}
      {pose.kind === 'legacy' && pose.entity === 'marker' && pose.cameraPosition && <><h4>Camera in marker frame <em>m</em></h4><Vector value={pose.cameraPosition} /></>}
      {pose.kind === 'legacy' && pose.entity === 'camera' && pose.markerPosition && <><h4>Marker in camera frame <em>m</em></h4><Vector value={pose.markerPosition} /></>}
      <Detail title={`Complete pose ${index + 1} JSON`}><pre className="mini-json">{JSON.stringify(raw, null, 2)}</pre></Detail>
    </article>
  );
}

export default function FrameInspector({ model, frameIndex }) {
  const frame = model.frames[frameIndex];
  if (!frame) return <aside className="panel inspector full-inspector"><div className="empty-state"><strong>No frame selected</strong></div></aside>;
  const grid = frame.grid;
  const gridCounters = grid ? [
    ['Tracked', grid.tracked_decoded_marker_count], ['Eligible', grid.decoded_marker_count],
    ['Accepted', grid.accepted_marker_count], ['Required', grid.required_marker_count],
    ['Windows', grid.complete_window_count], ['Candidates', grid.candidate_count],
    ['Best match', grid.best_match_count], ['Max age', `${formatNumber(grid.max_marker_age, 4)} s`],
  ] : [];
  return (
    <aside className="panel inspector full-inspector">
      <div className="panel-head inspector-head">
        <div><span className="eyebrow">CURRENT SAMPLE</span><h2>Frame {frame.frameId}</h2></div>
        <span className={frame.poseValid ? 'success-pill' : 'failure-pill'}>{frame.poseValid ? <CheckCircle2 size={12} /> : <AlertTriangle size={12} />}{frame.poseValid ? 'POSE VALID' : 'NO VALID POSE'}</span>
      </div>
      <div className="inspect-time">
        <div><span>Normalized time</span><strong>{formatNumber(frame.t, 6)} <small>s</small></strong></div>
        <div><span>Raw time</span><b title={formatRawTime(frame.rawTime)}>{formatRawTime(frame.rawTime)}</b></div>
      </div>
      <div className="inspector-scroll">
        {frame.poses.length ? frame.poses.map((pose, index) => <PoseCard key={index} pose={pose} index={index} />) : (
          <div className="no-pose-card"><ScanLine size={18} /><div><strong>No pose record</strong><span>Detection and lookup diagnostics may still be available below.</span></div></div>
        )}
        <section className="quality summary-quality">
          <div><span>Reprojection error</span><b>{formatNumber(frame.reprojectionError, 4)} <small>px</small></b></div>
          <div><span>Pose records</span><b>{frame.poseRecords.length}</b></div>
          <div><span>Blobs</span><b>{frame.blobs.length}</b></div>
        </section>

        {grid && <Detail title="Localization diagnostics" open>
          <div className="diagnostic-status"><span className={`status-dot status-${grid.status}`} /> <div><strong>{grid.status || 'unknown'}</strong><p>{grid.message || 'No diagnostic message.'}</p></div></div>
          <div className="lookup-row"><span>lookup</span><b>{grid.lookup_status || 'unknown'}</b><span>attempted</span><b>{valueToText(grid.lookup_attempted)}</b></div>
          <div className="counter-grid">{gridCounters.map(([label, value]) => <div key={label}><span>{label}</span><b>{valueToText(value)}</b></div>)}</div>
          {grid.window_match && <><h4>Window match</h4><KeyValueView data={grid.window_match} /></>}
          {grid.reprojection_error !== undefined && <div className="diagnostic-foot"><span>Grid RMS reprojection error</span><b>{valueToText(grid.reprojection_error)} px</b></div>}
        </Detail>}

        <Detail title="Blob detections" count={frame.blobs.length}>
          <AutoDataTable rows={frame.blobs} />
        </Detail>
        {grid && <>
          <Detail title="Decoded tracks" count={grid.decoded_tracks?.length || 0}>
            <AutoDataTable rows={grid.decoded_tracks || []} />
          </Detail>
          <Detail title="Relative markers" count={grid.relative_markers?.length || 0}>
            <AutoDataTable rows={grid.relative_markers || []} />
          </Detail>
          <Detail title="Matched markers" count={grid.matched_markers?.length || 0}>
            <AutoDataTable rows={grid.matched_markers || []} />
          </Detail>
        </>}
        <Detail title="Complete frame JSON">
          <pre className="mini-json">{JSON.stringify(frame.safeRaw, null, 2)}</pre>
        </Detail>
        <div className="coverage-note"><Layers3 size={14} /> Every field in this frame is preserved in the complete JSON view.</div>
      </div>
    </aside>
  );
}
