import { useDeferredValue, useMemo, useState } from 'react';
import { Check, Copy, Search } from 'lucide-react';
import { flattenObject, valueToText } from '../lib/logModel.js';

export function DataTable({ rows, columns, empty = 'No records in this frame.' }) {
  if (!rows?.length) return <div className="table-empty">{empty}</div>;
  return (
    <div className="data-table-wrap">
      <table className="data-table">
        <thead><tr>{columns.map((column) => <th key={column.key}>{column.label}</th>)}</tr></thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={row.__key ?? index}>
              {columns.map((column) => <td key={column.key}>{column.render ? column.render(row, index) : valueToText(row?.[column.key])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function AutoDataTable({ rows, empty }) {
  const columns = useMemo(() => {
    const keys = [];
    for (const row of rows || []) {
      if (!row || typeof row !== 'object') continue;
      for (const key of Object.keys(row)) if (!keys.includes(key)) keys.push(key);
    }
    return keys.map((key) => ({ key, label: key.replaceAll('_', ' ') }));
  }, [rows]);
  return <DataTable rows={rows} columns={columns} empty={empty} />;
}

export function KeyValueView({ data, prefix = '' }) {
  const entries = useMemo(() => flattenObject(data, prefix), [data, prefix]);
  return (
    <div className="key-value-view">
      {entries.map((entry, index) => (
        <div key={`${entry.path}-${index}`}>
          <code>{entry.path}</code>
          <span className={`value-type type-${entry.type.replace(' ', '-')}`}>{entry.type}</span>
          <strong>{valueToText(entry.value)}</strong>
        </div>
      ))}
    </div>
  );
}

export function RawJsonView({ model, selectedFrameIndex }) {
  const [scope, setScope] = useState('frame');
  const [query, setQuery] = useState('');
  const [copied, setCopied] = useState(false);
  const scopedValue = scope === 'all' ? model.raw
    : scope === 'args' ? model.args
      : scope === 'config' ? model.config
        : (model.frames[selectedFrameIndex]?.safeRaw ?? null);
  const pretty = useMemo(() => JSON.stringify(scopedValue, null, 2) ?? 'null', [scopedValue]);
  const flatFields = useMemo(() => flattenObject(scopedValue), [scopedValue]);
  const deferredQuery = useDeferredValue(query);
  const results = useMemo(() => {
    if (!deferredQuery.trim()) return [];
    const needle = deferredQuery.trim().toLowerCase();
    return flatFields.filter((entry) => `${entry.path} ${valueToText(entry.value)}`.toLowerCase().includes(needle)).slice(0, 500);
  }, [flatFields, deferredQuery]);
  const copy = async () => {
    await navigator.clipboard.writeText(pretty);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  };
  return (
    <section className="raw-view panel">
      <div className="raw-toolbar">
        <div className="segmented" role="tablist" aria-label="Raw JSON scope">
          {['frame', 'args', 'config', 'all'].map((value) => <button role="tab" aria-selected={scope === value} key={value} className={scope === value ? 'selected' : ''} onClick={() => setScope(value)}>{value === 'all' ? 'Complete log' : value}</button>)}
        </div>
        <label className="search-field"><Search size={14} /><input aria-label="Search JSON path or value" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search path or value" /></label>
        <button className="plain-button inline" onClick={copy}>{copied ? <Check size={14} /> : <Copy size={14} />}{copied ? 'Copied' : 'Copy JSON'}</button>
      </div>
      {query ? (
        <div className="raw-results">
          <p>{results.length === 500 ? 'First 500 matching scalar fields' : `${results.length} matching scalar field${results.length === 1 ? '' : 's'}`}</p>
          <KeyValueView data={Object.fromEntries(results.map((entry) => [entry.path, entry.value]))} />
        </div>
      ) : <pre className="raw-json"><code>{pretty}</code></pre>}
    </section>
  );
}
