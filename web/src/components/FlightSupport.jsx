import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { Box, Camera, FileJson, FolderOpen, Info, Layers3, LoaderCircle, SlidersHorizontal } from 'lucide-react';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { cameraOverlayGeometry } from '../lib/cameraOverlay.js';
import { analyticHeightSlice } from '../lib/levelFlightRegion.js';
import { supportLayout, validateCamera, processingRoi } from '../lib/flightSupport.js';
import { buildMarkerGridFile } from '../lib/markerGrid.js';
import cameraDefault from '../../../high_rate_localizer/config/localizer.json';
import { rawToScene, AXIS_COLORS } from '../lib/sceneMath.js';
const demo=buildMarkerGridFile({},Array.from({length:16},(_,i)=>({i:i%4-1,j:Math.floor(i/4)-1})));
function geometryFromWorld(positions, normals = true) {
  const mapped = [];
  for (let i=0;i<positions.length;i+=3) mapped.push(positions[i],positions[i+2],-positions[i+1]);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(mapped,3));
  if (normals) geometry.computeVertexNormals();
  return geometry;
}
function disposeGroup(group) {
  group.traverse(object=>{object.geometry?.dispose();object.material?.dispose();});
  group.clear();
}
function Scene({layout,result,showLevel,slice,sliceEnabled,config,cameraEnabled,cameraPose,cuboidEnabled,maxHeight}) {
  const ref=useRef(null), sceneRef=useRef(null), viewRef=useRef(null);
  const [error,setError]=useState('');
  useEffect(()=>{
    const host=ref.current; let renderer;
    try { renderer=new THREE.WebGLRenderer({antialias:true}); }
    catch { setError('WebGL is unavailable. The support statistics are still available.');return; }
    setError('');renderer.setPixelRatio(Math.min(devicePixelRatio,2));host.appendChild(renderer.domElement);
    const scene=new THREE.Scene();scene.background=new THREE.Color('#091510');
    const camera=new THREE.PerspectiveCamera(45,1,.001,1000);
    const controls=new OrbitControls(camera,renderer.domElement);
    const bounds=result?.planeBounds??result?.searchBounds??layout.bounds;
    const viewKey=JSON.stringify([bounds,maxHeight]);
    const center=[(bounds[0][0]+bounds[0][1])/2,(bounds[1][0]+bounds[1][1])/2,layout.floor];
    const size=Math.max(bounds[0][1]-bounds[0][0],bounds[1][1]-bounds[1][0],maxHeight||2,2);
    controls.target.copy(rawToScene([center[0],center[1],center[2]+.7]));
    camera.position.copy(rawToScene([center[0]+size*1.4,center[1]-size*1.4,center[2]+size*1.2]));
    if(viewRef.current?.layout===layout&&viewRef.current?.viewKey===viewKey) {camera.position.copy(viewRef.current.position);controls.target.copy(viewRef.current.target);}
    controls.update();
    scene.add(new THREE.HemisphereLight(0xffffff,0x325444,3));
    const axes=new THREE.Group();
    for(const [axis,direction] of Object.entries({x:[1,0,0],y:[0,1,0],z:[0,0,1]})) {
      axes.add(new THREE.ArrowHelper(rawToScene(direction),new THREE.Vector3(),.3,AXIS_COLORS[axis],.06,.025));
    }
    axes.position.copy(rawToScene(center));scene.add(axes);
    const fullMaterials=[];
    const addInstances=(positions,geometry,color,opacity,scale,region=false)=>{
      if(!positions.length){geometry.dispose();return;}
      const material=new THREE.MeshStandardMaterial({color,transparent:opacity<1,opacity,depthWrite:opacity===1});
      if(region) fullMaterials.push({material,opacity});
      const mesh=new THREE.InstancedMesh(geometry,material,positions.length);const matrix=new THREE.Matrix4();
      positions.forEach((p,i)=>{matrix.makeScale(...scale);matrix.setPosition(rawToScene(p));mesh.setMatrixAt(i,matrix);});scene.add(mesh);
    };
    addInstances(layout.markers,new THREE.SphereGeometry(1,8,6),0xffad68,1,[layout.diameter/2,layout.diameter/2,layout.diameter/2]);
    if(result?.type==='continuous') {
      const material=new THREE.MeshStandardMaterial({color:0x79e9aa,side:THREE.DoubleSide,transparent:true,opacity:.65,depthWrite:false,roughness:.55});
      fullMaterials.push({material,opacity:.65});
      scene.add(new THREE.Mesh(geometryFromWorld(result.positions),material));
    } else if(result) {
      const cell=[result.cell[0]*.92,result.cell[2]*.92,result.cell[1]*.92];
      if(showLevel) addInstances(result.level,new THREE.BoxGeometry(1,1,1),0x629ace,.06,cell,true);
      addInstances(result.robust,new THREE.BoxGeometry(1,1,1),0x79e9aa,.42,cell,true);
    }
    const sliceGroup=new THREE.Group();scene.add(sliceGroup);
    const overlayGroup=new THREE.Group();scene.add(overlayGroup);
    sceneRef.current={sliceGroup,fullMaterials,overlayGroup};
    const observer=new ResizeObserver(()=>{const w=host.clientWidth,h=host.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();});observer.observe(host);
    let frame;const render=()=>{frame=requestAnimationFrame(render);controls.update();renderer.render(scene,camera);};render();
    return()=>{
      viewRef.current={layout,viewKey,position:camera.position.clone(),target:controls.target.clone()};
      sceneRef.current=null;cancelAnimationFrame(frame);observer.disconnect();controls.dispose();
      disposeGroup(scene);renderer.dispose();renderer.domElement.remove();
    };
  },[layout,result,showLevel,maxHeight]);
  useEffect(()=>{
    const state=sceneRef.current;if(!state)return;
    disposeGroup(state.sliceGroup);
    for(const {material,opacity} of state.fullMaterials) material.opacity=sliceEnabled?opacity*.16:opacity;
    if(!result||!sliceEnabled)return;
    const bounds=result.planeBounds??result.searchBounds??layout.bounds;
    const z=layout.floor+slice;
    const corners=[[bounds[0][0],bounds[1][0],z],[bounds[0][1],bounds[1][0],z],[bounds[0][1],bounds[1][1],z],[bounds[0][0],bounds[1][1],z]];
    const planePositions=[...corners[0],...corners[1],...corners[2],...corners[0],...corners[2],...corners[3]];
    const plane=new THREE.Mesh(geometryFromWorld(planePositions),new THREE.MeshBasicMaterial({color:0x61d9f4,side:THREE.DoubleSide,transparent:true,opacity:.07,depthWrite:false}));
    state.sliceGroup.add(plane);
    const edgePositions=[...corners[0],...corners[1],...corners[2],...corners[3],...corners[0]];
    state.sliceGroup.add(new THREE.Line(geometryFromWorld(edgePositions,false),new THREE.LineBasicMaterial({color:0x61d9f4,transparent:true,opacity:.5,depthWrite:false})));
    let positions=[];
    if(result.type==='continuous') positions=analyticHeightSlice(result,slice);
    else {
      for(const p of result.robust.filter(p=>Math.abs(p[2]-z)<=result.cell[2]/2)) {
        const x=result.cell[0]/2,y=result.cell[1]/2;
        positions.push(p[0]-x,p[1]-y,z,p[0]+x,p[1]-y,z,p[0]+x,p[1]+y,z,p[0]-x,p[1]-y,z,p[0]+x,p[1]+y,z,p[0]-x,p[1]+y,z);
      }
    }
    const fill=new THREE.Mesh(geometryFromWorld(positions),new THREE.MeshBasicMaterial({color:0x61d9f4,side:THREE.DoubleSide,polygonOffset:true,polygonOffsetFactor:-1,polygonOffsetUnits:-1}));
    fill.renderOrder=2;state.sliceGroup.add(fill);
  },[layout,result,showLevel,slice,sliceEnabled,maxHeight]);
  useEffect(()=>{
    const state=sceneRef.current;if(!state)return;
    const group=state.overlayGroup;disposeGroup(group);
    const line=(points,color,opacity=1,loop=false)=>{
      const positions=(loop?[...points,points[0]]:points).flat();
      const outline=new THREE.Line(geometryFromWorld(positions,false),new THREE.LineBasicMaterial({color,transparent:opacity<1,opacity,depthWrite:false,depthTest:false}));
      outline.renderOrder=3;group.add(outline);
    };
    if(cuboidEnabled&&Number.isFinite(maxHeight)&&maxHeight>0) {
      for(const bounds of layout.tileFootprints) {
        const lower=[[bounds[0][0],bounds[1][0],layout.floor],[bounds[0][1],bounds[1][0],layout.floor],[bounds[0][1],bounds[1][1],layout.floor],[bounds[0][0],bounds[1][1],layout.floor]];
        const upper=lower.map(p=>[p[0],p[1],layout.floor+maxHeight]);
        line(lower,0xa889d8,.45,true);line(upper,0xa889d8,.65,true);
        for(let i=0;i<4;i++)line([lower[i],upper[i]],0xa889d8,.45);
      }
    }
    if(cameraEnabled) {
      const shape=cameraOverlayGeometry(config,cameraPose,layout.floor,maxHeight);
      if(!shape)return;
      const colors=[AXIS_COLORS.x,AXIS_COLORS.y,AXIS_COLORS.z];
      shape.axes.forEach((axis,i)=>{group.add(new THREE.ArrowHelper(rawToScene(axis).normalize(),rawToScene(shape.center),.14,colors[i],.035,.018));});
      const body=new THREE.Mesh(new THREE.SphereGeometry(.025,12,8),new THREE.MeshBasicMaterial({color:0xff9f66,depthTest:false}));
      body.renderOrder=4;
      body.position.copy(rawToScene(shape.center));group.add(body);
      line(shape.near,0xff9f66,1,true);line(shape.far,0xff9f66,.8,true);
      for(let i=0;i<4;i++)line([shape.near[i],shape.far[i]],0xff9f66,.8);
      if(shape.ground) {
        const p=shape.ground.map(point=>[point[0],point[1],point[2]+.001]);
        const footprint=new THREE.Mesh(geometryFromWorld([...p[0],...p[1],...p[2],...p[0],...p[2],...p[3]]),new THREE.MeshBasicMaterial({color:0xff9f66,side:THREE.DoubleSide,transparent:true,opacity:.16,depthWrite:false}));
        group.add(footprint);line(p,0xff9f66,1,true);
      }
    }
  },[layout,result,showLevel,config,cameraEnabled,cameraPose,cuboidEnabled,maxHeight]);
  return <div className="support-scene" ref={ref} aria-label="3D flight support region">{error&&<p>{error}</p>}</div>;
}
function HeightSlice({value,enabled,onChange,onEnable,min,max}) {
  return <div className="support-slice">
    <label className="support-reference"><input type="checkbox" checked={enabled} onChange={e=>onEnable(e.target.checked)}/>Slice plane</label>
    <div className="support-slice-value"><span>Height above floor</span><output>{value.toFixed(3)} m</output></div>
    <input className="support-slice-slider" type="range" aria-label="Height slice" min={min} max={max} step="0.001" value={value} onChange={e=>{onChange(Number(e.target.value));onEnable(true);}}/>
    <div className="support-slice-scale"><span>{min.toFixed(2)} m</span><span>{max.toFixed(2)} m</span></div>
  </div>;
}
export default function FlightSupport({source}) {
  const [layout,setLayout]=useState(()=>supportLayout(source?.raw??demo));
  const [name,setName]=useState(source?.name??'Finite 4 × 4 tile example');
  const [config,setConfig]=useState(cameraDefault);
  const [cameraName,setCameraName]=useState('Localizer camera');
  const layoutInput=useRef(null), cameraInput=useRef(null);
  const [options,setOptions]=useState({mode:'yaw',surfaceResolution:48,tilt:15,yaw:0,minHeight:.1,maxHeight:3,resolution:20,attitudeSteps:5,minArea:8,padding:.3});
  const [result,setResult]=useState(null),[error,setError]=useState(''),[progress,setProgress]=useState(null),[showLevel,setShowLevel]=useState(true),[slice,setSlice]=useState(.6),[sliceEnabled,setSliceEnabled]=useState(false);
  const [cameraEnabled,setCameraEnabled]=useState(false),[cuboidEnabled,setCuboidEnabled]=useState(false);
  const [cameraPose,setCameraPose]=useState(()=>({x:(layout.bounds[0][0]+layout.bounds[0][1])/2,y:(layout.bounds[1][0]+layout.bounds[1][1])/2,height:1,roll:0,pitch:0,yaw:0}));
  const cameraGeometry=useMemo(()=>cameraOverlayGeometry(config,cameraPose,layout.floor,Number(options.maxHeight)),[config,cameraPose,layout.floor,options.maxHeight]);
  const worker=useRef(null);
  useEffect(()=>()=>worker.current?.terminate(),[]);
  const invalidate=()=>{worker.current?.terminate();worker.current=null;setProgress(null);setResult(null);setSliceEnabled(false);};
  const update=(key,value)=>{invalidate();setOptions(o=>({...o,[key]:value}));};
  const calculate=()=>{
    invalidate();setError('');setProgress(0);
    const w=new Worker(new URL('../lib/flightSupport.worker.js',import.meta.url),{type:'module'});worker.current=w;
    w.onmessage=({data})=>{if(data.progress!==undefined)setProgress(data.progress);if(data.result||data.error){setResult(data.result??null);if(data.result)setSlice(Math.max(Number(options.minHeight),Math.min(Number(options.maxHeight),.6)));setError(data.error??'');setProgress(null);w.terminate();worker.current=null;}};
    w.onerror=()=>{setError('The region calculation failed. Try a smaller sampling resolution.');setProgress(null);w.terminate();worker.current=null;};
    w.postMessage({layout,config,options});
  };
  const upload=async(event,camera)=>{const file=event.target.files?.[0];event.target.value='';if(!file)return;try{const raw=JSON.parse(await file.text());if(camera){validateCamera(raw);invalidate();setConfig(raw);setCameraName(file.name);setOptions(o=>({...o,minArea:raw.detector?.minimum_area??8}));}else{const parsed=supportLayout(raw);invalidate();setLayout(parsed);setName(file.name);setCameraPose(p=>({...p,x:(parsed.bounds[0][0]+parsed.bounds[0][1])/2,y:(parsed.bounds[1][0]+parsed.bounds[1][1])/2}));}setError('');}catch(e){setError(e.message);}};
  return <section className="generator-page support-page" onDrop={e=>e.stopPropagation()}>
    <header className="generator-heading"><div><span className="eyebrow">FLIGHT SUPPORT</span><h1>Supported flight region</h1><p>Explore a continuous level-camera region guaranteed through every yaw, or compare sampled tilt support.</p></div></header>
    <div className="support-layout">
      <aside className="panel generator-sidebar support-sidebar">
        <section>
          <div className="generator-section-head"><div><span className="eyebrow">01 · INPUTS</span><h2>Layout and camera</h2></div><Layers3 size={18}/></div>
          <div className="support-source"><FileJson size={15}/><div><strong title={name}>{name}</strong><span>{layout.markers.length} installed flight markers</span></div></div>
          <div className="support-source"><Camera size={15}/><div><strong title={cameraName}>{cameraName}</strong><span>{config.camera.width} × {config.camera.height} px</span></div></div>
          <div className="support-file-actions">
            <button className="support-load" onClick={()=>layoutInput.current?.click()}><FolderOpen size={13}/>Layout JSON</button>
            <button className="support-load" onClick={()=>cameraInput.current?.click()}><Camera size={13}/>Camera config</button>
          </div>
          <input ref={layoutInput} hidden type="file" accept=".json" aria-label="Load layout JSON" onChange={e=>upload(e,false)}/>
          <input ref={cameraInput} hidden type="file" accept=".json" aria-label="Load camera config" onChange={e=>upload(e,true)}/>
          <div className="support-roi"><span className="eyebrow">PROCESSING ROI</span><code>{processingRoi(config).width} × {processingRoi(config).height} px · offset {processingRoi(config).x}, {processingRoi(config).y}</code><small>{config.processing_crop?.enabled ? 'Configured crop enabled' : 'Full frame · crop disabled'}</small></div>
          <p>Selected MyGrid tiles define the installed HyperGrid floor. Flight support uses its always-on markers.</p>
        </section>
        <section>
          <div className="generator-section-head"><div><span className="eyebrow">02 · GEOMETRY</span><h2>Flight envelope</h2></div><SlidersHorizontal size={18}/></div>
          <label className="generator-field"><span>Region model</span><select value={options.mode} onChange={e=>update('mode',e.target.value)}><option value="yaw">Level camera · every yaw</option><option value="fixed">Level camera · fixed yaw</option><option value="tilt">Sampled roll / pitch</option></select></label>
          {[[ 'tilt','Maximum roll / pitch','°',0,45],['yaw','Fixed drone yaw','°',-180,180],['minHeight','Minimum drone height','m',.01,10],['maxHeight','Maximum drone height','m',.02,10],['minArea','Minimum projected dot area','px²',.1,1000],['padding',options.mode==='tilt' ? 'Horizontal search border' : 'Extra plane border','m',0,10]].filter(([key])=>key==='tilt' ? options.mode==='tilt' : key==='yaw' ? options.mode!=='yaw' : true).map(([key,label,unit,min,max])=><label className="generator-field" key={key}><span>{label}</span><span className="generator-input-wrap"><input type="number" min={min} max={max} step={unit==='m'?.05:1} value={options[key]} onChange={e=>update(key,e.target.value===''?'':Number(e.target.value))}/><small>{unit}</small></span></label>)}
          {options.mode !== 'tilt' ? <label className="generator-field"><span>Surface detail</span><select value={options.surfaceResolution} onChange={e=>update('surfaceResolution',Number(e.target.value))}><option value={24}>Quick · 24 × 24</option><option value={48}>Standard · 48 × 48</option><option value={80}>Fine · 80 × 80</option></select></label> : <>
          <label className="generator-field"><span>Position sampling</span><select value={options.resolution} onChange={e=>update('resolution',Number(e.target.value))}><option value={12}>Quick · 12 × 12 × 16</option><option value={20}>Standard · 20 × 20 × 16</option><option value={28}>Fine · 28 × 28 × 16</option></select></label>
          <label className="generator-field"><span>Roll / pitch samples per axis</span><select value={options.attitudeSteps} onChange={e=>update('attitudeSteps',Number(e.target.value))}>{[3,5,7].map(n=><option key={n} value={n}>{n} ({n*n} combined attitudes)</option>)}</select></label></>}
          <button className="generator-download support-calculate" disabled={progress!==null} onClick={calculate}>{progress!==null ? <LoaderCircle size={14} className="spin"/> : <Box size={14}/>} {progress!==null?`Calculating… ${Math.round(progress*100)}%`:'Calculate region'}</button>
        </section>
        <section>
          <div className="generator-section-head"><div><span className="eyebrow">03 · SCENE OVERLAYS</span><h2>Camera and tile volume</h2></div><Camera size={18}/></div>
          <label className="support-overlay-toggle"><input type="checkbox" checked={cameraEnabled} onChange={e=>setCameraEnabled(e.target.checked)}/>Camera, frustum and floor footprint</label>
          <label className="support-overlay-toggle"><input type="checkbox" checked={cuboidEnabled} onChange={e=>setCuboidEnabled(e.target.checked)}/>Tile cuboids to maximum height</label>
          {cameraEnabled&&<>
            <span className="generator-subtitle">Camera center · world coordinates</span>
            {[['x','Camera X','m'],['y','Camera Y','m'],['height','Camera height above floor','m'],['roll','Camera roll','°'],['pitch','Camera pitch','°'],['yaw','Camera yaw','°']].map(([key,label,unit])=><label className="generator-field" key={key}><span>{label}</span><span className="generator-input-wrap"><input aria-label={label} type="number" step={unit==='m'?.05:1} value={cameraPose[key]} onChange={e=>setCameraPose(p=>({...p,[key]:e.target.value===''?'':Number(e.target.value)}))}/><small>{unit}</small></span></label>)}
            <p>Camera position is its optical center. Roll, pitch and yaw rotate it from the configured mount orientation. The orange pinhole frustum uses the processing ROI and ends at the marker plane.</p>
            {!cameraGeometry&&<p className="support-overlay-warning">Enter a finite pose and a positive camera height.</p>}
            {cameraGeometry&&!cameraGeometry.ground&&<p className="support-overlay-warning">The ROI crosses the horizon: no bounded ground footprint. Frustum rays are shown at a finite depth.</p>}
          </>}
          {cuboidEnabled&&<p>Purple cuboids extend each installed tile from the marker plane to {options.maxHeight} m. They show tile volumes independently of optical support.</p>}
        </section>
      </aside>
      <div className="support-main"><div className="panel support-stage"><div className="generator-stage-head"><div><span className="eyebrow">WORLD FLU · METRES</span><h2>Flight envelope · 3D</h2></div>{options.mode === 'tilt' && <label className="support-reference"><input type="checkbox" checked={showLevel} onChange={e=>setShowLevel(e.target.checked)}/> Level reference</label>}</div>
        <Scene {...{layout,result,showLevel,slice,sliceEnabled,config,cameraEnabled,cameraPose,cuboidEnabled}} maxHeight={Number(options.maxHeight)}/>
        <div className="support-summary"><div className="support-legend"><span><i className="support-dot robust"/>{options.mode === 'yaw' ? 'Level · every yaw' : options.mode==='fixed' ? `Level · yaw ${options.yaw}°` : 'All sampled tilts'}</span>{options.mode === 'tilt' && <span><i className="support-dot level"/>Level reference</span>}<span><i className="support-dot markers"/>Installed markers</span></div><span className="support-axis-legend" aria-label="World frame axes: X red, Y green, Z blue"><b className="axis-x">X</b><b className="axis-y">Y</b><b className="axis-z">Z</b><span>world FLU</span></span>{(cameraEnabled||cuboidEnabled)&&<span className="support-overlay-legend">{cameraEnabled&&<span className="camera">Orange · camera / ground</span>}{cuboidEnabled&&<span className="tiles">Purple · tile cuboids</span>}</span>}<span className="support-interaction">{sliceEnabled&&<span className="support-plane-legend">Cyan · height slice · </span>}Drag to orbit · scroll to zoom</span></div>
        {result?.type === 'continuous' ? <div className="support-results"><div className="support-volume"><span>ESTIMATED VOLUME</span><strong>{result.volume.toFixed(3)} <small>m³</small></strong></div><span className="support-result-meta">{result.blocks} complete blocks · {result.mode==='fixed' ? `fixed yaw ${result.yaw}°` : 'continuous 0–360° yaw guarantee'} · {result.n} × {result.n} surface mesh{result.upper < result.maxHeight && <> · dot-area ceiling {result.upper.toFixed(2)} m</>}</span><HeightSlice value={slice} enabled={sliceEnabled} onChange={setSlice} onEnable={setSliceEnabled} min={result.minHeight} max={result.maxHeight}/>{!result.positions.length && <p>No supported region within the selected heights and configured ROI.</p>}</div> : result?<div className="support-results"><div className="support-volume"><span>ESTIMATED VOLUME</span><strong>{result.volume.toFixed(3)} <small>m³</small></strong></div><span className="support-result-meta">{result.robust.length.toLocaleString()} supported sample cells · {result.attitudes} attitudes · cell {result.cell.map(v=>v.toFixed(3)).join(' × ')} m</span><HeightSlice value={slice} enabled={sliceEnabled} onChange={setSlice} onEnable={setSliceEnabled} min={Number(options.minHeight)} max={Number(options.maxHeight)}/>{!result.robust.length&&<p>No sampled position supports the full tilt envelope. Reduce tilt or enlarge the layout.</p>}</div>:<p className="support-results">Choose your flight envelope, then calculate to display the supported region.</p>}
        {error&&<p role="alert" className="generator-error support-error">{error}</p>}
        </div>
        <details className="panel support-method"><summary><Info size={14}/><span>Model assumptions and geometry</span><span className="support-method-tag">{options.mode !== 'tilt' ? 'ANALYTIC' : 'SAMPLED'}</span></summary>
        {options.mode === 'fixed' ? <div className="support-note"><p>A level, downward pinhole camera at the selected yaw. Each complete block defines four image-edge inequalities; their intersection determines its continuous flight region. The height slice is a union of the resulting rotated rectangles, computed directly from the marker positions.</p><p>The whole region stays visible while the cyan plane highlights the slice. Camera offset, intrinsics and the configured processing ROI are included. Lens distortion and measured image quality are excluded.</p></div> : options.mode === 'yaw' ? <div className="support-note"><p>Level means the camera optical axis points straight down. Yaw is unrestricted over the full 360°. The green surface bounds positions where the same complete 2×2 flight block stays visible at every yaw, including full marker disks and the rotating camera offset. This is a conservative block guarantee; switching blocks during yaw can support additional positions.</p><p>Analytic pinhole model: ρ(h) = minⱼ[sⱼ(h + t_z) + bⱼ] − marker radius. Let D(x,y) be the smallest farthest-corner distance among installed blocks. Support requires D(x,y) ≤ ρ(h), with the projected dot area inside the detector limits. The mesh interpolates this continuous height boundary; surface detail changes rendering accuracy, not the yaw guarantee in the equation.</p><p>Intrinsics and the configured processing ROI are used. Lens distortion is excluded from this analytic model{result?.distorted ? ' (the loaded calibration has nonzero distortion)' : ''}. Exposure, occlusion, association and solver quality need separate validation.</p></div> : <p className="support-note">Green cells pass every sampled roll and pitch in [−max, +max] at the fixed yaw. Combined roll and pitch can produce a larger total lean angle. Cell centers and 16 points around each dot are sampled, so boundaries and intermediate attitudes are approximate. The model requires four separated, non-collinear markers nearest the image center, with full dots inside the image and sufficient projected area. Exposure, occlusion, association and pose quality still require flight validation.</p>}
        </details>
      </div>
    </div>
  </section>;
}
