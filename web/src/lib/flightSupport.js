// Geometric support estimate; attitude and position samples are not a certificate.
const finite = (x) => typeof x === 'number' && Number.isFinite(x);
const vec = (v) => Array.isArray(v) && v.length === 3 && v.every(finite);
export function supportLayout(raw) {
  let markers = []; const tileFootprints=[]; let diameter; let spacing;
  if (raw.schema === 'fls-marker-grid') {
    const s = raw.hypergrid?.marker_spacing; const origin = raw.grid_origin;
    if (!finite(s) || s <= 0 || !vec(origin)) throw new Error('Invalid HyperGrid geometry.');
    diameter = raw.hypergrid.marker_diameter; spacing = s;
    // Selected tiles explicitly define the finite installation for this view.
    if (!Array.isArray(raw.mygrid?.tiles) || raw.mygrid.tiles.length > 400) throw new Error('Select between 1 and 400 installed tiles.');
    for (const tile of raw.mygrid.tiles) {
      if (![tile.i, tile.j].every(Number.isInteger)) throw new Error('Invalid tile coordinates.');
      const x=origin[0]+2*tile.i*s,y=origin[1]+2*tile.j*s;
      tileFootprints.push([[x-s,x+s],[y-s,y+s]]);
      for (const [x, y] of [[-.5,.5],[.5,.5],[.5,-.5],[-.5,-.5]]) markers.push([origin[0]+(2*tile.i+x)*s, origin[1]+(2*tile.j+y)*s, origin[2]]);
    }
  } else {
    if (!Number.isInteger(raw.rows) || !Number.isInteger(raw.cols) || raw.rows < 2 || raw.cols < 2 || raw.rows * raw.cols > 1600 || !finite(raw.cell_spacing) || raw.cell_spacing <= 0 || !vec(raw.grid_origin)) throw new Error('Expected a localization grid or fls-marker-grid JSON.');
    diameter = raw.marker_size; spacing = raw.cell_spacing;
    for(let r=0;r<raw.rows;r+=2) for(let c=0;c<raw.cols;c+=2) {
      tileFootprints.push([[raw.grid_origin[0]-(Math.min(r+2,raw.rows)-.5)*spacing,raw.grid_origin[0]-(r-.5)*spacing],[raw.grid_origin[1]-(Math.min(c+2,raw.cols)-.5)*spacing,raw.grid_origin[1]-(c-.5)*spacing]]);
    }
    for (let r=0;r<raw.rows;r++) for(let c=0;c<raw.cols;c++) markers.push([raw.grid_origin[0]-r*raw.cell_spacing,raw.grid_origin[1]-c*raw.cell_spacing,raw.grid_origin[2]]);
  }
  markers = [...new Map(markers.map(p=>[p.join(','),p])).values()];
  if (markers.length < 4 || markers.length > 1600 || !finite(diameter) || diameter <= 0) throw new Error('Layout needs 4–1600 markers and a positive marker diameter.');
  const bounds = [0,1].map(axis=>[Math.min(...markers.map(p=>p[axis]))-diameter/2,Math.max(...markers.map(p=>p[axis]))+diameter/2]);
  return { markers, tileFootprints, diameter, spacing, bounds, floor: markers[0][2] };
}
export function rotation(roll,pitch,yaw) {
  const a=Math.cos(roll),b=Math.sin(roll),c=Math.cos(pitch),d=Math.sin(pitch),e=Math.cos(yaw),f=Math.sin(yaw);
  return [[e*c,e*d*b-f*a,e*d*a+f*b],[f*c,f*d*b+e*a,f*d*a-e*b],[-d,c*b,c*a]];
}
const multiply=(A,v)=>A.map(row=>row.reduce((s,x,i)=>s+x*v[i],0));
const transpose=A=>A[0].map((_,i)=>A.map(row=>row[i]));
export function cameraPose(position, attitude, mount) {
  const R=rotation(...attitude); const offset=multiply(R,mount.position_drone_flu_m);
  const inverse=transpose(R), mountInverse=transpose(mount.camera_to_drone_rotation);
  return { center: position.map((v,i)=>v+offset[i]), transform:v=>multiply(mountInverse,multiply(inverse,v)) };
}
export function project(point, pose, calibration) {
  const [X,Y,Z]=pose.transform(point.map((v,i)=>v-pose.center[i])); if(Z<=1e-8) return null;
  const x=X/Z,y=Y/Z,r2=x*x+y*y; const [k1=0,k2=0,p1=0,p2=0,k3=0,k4=0,k5=0,k6=0]=calibration.distortion_coefficients;
  const radial=(1+k1*r2+k2*r2*r2+k3*r2**3)/(1+k4*r2+k5*r2*r2+k6*r2**3);
  const xd=x*radial+2*p1*x*y+p2*(r2+2*x*x), yd=y*radial+p1*(r2+2*y*y)+2*p2*x*y;
  const K=calibration.camera_matrix; const result=[K[0][0]*xd+K[0][2],K[1][1]*yd+K[1][2]];
  return result.every(finite)?result:null;
}
export function validateCamera(config) {
  if (!config.camera || ![config.camera.width,config.camera.height].every(v=>finite(v)&&v>0) || !config.calibration || !Array.isArray(config.calibration.camera_matrix) || config.calibration.camera_matrix.length!==3 || !config.calibration.camera_matrix.every(vec) || ![config.calibration.camera_matrix[0][0],config.calibration.camera_matrix[1][1]].every(v=>v>0) || !Array.isArray(config.calibration.distortion_coefficients) || ![0,4,5,8].includes(config.calibration.distortion_coefficients.length) || !config.calibration.distortion_coefficients.every(finite) || !vec(config.camera_mount?.position_drone_flu_m) || !Array.isArray(config.camera_mount?.camera_to_drone_rotation) || config.camera_mount.camera_to_drone_rotation.length!==3 || !config.camera_mount.camera_to_drone_rotation.every(vec)) throw new Error('Camera JSON needs intrinsics, supported distortion (0/4/5/8 coefficients), image size and camera_mount.');
  if(config.processing_crop?.enabled) {
    const c=config.processing_crop;
    if(![c.x,c.y,c.width,c.height].every(finite)||c.x<0||c.y<0||c.width<=0||c.height<=0||c.x+c.width>config.camera.width||c.y+c.height>config.camera.height) throw new Error('Processing crop must fit in the camera image.');
  }
  const R=config.camera_mount.camera_to_drone_rotation;
  const det=R[0][0]*(R[1][1]*R[2][2]-R[1][2]*R[2][1])-R[0][1]*(R[1][0]*R[2][2]-R[1][2]*R[2][0])+R[0][2]*(R[1][0]*R[2][1]-R[1][1]*R[2][0]);
  if(Math.abs(det-1)>1e-4 || R.some((a,i)=>R.some((b,j)=>Math.abs(a.reduce((s,x,k)=>s+x*b[k],0)-(i===j?1:0))>1e-4))) throw new Error('Camera mount must be a proper rotation matrix.');
}
export function processingRoi(config) {
  return config.processing_crop?.enabled ? config.processing_crop : { x:0, y:0, width:config.camera.width, height:config.camera.height };
}
export function poseSupported(layout, config, position, attitude, options) {
  const pose=cameraPose(position,attitude,config.camera_mount); const visible=[];
  const {width,height}=config.camera;
  const crop=processingRoi(config);
  const inside=p=>p&&p[0]>=crop.x&&p[0]<=crop.x+crop.width&&p[1]>=crop.y&&p[1]<=crop.y+crop.height;
  for(const marker of layout.markers) {
    const center=project(marker,pose,config.calibration); if(!inside(center)) continue;
    const contour=[];
    for(let n=0;n<16;n++) { const t=n*Math.PI/8; const p=project([marker[0]+layout.diameter/2*Math.cos(t),marker[1]+layout.diameter/2*Math.sin(t),marker[2]],pose,config.calibration); if(!inside(p)) break; contour.push(p); }
    if(contour.length!==16) continue;
    let area=0; for(let n=0;n<16;n++){const a=contour[n],b=contour[(n+1)%16];area+=a[0]*b[1]-b[0]*a[1];}
    area=Math.abs(area)/2;
    if(area<options.minArea || area>(config.detector?.maximum_area??Infinity)) continue;
    visible.push({marker,center,radius:Math.max(...contour.map(p=>Math.hypot(p[0]-center[0],p[1]-center[1])))});
  }
  if(visible.length<4) return false;
  // Match the default nearest-four point selection and reject degenerate geometry.
  visible.sort((a,b)=>Math.hypot(a.center[0]-width/2,a.center[1]-height/2)-Math.hypot(b.center[0]-width/2,b.center[1]-height/2));
  const points=visible.slice(0,4);
  for(let i=0;i<4;i++) for(let j=i+1;j<4;j++) if(Math.hypot(points[i].center[0]-points[j].center[0],points[i].center[1]-points[j].center[1])<=points[i].radius+points[j].radius) return false;
  return points.some((a,i)=>points.some((b,j)=>j>i&&points.some((c,k)=>k>j&&Math.abs((b.marker[0]-a.marker[0])*(c.marker[1]-a.marker[1])-(b.marker[1]-a.marker[1])*(c.marker[0]-a.marker[0]))>1e-10)));
}
export function computeSupport({layout,config,options}, progress=()=>{}) {
  validateCamera(config);
  const {tilt,yaw,minHeight,maxHeight,resolution,attitudeSteps,padding=0}=options;
  if(![tilt,yaw,minHeight,maxHeight,resolution,attitudeSteps,options.minArea,padding].every(finite)||padding<0||padding>10||tilt<0||tilt>45||minHeight<=0||maxHeight<=minHeight||options.minArea<=0||!Number.isInteger(resolution)||resolution<8||resolution>32||![3,5,7].includes(attitudeSteps)) throw new Error('Check heights, tilt (0–45°), area and sampling settings.');
  const n=resolution,nz=16,dx=(layout.bounds[0][1]-layout.bounds[0][0]+2*padding)/n,dy=(layout.bounds[1][1]-layout.bounds[1][0]+2*padding)/n,dz=(maxHeight-minHeight)/nz;
  const attitudes=[]; const angles=tilt===0?[0]:Array.from({length:attitudeSteps},(_,i)=>(-tilt+2*tilt*i/(attitudeSteps-1))*Math.PI/180);
  for(const r of angles) for(const p of angles) attitudes.push([r,p,yaw*Math.PI/180]);
  const robust=[],level=[];
  for(let k=0;k<nz;k++) {
    for(let i=0;i<n;i++) for(let j=0;j<n;j++) {
      const point=[layout.bounds[0][0]-padding+(i+.5)*dx,layout.bounds[1][0]-padding+(j+.5)*dy,layout.floor+minHeight+(k+.5)*dz];
      if(!poseSupported(layout,config,point,[0,0,yaw*Math.PI/180],options)) continue;
      level.push(point);
      if(attitudes.every(a=>poseSupported(layout,config,point,a,options))) robust.push(point);
    }
    progress((k+1)/nz);
  }
  return {robust,level,searchBounds:layout.bounds.map(([min,max])=>[min-padding,max+padding]),cell:[dx,dy,dz],volume:robust.length*dx*dy*dz,attitudes:attitudes.length};
}
