import { validateCamera, processingRoi } from './flightSupport.js';

// Pinhole, downward level camera. These four affine functions are the
// distances from the drone origin to the eroded footprint's four sides.
// Intersecting its rotations over ALL yaw angles gives a disk of radius
// min_j(slope_j * cameraHeight + intercept_j) - markerRadius.
export function yawKernel(config) {
  validateCamera(config);
  const R = config.camera_mount.camera_to_drone_rotation;
  if (Math.abs(R[2][2] + 1) > 1e-6 || Math.abs(R[0][2]) > 1e-6 || Math.abs(R[1][2]) > 1e-6) throw new Error('The every-yaw model requires a level, downward camera mount.');
  const { width, height } = config.camera;
  const crop = processingRoi(config);
  const K = config.calibration.camera_matrix;
  if (Math.abs(K[0][1]) > 1e-8) throw new Error('The analytic model requires zero camera skew.');
  const left = (crop.x - K[0][2]) / K[0][0];
  const right = (crop.x + crop.width - K[0][2]) / K[0][0];
  const top = (crop.y - K[1][2]) / K[1][1];
  const bottom = (crop.y + crop.height - K[1][2]) / K[1][1];
  const t = config.camera_mount.position_drone_flu_m;
  const offsetU = R[0][0] * t[0] + R[1][0] * t[1];
  const offsetV = R[0][1] * t[0] + R[1][1] * t[1];
  const sides = [[-left, -offsetU], [right, offsetU], [-top, -offsetV], [bottom, offsetV]];
  if (sides.some(([slope]) => slope <= 0)) throw new Error('The usable image must contain the principal point for this every-yaw model.');
  return { sides, cameraZ: t[2], fx: K[0][0], fy: K[1][1] };
}

export function completeBlocks(layout) {
  const spacing = layout.spacing;
  if (!(spacing > 0)) throw new Error('A regular marker spacing is required.');
  const anchor = layout.markers[0];
  const key = (x, y) => `${Math.round((x - anchor[0]) / spacing)}:${Math.round((y - anchor[1]) / spacing)}`;
  const byCell = new Map(layout.markers.map(p => [key(p[0], p[1]), p]));
  const blocks = [];
  for (const p of layout.markers) {
    const block = [p, byCell.get(key(p[0] + spacing, p[1])), byCell.get(key(p[0], p[1] + spacing)), byCell.get(key(p[0] + spacing, p[1] + spacing))];
    if (block.every(Boolean)) blocks.push(block);
  }
  if (!blocks.length) throw new Error('The installed layout has no complete adjacent 2×2 flight block.');
  if (spacing <= layout.diameter) throw new Error('Marker disks overlap; a separated four-marker guarantee is unavailable.');
  return blocks;
}

export function minimumYawHeight(x, y, blocks, radius, kernel) {
  let distance = Infinity;
  for (const block of blocks) {
    const farthest = Math.max(...block.map(p => Math.hypot(p[0] - x, p[1] - y)));
    distance = Math.min(distance, farthest);
  }
  return Math.max(...kernel.sides.map(([slope, offset]) => (distance + radius - offset) / slope)) - kernel.cameraZ;
}

export function fixedYawAxes(config, yawDegrees) {
  const R = config.camera_mount.camera_to_drone_rotation;
  const yaw = yawDegrees * Math.PI / 180, c = Math.cos(yaw), s = Math.sin(yaw);
  return [0,1].map(i=>[c*R[0][i]-s*R[1][i],s*R[0][i]+c*R[1][i]]);
}

export function minimumFixedYawHeight(x,y,blocks,radius,kernel,axes) {
  let minimum = Infinity;
  for(const block of blocks) {
    let height = -Infinity;
    for(const p of block) {
      const u=axes[0][0]*(p[0]-x)+axes[0][1]*(p[1]-y);
      const v=axes[1][0]*(p[0]-x)+axes[1][1]*(p[1]-y);
      const distances=[-u,u,-v,v];
      height=Math.max(height,...kernel.sides.map(([s,b],i)=>(distances[i]+radius-b)/s-kernel.cameraZ));
    }
    minimum=Math.min(minimum,height);
  }
  return minimum;
}

// Clip a triangle against its linearly interpolated lower-height surface.
function clipAtHeight(triangle, height) {
  const output = [];
  for (let i = 0; i < triangle.length; i++) {
    const a = triangle[i], b = triangle[(i + 1) % triangle.length];
    if (a[2] <= height) output.push(a);
    if ((a[2] <= height) !== (b[2] <= height)) {
      const t = (height - a[2]) / (b[2] - a[2]);
      output.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), height]);
    }
  }
  return output;
}

export function continuousMesh(nodes, n, upper, floor, slice = null) {
  const positions = [], boundary = new Map(); let volume = 0;
  const push = (...points) => positions.push(...points.flat());
  const key = p => `${p[0].toFixed(10)}:${p[1].toFixed(10)}`;
  const add = triangle => {
    const polygon = clipAtHeight(triangle, slice ?? upper);
    for (let k = 1; k < polygon.length - 1; k++) {
      const a = polygon[0], b = polygon[k], c = polygon[k + 1];
      const top = p => [p[0], p[1], floor + (slice ?? upper)];
      push(top(a), top(b), top(c));
      if (slice !== null) continue;
      push([a[0],a[1],floor+a[2]], [c[0],c[1],floor+c[2]], [b[0],b[1],floor+b[2]]);
      const area = Math.abs((b[0]-a[0])*(c[1]-a[1])-(c[0]-a[0])*(b[1]-a[1])) / 2;
      volume += area * (upper - (a[2]+b[2]+c[2])/3);
    }
    if (slice !== null) return;
    for (let i = 0; i < polygon.length; i++) {
      const a = polygon[i], b = polygon[(i+1)%polygon.length];
      const id = [key(a),key(b)].sort().join('|');
      if (boundary.has(id)) boundary.delete(id); else boundary.set(id, [a,b]);
    }
  };
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const a=nodes[i*(n+1)+j], b=nodes[(i+1)*(n+1)+j], c=nodes[(i+1)*(n+1)+j+1], d=nodes[i*(n+1)+j+1];
    add([a,b,c]); add([a,c,d]);
  }
  for (const [a,b] of boundary.values()) {
    const low = p => [p[0],p[1],floor+p[2]], high = p => [p[0],p[1],floor+upper];
    push(low(a),low(b),high(b),low(a),high(b),high(a));
  }
  return { positions, volume };
}

// Footprints expand monotonically with height. Bounds at the top therefore
// enclose every lower slice; these are not a fixed border around the floor.
export function modelHorizontalBounds(layout, blocks, kernel, axes, height, padding=0) {
  const d=height+kernel.cameraZ, radius=layout.diameter/2;
  const sides=kernel.sides.map(([s,b])=>s*d+b-radius);
  const candidates=[];
  for(const block of blocks) {
    if(axes) {
      const values=axes.map(axis=>block.map(p=>axis[0]*p[0]+axis[1]*p[1]));
      const u0=Math.max(...values[0])-sides[1],u1=Math.min(...values[0])+sides[0];
      const v0=Math.max(...values[1])-sides[3],v1=Math.min(...values[1])+sides[2];
      if(u0>u1||v0>v1)continue;
      for(const [u,v] of [[u0,v0],[u1,v0],[u1,v1],[u0,v1]]) candidates.push([u*axes[0][0]+v*axes[1][0],u*axes[0][1]+v*axes[1][1]]);
    } else {
      const r=Math.min(...sides);
      const x0=Math.max(...block.map(p=>p[0]-r)),x1=Math.min(...block.map(p=>p[0]+r));
      const y0=Math.max(...block.map(p=>p[1]-r)),y1=Math.min(...block.map(p=>p[1]+r));
      if(x0>x1||y0>y1)continue;
      candidates.push([x0,y0],[x1,y1]);
    }
  }
  if(!candidates.length)return layout.bounds.map(([min,max])=>[min-padding,max+padding]);
  return [0,1].map(axis=>[Math.min(...candidates.map(p=>p[axis]))-padding,Math.max(...candidates.map(p=>p[axis]))+padding]);
}

export function computeLevelRegion({layout,config,options}, progress = () => {}) {
  const { minHeight, maxHeight, minArea, padding = 0, surfaceResolution = 48 } = options;
  if (![minHeight,maxHeight,minArea,padding].every(Number.isFinite) || minHeight <= 0 || maxHeight <= minHeight || minArea <= 0 || padding < 0 || padding > 10 || ![24,48,80].includes(surfaceResolution)) throw new Error('Check height, area, border and surface detail settings.');
  const kernel = yawKernel(config), blocks = completeBlocks(layout);
  const mode = options.mode === 'fixed' ? 'fixed' : 'yaw';
  if(mode==='fixed' && (!Number.isFinite(options.yaw)||Math.abs(options.yaw)>180)) throw new Error('Fixed yaw must be between −180° and 180°.');
  const axes = mode==='fixed' ? fixedYawAxes(config,options.yaw) : null;
  const radius = layout.diameter / 2;
  const factor = Math.PI * radius * radius * kernel.fx * kernel.fy;
  const upper = Math.min(maxHeight, Math.sqrt(factor/minArea) - kernel.cameraZ);
  const lower = Math.max(minHeight, Math.sqrt(factor/(config.detector?.maximum_area ?? Infinity)) - kernel.cameraZ);
  const searchBounds=modelHorizontalBounds(layout,blocks,kernel,axes,Math.max(lower,upper),padding);
  const planeBounds=modelHorizontalBounds(layout,blocks,kernel,axes,maxHeight,padding);
  const n = surfaceResolution, nodes = [];
  const dx = (searchBounds[0][1]-searchBounds[0][0])/n;
  const dy = (searchBounds[1][1]-searchBounds[1][0])/n;
  for (let i=0;i<=n;i++) {
    for (let j=0;j<=n;j++) {
      const x=searchBounds[0][0]+i*dx, y=searchBounds[1][0]+j*dy;
      nodes.push([x,y,Math.max(lower,mode==='fixed' ? minimumFixedYawHeight(x,y,blocks,radius,kernel,axes) : minimumYawHeight(x,y,blocks,radius,kernel))]);
    }
    progress((i+1)/(n+1));
  }
  const mesh = upper >= lower ? continuousMesh(nodes,n,upper,layout.floor) : {positions:[],volume:0};
  return { type:'continuous', mode, yaw:options.yaw??0, axes, blockMarkers:blocks, markerRadius:radius, searchBounds, planeBounds, ...mesh, nodes, n, upper, lower, blocks:blocks.length, kernel, floor:layout.floor, minHeight, maxHeight, distorted:config.calibration.distortion_coefficients.some(v=>v!==0) };
}

// A height slice is a union of four-disk intersections, one per block.
// Trace valid circle arcs directly so narrow gaps cannot be bridged by a grid.
export function analyticHeightSlice(result, height, arcStep = Math.PI / 180) {
  if (height < result.lower || height > result.upper) return [];
  const radius = Math.min(...result.kernel.sides.map(([s,b]) => s * (height + result.kernel.cameraZ) + b)) - result.markerRadius;
  if (radius <= 0 && result.mode!=='fixed') return [];
  const positions = [], twoPi = 2 * Math.PI;
  const normalize = a => (a + twoPi) % twoPi;
  for (const block of result.blockMarkers) {
    const boundary = [];
    if(result.mode==='fixed') {
      const d=height+result.kernel.cameraZ;
      const distances=result.kernel.sides.map(([s,b])=>s*d+b-result.markerRadius);
      const values=result.axes.map(axis=>block.map(p=>axis[0]*p[0]+axis[1]*p[1]));
      const u0=Math.max(...values[0])-distances[1],u1=Math.min(...values[0])+distances[0];
      const v0=Math.max(...values[1])-distances[3],v1=Math.min(...values[1])+distances[2];
      if(u0>=u1||v0>=v1) continue;
      for(const [u,v] of [[u0,v0],[u1,v0],[u1,v1],[u0,v1]]) boundary.push([u*result.axes[0][0]+v*result.axes[1][0],u*result.axes[0][1]+v*result.axes[1][1]]);
    } else for (let i = 0; i < block.length; i++) {
      const center = block[i], angles = [0, twoPi];
      for (let j = 0; j < block.length; j++) {
        if (j === i) continue;
        const dx = block[j][0] - center[0], dy = block[j][1] - center[1], distance = Math.hypot(dx,dy);
        if (distance > 2 * radius) continue;
        const base = Math.atan2(dy,dx), delta = Math.acos(Math.min(1,distance/(2*radius)));
        angles.push(normalize(base-delta), normalize(base+delta));
      }
      angles.sort((a,b)=>a-b);
      for (let j = 0; j < angles.length - 1; j++) {
        const start = angles[j], end = angles[j+1], mid = (start+end)/2;
        const x = center[0]+radius*Math.cos(mid), y = center[1]+radius*Math.sin(mid);
        if (!block.every(p=>Math.hypot(x-p[0],y-p[1]) <= radius+1e-10)) continue;
        const count = Math.max(1, Math.ceil((end-start)/arcStep));
        for (let k=0;k<=count;k++) {
          const a=start+(end-start)*k/count;
          boundary.push([center[0]+radius*Math.cos(a),center[1]+radius*Math.sin(a)]);
        }
      }
    }
    if (boundary.length < 3) continue;
    const centroid = [0,1].map(axis=>boundary.reduce((sum,p)=>sum+p[axis],0)/boundary.length);
    boundary.sort((a,b)=>Math.atan2(a[1]-centroid[1],a[0]-centroid[0])-Math.atan2(b[1]-centroid[1],b[0]-centroid[0]));
    let polygon = boundary;
    for (const [axis,limit,sign] of [[0,result.searchBounds[0][0],1],[0,result.searchBounds[0][1],-1],[1,result.searchBounds[1][0],1],[1,result.searchBounds[1][1],-1]]) {
      const clipped=[];
      for(let k=0;k<polygon.length;k++) {
        const a=polygon[k],b=polygon[(k+1)%polygon.length];
        const insideA=sign*(a[axis]-limit)>=0,insideB=sign*(b[axis]-limit)>=0;
        if(insideA) clipped.push(a);
        if(insideA!==insideB) {const t=(limit-a[axis])/(b[axis]-a[axis]);clipped.push([a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])]);}
      }
      polygon=clipped;
    }
    for(let k=1;k<polygon.length-1;k++) for(const p of [polygon[0],polygon[k],polygon[k+1]]) positions.push(p[0],p[1],result.floor+height);
  }
  return positions;
}
