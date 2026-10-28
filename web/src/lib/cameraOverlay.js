import { processingRoi, rotation } from './flightSupport.js';

// RPY rotates the configured downward camera mount; position is the camera
// center itself, in world XY and height above the marker plane.
export function cameraOverlayGeometry(config, pose, floor, fallbackDepth = 3) {
  if (![pose.x,pose.y,pose.height,pose.roll,pose.pitch,pose.yaw,floor].every(Number.isFinite) || pose.height <= 0) return null;
  const center=[pose.x,pose.y,floor+pose.height];
  const R=rotation(...[pose.roll,pose.pitch,pose.yaw].map(a=>a*Math.PI/180));
  const mount=config.camera_mount.camera_to_drone_rotation;
  const worldVector=v=>R.map(row=>row.reduce((sum,r,i)=>sum+r*mount[i].reduce((s,m,j)=>s+m*v[j],0),0));
  const K=config.calibration.camera_matrix,roi=processingRoi(config);
  const pixels=[[roi.x,roi.y],[roi.x+roi.width,roi.y],[roi.x+roi.width,roi.y+roi.height],[roi.x,roi.y+roi.height]];
  const rays=pixels.map(([u,v])=>worldVector([(u-K[0][2])/K[0][0],(v-K[1][2])/K[1][1],1]));
  const at=(ray,depth)=>center.map((v,i)=>v+depth*ray[i]);
  const hits=rays.map(ray=>ray[2]<-1e-8?at(ray,-pose.height/ray[2]):null);
  const bounded=hits.every(Boolean);
  const nearDepth=bounded?Math.min(.025,...rays.map(ray=>-pose.height/ray[2]/2)):.025;
  const near=rays.map(ray=>at(ray,nearDepth));
  const far=bounded?hits:rays.map(ray=>at(ray,Math.max(.1,fallbackDepth)));
  return {center,near,far,ground:bounded?hits:null,axes:[[1,0,0],[0,1,0],[0,0,1]].map(worldVector)};
}
