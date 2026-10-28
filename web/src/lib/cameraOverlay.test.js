import test from 'node:test';
import assert from 'node:assert/strict';
import { cameraOverlayGeometry } from './cameraOverlay.js';
import { cameraPose, project, supportLayout } from './flightSupport.js';
const config={camera:{width:640,height:400},calibration:{camera_matrix:[[475,0,320],[0,475,200],[0,0,1]],distortion_coefficients:[]},camera_mount:{position_drone_flu_m:[0,0,0],camera_to_drone_rotation:[[0,-1,0],[-1,0,0],[0,0,-1]]}};
const pose={x:.1,y:.2,height:1,roll:0,pitch:0,yaw:0};
test('camera frustum corners meet the floor and project to the configured ROI corners',()=>{
 const cropped={...config,processing_crop:{enabled:true,x:120,y:0,width:400,height:400}};
 for(const yaw of [0,30,90]) {
   const shape=cameraOverlayGeometry(cropped,{...pose,pitch:15,yaw},.3);
   assert.ok(shape.ground);
   const transform=cameraPose(shape.center,[0,15*Math.PI/180,yaw*Math.PI/180],config.camera_mount);
   const expected=[[120,0],[520,0],[520,400],[120,400]];
   shape.ground.forEach((p,i)=>{
     assert.ok(Math.abs(p[2]-.3)<1e-12);
     const pixel=project(p,transform,cropped.calibration);
     assert.ok(Math.hypot(pixel[0]-expected[i][0],pixel[1]-expected[i][1])<1e-8);
   });
 }
});
test('horizon crossing and invalid camera pose do not fabricate a ground polygon',()=>{
 const shape=cameraOverlayGeometry(config,{...pose,pitch:90},0);
 assert.equal(shape.ground,null);
 assert.ok(shape.far.flat().every(Number.isFinite));
 assert.equal(cameraOverlayGeometry(config,{...pose,height:0},0),null);
 assert.equal(cameraOverlayGeometry(config,{...pose,x:''},0),null);
});
test('low camera heights keep the near frustum above the floor',()=>{
 const shape=cameraOverlayGeometry(config,{...pose,height:.01},0);
 assert.ok(shape.near.every(p=>p[2]>0));
});
test('tile cuboids use full tile boundaries and preserve missing tile gaps',()=>{
 const map={schema:'fls-marker-grid',grid_origin:[1,2,.3],hypergrid:{marker_spacing:.155,marker_diameter:.01},mygrid:{tiles:[{i:0,j:0},{i:2,j:0}]}};
 const layout=supportLayout(map);
 assert.equal(layout.tileFootprints.length,2);
 assert.deepEqual(layout.tileFootprints[0],[[1-.155,1+.155],[2-.155,2+.155]]);
 assert.ok(layout.tileFootprints[0][0][1]<layout.tileFootprints[1][0][0]);
});
