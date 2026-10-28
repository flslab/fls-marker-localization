import test from 'node:test';
import assert from 'node:assert/strict';
import { supportLayout, cameraPose, project, poseSupported, computeSupport, validateCamera } from './flightSupport.js';
const config={camera:{width:640,height:400},calibration:{camera_matrix:[[475,0,320],[0,475,200],[0,0,1]],distortion_coefficients:[]},camera_mount:{position_drone_flu_m:[.02,0,0],camera_to_drone_rotation:[[0,-1,0],[-1,0,0],[0,0,-1]]}};
const raw={rows:2,cols:2,cell_spacing:.155,marker_size:.01,grid_origin:[.0775,.0775,0]};
const layout=supportLayout(raw);const options={tilt:0,yaw:0,minHeight:.2,maxHeight:.8,resolution:8,attitudeSteps:3,margin:10,minArea:8};
test('configured camera offset and rotated axes affect projection',()=>{
 const pose=cameraPose([-.02,0,.5],[0,0,0],config.camera_mount);
 assert.deepEqual(pose.center,[0,0,.5]);
 assert.deepEqual(project([.1,0,0],pose,config.calibration),[320,105]);
 assert.deepEqual(project([0,.1,0],pose,config.calibration),[225,200]);
 assert.equal(project([0,0,1],pose,config.calibration),null);
});
test('yaw rotates both camera offset and image frame',()=>{
 const pose=cameraPose([0,0,.5],[0,0,Math.PI/2],config.camera_mount);
 assert.ok(Math.abs(pose.center[0])<1e-12);assert.ok(Math.abs(pose.center[1]-.02)<1e-12);
});
test('whole dots must fit and be large enough, with nondegenerate support',()=>{
 assert.equal(poseSupported(layout,config,[-.02,0,.5],[0,0,0],options),true);
 assert.equal(poseSupported(layout,config,[1,0,.5],[0,0,0],options),false);
 assert.equal(poseSupported(layout,config,[-.02,0,.5],[0,0,0],{...options,minArea:1000}),false);
 const line={...layout,markers:[[0,0,0],[.1,0,0],[.2,0,0],[.3,0,0]]};
 assert.equal(poseSupported(line,config,[.13,0,.8],[0,0,0],options),false);
});
test('finite selected tiles generate only installed HyperGrid dots',()=>{
 const map={schema:'fls-marker-grid',grid_origin:[0,0,0],hypergrid:{marker_spacing:.155,marker_diameter:.01,infinite:true},mygrid:{tiles:[{i:0,j:0}]}};
 assert.equal(supportLayout(map).markers.length,4);
 assert.throws(()=>supportLayout({...map,mygrid:{tiles:[]}}),/4–1600/);
});
test('sampled tilt region is contained in level reference and zero tilt matches it',()=>{
 const zero=computeSupport({layout,config,options});assert.ok(zero.level.length>0);assert.deepEqual(zero.robust,zero.level);
 const tilted=computeSupport({layout,config,options:{...options,tilt:20}});
 assert.ok(tilted.robust.length<zero.robust.length);
 assert.ok(tilted.robust.every(p=>tilted.level.some(q=>q.join()===p.join())));
 assert.equal(tilted.volume,tilted.robust.length*tilted.cell.reduce((a,b)=>a*b));
});
test('invalid camera and sampling settings report useful errors',()=>{
 assert.throws(()=>validateCamera({...config,camera_mount:{...config.camera_mount,camera_to_drone_rotation:[[1,0,0],[0,1,0],[0,0,-1]]}}),/proper rotation/);
 assert.throws(()=>computeSupport({layout,config,options:{...options,maxHeight:.1}}),/Check heights/);
});
test('enabled processing crop restricts the usable image',()=>{
 const cropped={...config,processing_crop:{enabled:true,x:300,y:190,width:20,height:20}};
 validateCamera(cropped);
 assert.equal(poseSupported(layout,cropped,[-.02,0,.5],[0,0,0],options),false);
 assert.throws(()=>validateCamera({...cropped,processing_crop:{enabled:true,x:600,y:0,width:100,height:400}}),/crop/);
});
