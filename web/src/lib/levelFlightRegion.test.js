import test from 'node:test';
import assert from 'node:assert/strict';
import { yawKernel, completeBlocks, minimumYawHeight, continuousMesh, computeLevelRegion, analyticHeightSlice, minimumFixedYawHeight, fixedYawAxes } from './levelFlightRegion.js';
import { supportLayout, cameraPose, project } from './flightSupport.js';
const config={camera:{width:640,height:400},calibration:{camera_matrix:[[475,0,320],[0,475,200],[0,0,1]],distortion_coefficients:[]},processing_crop:{enabled:true,x:64,y:40,width:512,height:320},camera_mount:{position_drone_flu_m:[0,0,0],camera_to_drone_rotation:[[0,-1,0],[-1,0,0],[0,0,-1]]}};
const layout=supportLayout({rows:2,cols:2,cell_spacing:.155,marker_size:.01,grid_origin:[.0775,.0775,0]});
const options={minHeight:.1,maxHeight:1.8,minArea:8,padding:.2,surfaceResolution:24};
test('centered square block has the closed-form every-yaw height',()=>{
 const k=yawKernel(config),blocks=completeBlocks(layout);
 const expected=(Math.hypot(.0775,.0775)+.005)/(160/475);
 assert.ok(Math.abs(minimumYawHeight(0,0,blocks,.005,k)-expected)<1e-12);
});
test('analytic bound keeps complete disks inside at all yaw, including asymmetric mount and principal point',()=>{
 const asymmetric=structuredClone(config);
 asymmetric.camera_mount.position_drone_flu_m=[.02,.013,.008];
 asymmetric.calibration.camera_matrix[0][2]=324;asymmetric.calibration.camera_matrix[1][2]=199;
 const k=yawKernel(asymmetric),blocks=completeBlocks(layout);
 const height=minimumYawHeight(.05,-.03,blocks,.005,k)+1e-8;
 for(let degrees=0;degrees<360;degrees++) {
   const pose=cameraPose([.05,-.03,height],[0,0,degrees*Math.PI/180],asymmetric.camera_mount);
   for(const marker of layout.markers) for(let a=0;a<64;a++) {
     const t=a*Math.PI/32,p=project([marker[0]+.005*Math.cos(t),marker[1]+.005*Math.sin(t),0],pose,asymmetric.calibration);
     assert.ok(p[0]>=64-1e-7&&p[0]<=576+1e-7&&p[1]>=40-1e-7&&p[1]<=360+1e-7);
   }
 }
});
test('missing installed tile does not create blocks across the gap',()=>{
 const l=supportLayout({schema:'fls-marker-grid',grid_origin:[0,0,0],hypergrid:{marker_spacing:.155,marker_diameter:.01},mygrid:{tiles:[{i:0,j:0},{i:2,j:0}]}});
 assert.equal(completeBlocks(l).length,2);
});
test('continuous mesh integrates and clips planar height boundaries correctly',()=>{
 const flat=continuousMesh([[0,0,.25],[0,1,.25],[1,0,.25],[1,1,.25]],1,1,0);
 assert.ok(Math.abs(flat.volume-.75)<1e-12);
 const slope=continuousMesh([[0,0,0],[0,1,0],[1,0,1],[1,1,1]],1,.5,0);
 assert.ok(Math.abs(slope.volume-.125)<1e-12);
 assert.ok(slope.positions.length>0);
 assert.ok(slope.positions.every(Number.isFinite));
});
test('region upper bound follows dot-area ceiling and empty bands produce no surface',()=>{
 const result=computeLevelRegion({layout,config,options});
 assert.ok(result.positions.length>0);
 assert.ok(Math.abs(result.upper-Math.sqrt(Math.PI*.005**2*475**2/8))<1e-12);
 const empty=computeLevelRegion({layout,config,options:{...options,minArea:1e8}});
 assert.equal(empty.positions.length,0);assert.equal(empty.volume,0);
});
test('tilted mount is rejected for the analytic level-camera model',()=>{
 const c=structuredClone(config); c.camera_mount.camera_to_drone_rotation=[[1,0,0],[0,0,-1],[0,1,0]];
 assert.throws(()=>yawKernel(c),/downward/);
});

test('arc slices are independent of surface detail and never bridge unsupported gaps',()=>{
 const l=supportLayout({rows:2,cols:3,cell_spacing:.155,marker_size:.01,grid_origin:[.0775,.0775,0]});
 const coarse=computeLevelRegion({layout:l,config,options:{...options,surfaceResolution:24}});
 const fine=computeLevelRegion({layout:l,config,options:{...options,surfaceResolution:80}});
 const height=(.16+.005)/(160/475);
 const positions=analyticHeightSlice(coarse,height);
 assert.ok(positions.length>0);
 assert.deepEqual(positions,analyticHeightSlice(fine,height));
 for(let i=0;i<positions.length;i+=9) {
   const x=(positions[i]+positions[i+3]+positions[i+6])/3;
   const y=(positions[i+1]+positions[i+4]+positions[i+7])/3;
   assert.ok(minimumYawHeight(x,y,coarse.blockMarkers,.005,coarse.kernel)<=height+1e-9);
 }
 assert.ok(minimumYawHeight(0,-.0775,coarse.blockMarkers,.005,coarse.kernel)>height);
});
test('arc slices respect height ceilings and the search boundary',()=>{
 const result=computeLevelRegion({layout,config,options});
 assert.deepEqual(analyticHeightSlice(result,result.upper+.001),[]);
 assert.deepEqual(analyticHeightSlice(result,result.lower-.001),[]);
 for(let i=0,p=analyticHeightSlice(result,1);i<p.length;i+=3) {
   assert.ok(p[i]>=result.searchBounds[0][0]-1e-10&&p[i]<=result.searchBounds[0][1]+1e-10);
   assert.ok(p[i+1]>=result.searchBounds[1][0]-1e-10&&p[i+1]<=result.searchBounds[1][1]+1e-10);
   assert.equal(p[i+2],result.floor+1);
 }
});

test('fixed yaw has the rectangular single-block minimum and includes the every-yaw guarantee',()=>{
 const blocks=completeBlocks(layout),kernel=yawKernel(config);
 const fixed=minimumFixedYawHeight(0,0,blocks,.005,kernel,fixedYawAxes(config,0));
 assert.ok(Math.abs(fixed-(.0775+.005)*475/160)<1e-12);
 assert.ok(fixed<minimumYawHeight(0,0,blocks,.005,kernel));
});
test('fixed-yaw rectangles project inside the image and account for the rotated block span',()=>{
 let referenceArea;
 for(const yaw of [0,45,90]) {
   const result=computeLevelRegion({layout,config,options:{...options,mode:'fixed',yaw}});
   const positions=analyticHeightSlice(result,.5);
   let area=0;
   for(let i=0;i<positions.length;i+=9) {
     area+=Math.abs((positions[i+3]-positions[i])*(positions[i+7]-positions[i+1])-(positions[i+6]-positions[i])*(positions[i+4]-positions[i+1]))/2;
     for(let j=0;j<9;j+=3) {
       const pose=cameraPose([positions[i+j],positions[i+j+1],.5],[0,0,yaw*Math.PI/180],config.camera_mount);
       for(const marker of layout.markers) for(let a=0;a<32;a++) {
         const t=a*Math.PI/16;
         const p=project([marker[0]+.005*Math.cos(t),marker[1]+.005*Math.sin(t),0],pose,config.calibration);
         assert.ok(p[0]>=64-1e-8&&p[0]<=576+1e-8&&p[1]>=40-1e-8&&p[1]<=360+1e-8);
       }
     }
   }
   assert.ok(area>0);
   if(yaw===0) referenceArea=area;
   if(yaw===45) assert.ok(area<referenceArea);
   if(yaw===90) assert.ok(Math.abs(area-referenceArea)<1e-10);
 }
});

test('model bounds expand with height and contain support beyond the former fixed border',()=>{
 const low=computeLevelRegion({layout,config,options:{...options,mode:'fixed',yaw:30,maxHeight:1}});
 const high=computeLevelRegion({layout,config,options:{...options,mode:'fixed',yaw:30,maxHeight:3,minArea:.5,padding:0}});
 for(let axis=0;axis<2;axis++) {
   assert.ok(high.planeBounds[axis][1]-high.planeBounds[axis][0]>low.planeBounds[axis][1]-low.planeBounds[axis][0]);
   assert.ok(high.searchBounds[axis][1]>layout.bounds[axis][1]+.3);
 }
 const slice=analyticHeightSlice(high,2.5);
 assert.ok(slice.length>0);
 assert.ok(slice.some((value,i)=>i%3===0&&value>layout.bounds[0][1]+.3));
});
test('configured processing ROI defines bounds with no additional percentage margin',()=>{
 const full=structuredClone(config);full.processing_crop.enabled=false;
 const k=yawKernel(full);
 assert.equal(k.sides[0][0],320/475);
 assert.equal(k.sides[2][0],200/475);
 const roi=computeLevelRegion({layout,config,options});
 const frame=computeLevelRegion({layout,config:full,options});
 assert.ok(frame.volume>roi.volume);
});
