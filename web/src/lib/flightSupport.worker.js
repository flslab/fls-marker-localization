import { computeLevelRegion } from './levelFlightRegion.js';
import { computeSupport } from './flightSupport.js';
self.onmessage=({data})=>{try{self.postMessage({result:(['yaw','fixed'].includes(data.options.mode) ? computeLevelRegion : computeSupport)(data,progress=>self.postMessage({progress}))});}catch(error){self.postMessage({error:error.message});}};
