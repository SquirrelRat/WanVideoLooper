import { app } from "/scripts/app.js";

const max_segments = 10; 

/**
 * This is the main function that hides/shows inputs.
 * @param {LGraphNode} currentNode The instance of the node.
 */
function updateVisibility(currentNode) {
    if (!currentNode.wanSequencerAllInputs || currentNode.wanSequencerAllInputs.length === 0) {
        if (currentNode.inputs.length > 3) { 
            currentNode.wanSequencerAllInputs = [...currentNode.inputs];
        } else {
             return;
        }
    }
    
    const allInputs = currentNode.wanSequencerAllInputs;
    if (!allInputs) return;

    let lastConnectedSegment = 0;

    for (let i = 1; i <= max_segments; i++) {
        const live_mh = currentNode.inputs.find(s => s.name === `model_high_${i}`);
        const live_ml = currentNode.inputs.find(s => s.name === `model_low_${i}`);
        const live_c = currentNode.inputs.find(s => s.name === `clip_${i}`);

        if ((live_mh && live_mh.link != null) || 
            (live_ml && live_ml.link != null) || 
            (live_c && live_c.link != null)) {
            lastConnectedSegment = i;
        }
    }
    
    const segmentsToShow = Math.min(lastConnectedSegment + 1, max_segments);

    const visibleInputs = [];
    for (let i = 1; i <= segmentsToShow; i++) {
        const mh = allInputs.find(s => s.name === `model_high_${i}`);
        const ml = allInputs.find(s => s.name === `model_low_${i}`);
        const c = allInputs.find(s => s.name === `clip_${i}`);
        
        if (mh && ml && c) {
            visibleInputs.push(mh, ml, c);
        }
    }
    
    if (segmentsToShow === 0 && allInputs.length > 0) {
         const mh = allInputs.find(s => s.name === `model_high_1`);
         const ml = allInputs.find(s => s.name === `model_low_1`);
         const c = allInputs.find(s => s.name === `clip_1`);
         if (mh && ml && c) {
            visibleInputs.push(mh, ml, c);
         }
    }

    currentNode.inputs = visibleInputs;
    
    currentNode.setSize(currentNode.computeSize());
}


function onWanLoraSequencerConfigure() {
    if (!this.wanSequencerAllInputs) {
        this.wanSequencerAllInputs = [...this.inputs];
    }
    setTimeout(() => updateVisibility(this), 10);
}


function onWanLoraSequencerConnectionsChange(type, index, connected, link_info) {
    updateVisibility(this);
}

app.registerExtension({
    name: "WanVideo.LoraSequencer.DynamicInputs",
    async nodeCreated(node) {
        if (node.comfyClass !== "WanVideoLoraSequencer") {
            return;
        }

        
        node.onConfigure = onWanLoraSequencerConfigure;
        
        node.onConnectionsChange = onWanLoraSequencerConnectionsChange;

        setTimeout(() => updateVisibility(node), 10);
    }
});