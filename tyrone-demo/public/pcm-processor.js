// public/pcm-processor.js

// An AudioWorkletProcessor that forwards each audio frame's Float32Array to the main thread
class PCMProcessor extends AudioWorkletProcessor {
    process(inputs) {
        const input = inputs[0];
        if (input && input[0] && input[0].length) {
            // input[0][0] is a Float32Array of mono samples for this 128-sample frame
            this.port.postMessage(input[0]);
        }
        return true; // keep running
    }
}

registerProcessor("pcm-processor", PCMProcessor);
