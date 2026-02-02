// AudioWorklet processor for microphone input
// This replaces the deprecated ScriptProcessorNode

class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input[0]) {
            return true;
        }

        const inputData = input[0];
        const int16Data = new Int16Array(inputData.length);

        // Convert Float32 to Int16
        for (let i = 0; i < inputData.length; i++) {
            const s = Math.max(-1, Math.min(1, inputData[i]));
            int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        // Send audio data to main thread
        this.port.postMessage(int16Data);

        return true; // Keep processor alive
    }
}

registerProcessor('audio-processor', AudioProcessor);
