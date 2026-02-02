import { useRef, useCallback } from 'react';

export function useAudio() {
    const audioContextRef = useRef<AudioContext | null>(null);

    const playAudio = useCallback(async (audioData: string, sampleRate: number = 24000) => {
        try {
            if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
                audioContextRef.current = new AudioContext({ sampleRate });
            }

            const audioContext = audioContextRef.current;

            // Decode base64 to array buffer
            const binaryString = atob(audioData);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }

            // Convert to Float32Array
            const int16Array = new Int16Array(bytes.buffer);
            const float32Array = new Float32Array(int16Array.length);
            for (let i = 0; i < int16Array.length; i++) {
                float32Array[i] = int16Array[i] / 32768.0;
            }

            // Create audio buffer
            const audioBuffer = audioContext.createBuffer(1, float32Array.length, sampleRate);
            audioBuffer.getChannelData(0).set(float32Array);

            // Play
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);
            source.start(0);

        } catch (error) {
            console.error('Error playing audio:', error);
        }
    }, []);

    const stop = useCallback(() => {
        if (audioContextRef.current) {
            audioContextRef.current.close();
            audioContextRef.current = null;
        }
    }, []);

    return {
        playAudio,
        stop
    };
}
