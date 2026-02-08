import { useRef, useCallback, useState } from 'react';

export function useAudio() {
    const audioContextRef = useRef<AudioContext | null>(null);
    const nextStartTimeRef = useRef<number>(0);
    const [isPlaying, setIsPlaying] = useState<boolean>(false);
    const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

    const getAudioContext = useCallback((sampleRate: number = 24000): AudioContext => {
        if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
            audioContextRef.current = new AudioContext({ sampleRate });
            nextStartTimeRef.current = 0;
        }
        return audioContextRef.current;
    }, []);

    const playAudio = useCallback(async (audioData: string, sampleRate: number = 24000) => {
        try {
            const audioContext = getAudioContext(sampleRate);

            // Resume audio context if suspended (browser autoplay policy)
            if (audioContext.state === 'suspended') {
                await audioContext.resume();
            }

            // Decode base64 to array buffer
            const binaryString = atob(audioData);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }

            // Convert Int16 to Float32
            const int16Array = new Int16Array(bytes.buffer);
            const float32Array = new Float32Array(int16Array.length);
            for (let i = 0; i < int16Array.length; i++) {
                float32Array[i] = int16Array[i] / 32768.0;
            }

            // Create audio buffer
            const audioBuffer = audioContext.createBuffer(1, float32Array.length, sampleRate);
            audioBuffer.getChannelData(0).set(float32Array);

            // Calculate when to start this chunk (queue sequentially)
            const currentTime = audioContext.currentTime;

            // If we've fallen behind or this is the first chunk, start now
            if (nextStartTimeRef.current < currentTime) {
                nextStartTimeRef.current = currentTime;
            }

            // Schedule playback
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);
            source.start(nextStartTimeRef.current);

            // Track active source
            activeSourcesRef.current.add(source);
            setIsPlaying(true);

            // Update next start time for the following chunk
            nextStartTimeRef.current += audioBuffer.duration;

            // Track when playback ends and cleanup
            source.onended = () => {
                // Disconnect and remove from active sources to prevent memory leak
                source.disconnect();
                activeSourcesRef.current.delete(source);

                // Check if there's more audio scheduled
                if (audioContext.currentTime >= nextStartTimeRef.current - 0.1) {
                    setIsPlaying(false);
                }
            };

        } catch (error) {
            console.error('Error playing audio:', error);
        }
    }, [getAudioContext]);

    const stop = useCallback(() => {
        // Stop all active sources
        activeSourcesRef.current.forEach(source => {
            try {
                source.stop();
                source.disconnect();
            } catch (e) {
                // Source may have already stopped
            }
        });
        activeSourcesRef.current.clear();

        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
            audioContextRef.current.close();
            audioContextRef.current = null;
        }
        nextStartTimeRef.current = 0;
        setIsPlaying(false);
    }, []);

    return {
        playAudio,
        stop,
        isPlaying
    };
}
