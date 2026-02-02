import { useState, useRef, useCallback, useEffect } from 'react';

export function useMicrophone() {
    const [isActive, setIsActive] = useState(false);
    const streamRef = useRef<MediaStream | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const workletNodeRef = useRef<AudioWorkletNode | null>(null);
    const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

    const startMicrophone = useCallback(async (onData: (data: Int16Array) => void) => {
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });

            const audioContext = new AudioContext({ sampleRate: 16000 });

            // Load AudioWorklet processor
            await audioContext.audioWorklet.addModule('/audio-processor.js');

            const source = audioContext.createMediaStreamSource(mediaStream);
            const workletNode = new AudioWorkletNode(audioContext, 'audio-processor');

            // Listen for processed audio data
            workletNode.port.onmessage = (event) => {
                onData(event.data);
            };

            source.connect(workletNode);
            workletNode.connect(audioContext.destination);

            streamRef.current = mediaStream;
            audioContextRef.current = audioContext;
            workletNodeRef.current = workletNode;
            sourceRef.current = source;
            setIsActive(true);

            return mediaStream;
        } catch (error) {
            console.error('Error accessing microphone:', error);
            throw error;
        }
    }, []);

    const stopMicrophone = useCallback(() => {
        if (sourceRef.current) {
            sourceRef.current.disconnect();
            sourceRef.current = null;
        }

        if (workletNodeRef.current) {
            workletNodeRef.current.disconnect();
            workletNodeRef.current = null;
        }

        if (audioContextRef.current) {
            audioContextRef.current.close();
            audioContextRef.current = null;
        }

        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }

        setIsActive(false);
    }, []); // No dependencies - uses refs

    useEffect(() => {
        return () => {
            stopMicrophone();
        };
    }, []); // Empty dependency array - cleanup on unmount only

    return {
        isActive,
        stream: streamRef.current,
        startMicrophone,
        stopMicrophone
    };
}
