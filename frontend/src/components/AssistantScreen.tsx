import { useEffect, useRef, useState } from 'react';
import { useApp } from '../context/AppContext';
import { useCamera } from '../hooks/useCamera';
import { useMicrophone } from '../hooks/useMicrophone';
import { useAudio } from '../hooks/useAudio';
import { BACKEND_WS_URL } from '../config/constants';
import styles from './AssistantScreen.module.css';

export function AssistantScreen() {
    const { user, setMode, aiState, setAiState, isMuted, toggleMute, isCameraOn, toggleCamera } = useApp();
    const { videoRef, startCamera, stopCamera, captureFrame } = useCamera();
    const { startMicrophone, stopMicrophone } = useMicrophone();
    const { playAudio, stop: stopAudio } = useAudio();

    // Dual Socket Architecture
    // Audio Socket - Gemini 
    const audioWsRef = useRef<WebSocket | null>(null);
    // Cognition Socket - Reasoning and memory (brain)
    const cognitionWsRef = useRef<WebSocket | null>(null);

    const frameIntervalRef = useRef<number | null>(null);
    const isCleanedUpRef = useRef<boolean>(false);
    const intentionalCloseRef = useRef<boolean>(false);
    const [cognitionStatus, setCognitionStatus] = useState<string>('disconnected');

    useEffect(() => {
        if (!user) return;

        // Reset cleanup flags
        isCleanedUpRef.current = false;
        intentionalCloseRef.current = false;

        // Check if sockets are already connected (StrictMode guard)
        if (audioWsRef.current || cognitionWsRef.current) {
            const audioState = audioWsRef.current?.readyState;
            const cognitionState = cognitionWsRef.current?.readyState;
            if (audioState === WebSocket.OPEN || cognitionState === WebSocket.OPEN) {
                console.log('[Sockets] Already connected, skipping duplicate');
                return;
            }
        }

        // ======== AUDIO SOCKET (Gemini Voice I/O) ========
        console.log('[Audio Socket] Connecting to Gemini voice...');
        const audioWs = new WebSocket(`${BACKEND_WS_URL}/api/assistant/stream`);
        audioWsRef.current = audioWs;

        audioWs.onopen = () => {
            if (isCleanedUpRef.current) {
                intentionalCloseRef.current = true;
                audioWs.close();
                return;
            }

            console.log('[Audio Socket] Connected to Gemini');
            audioWs.send(JSON.stringify({ username: user.username }));

            // Start microphone for audio input
            if (!isMuted) {
                startMicrophone((audioData) => {
                    if (audioWs.readyState === WebSocket.OPEN && !isMuted && !isCleanedUpRef.current) {
                        const base64 = btoa(String.fromCharCode(...new Uint8Array(audioData.buffer)));
                        audioWs.send(JSON.stringify({ type: 'audio', data: base64 }));
                    }
                }).catch(console.error);
            }

            // Start video frame sending (1 fps)
            frameIntervalRef.current = window.setInterval(async () => {
                if (isCameraOn && !isCleanedUpRef.current) {
                    const frameData = await captureFrame();
                    if (frameData && audioWs.readyState === WebSocket.OPEN) {
                        const base64Data = frameData.split(',')[1];
                        audioWs.send(JSON.stringify({ type: 'video', data: base64Data }));
                    }
                }
            }, 1000);
        };

        audioWs.onmessage = (event) => {
            if (isCleanedUpRef.current) return;

            try {
                const message = JSON.parse(event.data);

                // Audio Socket only handles audio playback
                if (message.type === 'audio_reply') {
                    playAudio(message.data, message.sample_rate || 24000);
                    setAiState('speaking');
                    setTimeout(() => setAiState('listening'), 1000);
                }
            } catch (error) {
                console.error('[Audio Socket] Message error:', error);
            }
        };

        audioWs.onerror = (error) => {
            if (!intentionalCloseRef.current) {
                console.error('[Audio Socket] Error:', error);
            }
        };

        audioWs.onclose = () => {
            if (!intentionalCloseRef.current) {
                console.log('[Audio Socket] Disconnected');
            }
        };

        // ======== COGNITION SOCKET (Reasoning) ========
        console.log('[Cognition Socket] Connecting to Brain...');
        const cognitionWs = new WebSocket(`${BACKEND_WS_URL}/api/cognition/stream`);
        cognitionWsRef.current = cognitionWs;

        cognitionWs.onopen = () => {
            if (isCleanedUpRef.current) {
                intentionalCloseRef.current = true;
                cognitionWs.close();
                return;
            }

            console.log('[Cognition Socket] Connected to Brain');
            cognitionWs.send(JSON.stringify({ username: user.username }));
            setCognitionStatus('connected');
        };

        cognitionWs.onmessage = (event) => {
            if (isCleanedUpRef.current) return;

            try {
                const message = JSON.parse(event.data);

                if (message.event === 'reasoning_complete') {
                    // Reasoning happened, update UI state
                    console.log('[Cognition] Reasoning:', message.context);
                    setAiState('thinking');
                    setTimeout(() => setAiState('listening'), 800);
                }

                else if (message.event === 'memory_stored') {
                    console.log('[Cognition] Memory stored:', message.content);
                }

                else if (message.event === 'state_update') {
                    if (message.state) {
                        setAiState(message.state);
                    }
                }
            } catch (error) {
                console.error('[Cognition Socket] Message error:', error);
            }
        };

        cognitionWs.onerror = (error) => {
            if (!intentionalCloseRef.current) {
                console.error('[Cognition Socket] Error:', error);
            }
            setCognitionStatus('error');
        };

        cognitionWs.onclose = () => {
            if (!intentionalCloseRef.current) {
                console.log('[Cognition Socket] Disconnected');
            }
            setCognitionStatus('disconnected');
        };

        // Cleanup function
        return () => {
            isCleanedUpRef.current = true;
            intentionalCloseRef.current = true;

            // Clear frame interval
            if (frameIntervalRef.current) {
                clearInterval(frameIntervalRef.current);
                frameIntervalRef.current = null;
            }

            // Close Audio Socket
            if (audioWs.readyState === WebSocket.OPEN || audioWs.readyState === WebSocket.CONNECTING) {
                try {
                    if (audioWs.readyState === WebSocket.OPEN) {
                        audioWs.send(JSON.stringify({ type: 'close' }));
                    }
                    audioWs.close();
                } catch (e) {
                    // Suppress
                }
            }

            // Close Cognition Socket
            if (cognitionWs.readyState === WebSocket.OPEN || cognitionWs.readyState === WebSocket.CONNECTING) {
                try {
                    if (cognitionWs.readyState === WebSocket.OPEN) {
                        cognitionWs.send(JSON.stringify({ event: 'close' }));
                    }
                    cognitionWs.close();
                } catch (e) {
                    // Suppress
                }
            }

            audioWsRef.current = null;
            cognitionWsRef.current = null;
            stopMicrophone();
            stopCamera();
            stopAudio();
        };
    }, [user]);

    // Handle camera toggle
    useEffect(() => {
        if (isCameraOn) {
            startCamera().catch(console.error);
        } else {
            stopCamera();
        }
    }, [isCameraOn]);

    // Handle mute toggle
    useEffect(() => {
        if (isMuted) {
            stopMicrophone();
        } else if (audioWsRef.current?.readyState === WebSocket.OPEN) {
            startMicrophone((audioData) => {
                if (audioWsRef.current?.readyState === WebSocket.OPEN && !isMuted) {
                    const base64 = btoa(String.fromCharCode(...new Uint8Array(audioData.buffer)));
                    audioWsRef.current.send(JSON.stringify({ type: 'audio', data: base64 }));
                }
            }).catch(console.error);
        }
    }, [isMuted]);

    const handleLogout = () => {
        if (audioWsRef.current) {
            audioWsRef.current.send(JSON.stringify({ type: 'close' }));
            audioWsRef.current.close();
        }
        if (cognitionWsRef.current) {
            cognitionWsRef.current.send(JSON.stringify({ event: 'close' }));
            cognitionWsRef.current.close();
        }
        setMode('login');
    };

    const getStateColor = () => {
        const colors = {
            listening: '#6366f1',
            thinking: '#f59e0b',
            speaking: '#10b981'
        };
        return colors[aiState];
    };

    return (
        <div className={styles.container}>
            {/* Profile badge (top-left) */}
            <div className={styles.profileBadge}>
                {user?.initials}
            </div>

            {/* Logout button (bottom-left) */}
            <button className={styles.logoutButton} onClick={handleLogout}>
                Logout
            </button>

            {/* AI State Indicator (center) */}
            <div className={styles.aiIndicator}>
                <div
                    className={styles.aiGlow}
                    style={{ backgroundColor: getStateColor(), opacity: 0.2 }}
                />
                <div
                    className={styles.aiCircle}
                    style={{ backgroundColor: getStateColor() }}
                />
                <p className={styles.aiStateLabel}>{aiState.charAt(0).toUpperCase() + aiState.slice(1)}</p>

                {/* Debug: Show cognition status */}
                {cognitionStatus !== 'connected' && (
                    <p style={{ fontSize: '10px', color: '#ff6b6b', marginTop: '4px' }}>
                        Cognition: {cognitionStatus}
                    </p>
                )}

                {/* Media controls */}
                <div className={styles.mediaControls}>
                    <button
                        className={`${styles.controlButton} ${isMuted ? styles.active : ''}`}
                        onClick={toggleMute}
                    >
                        {isMuted ? '🔇' : '🔊'}
                    </button>
                    <button
                        className={`${styles.controlButton} ${!isCameraOn ? styles.active : ''}`}
                        onClick={toggleCamera}
                    >
                        {isCameraOn ? '📹' : '🚫'}
                    </button>
                </div>
            </div>

            {/* Camera preview (bottom-right) */}
            {isCameraOn && (
                <div className={styles.cameraPreview}>
                    <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className={styles.previewVideo}
                    />
                </div>
            )}
        </div>
    );
}
