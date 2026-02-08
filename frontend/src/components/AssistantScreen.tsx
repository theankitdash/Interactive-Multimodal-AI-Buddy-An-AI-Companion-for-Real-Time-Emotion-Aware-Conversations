import { useEffect, useRef } from 'react';
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

    const wsRef = useRef<WebSocket | null>(null);
    const frameIntervalRef = useRef<number | null>(null);
    const isCleanedUpRef = useRef<boolean>(false);  // Guard against StrictMode double-mount
    const intentionalCloseRef = useRef<boolean>(false);  // Track intentional closes to suppress errors

    useEffect(() => {
        if (!user) return;

        // Reset cleanup flag on mount
        isCleanedUpRef.current = false;
        intentionalCloseRef.current = false;

        // If there's already an active or connecting WebSocket, don't create a new one
        // This handles StrictMode's double-mount behavior
        if (wsRef.current) {
            const state = wsRef.current.readyState;
            if (state === WebSocket.OPEN || state === WebSocket.CONNECTING) {
                console.log('[WebSocket] Already connected/connecting, skipping duplicate connection');
                return;
            }
            // If CLOSING, clear the ref and let it reconnect naturally on next render
            if (state === WebSocket.CLOSING) {
                console.log('[WebSocket] Previous connection closing, clearing ref');
                wsRef.current = null;
                return;
            }
        }

        // Connect WebSocket
        console.log('[WebSocket] Creating new connection...');
        const ws = new WebSocket(`${BACKEND_WS_URL}/api/assistant/stream`);
        wsRef.current = ws;

        ws.onopen = () => {
            // Check if we've been cleaned up while connecting
            if (isCleanedUpRef.current) {
                console.log('[WebSocket] Cleaned up during connection, closing');
                intentionalCloseRef.current = true;
                ws.close();
                return;
            }

            console.log('[WebSocket] Connected');
            // Send initialization
            ws.send(JSON.stringify({ username: user.username }));

            // Start microphone
            if (!isMuted) {
                startMicrophone((audioData) => {
                    if (ws.readyState === WebSocket.OPEN && !isMuted && !isCleanedUpRef.current) {
                        const base64 = btoa(String.fromCharCode(...new Uint8Array(audioData.buffer)));
                        ws.send(JSON.stringify({ type: 'audio', data: base64 }));
                    }
                }).catch(console.error);
            }

            // Start sending video frames (1 frame/second) - only if camera is on
            frameIntervalRef.current = window.setInterval(async () => {
                if (isCameraOn && !isCleanedUpRef.current) {
                    const frameData = await captureFrame();
                    if (frameData && ws.readyState === WebSocket.OPEN) {
                        const base64Data = frameData.split(',')[1];
                        ws.send(JSON.stringify({ type: 'video', data: base64Data }));
                    }
                }
            }, 1000);
        };

        ws.onmessage = (event) => {
            // Ignore messages if cleaned up
            if (isCleanedUpRef.current) return;

            try {
                const message = JSON.parse(event.data);

                if (message.type === 'audio_reply') {
                    playAudio(message.data, message.sample_rate || 24000);
                    setAiState('speaking');
                    setTimeout(() => setAiState('listening'), 1000);
                }
            } catch (error) {
                console.error('[WebSocket] Message error:', error);
            }
        };

        ws.onerror = (error) => {
            // Only log errors if not intentionally closing (e.g., during StrictMode cleanup)
            if (!intentionalCloseRef.current && !isCleanedUpRef.current) {
                console.error('[WebSocket] Connection error:', error);
            }
            setAiState('listening'); // Reset state on error
        };

        ws.onclose = () => {
            // Only log if not intentionally closed
            if (!intentionalCloseRef.current) {
                console.log('[WebSocket] Disconnected');
            }
        };

        return () => {
            // Mark as cleaned up to prevent any pending callbacks from executing
            isCleanedUpRef.current = true;
            intentionalCloseRef.current = true;  // Suppress expected errors during cleanup

            // Cleanup
            if (frameIntervalRef.current) {
                clearInterval(frameIntervalRef.current);
                frameIntervalRef.current = null;
            }

            // Send close message and close WebSocket safely
            if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
                try {
                    // Only send close message if connection is fully open
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ type: 'close' }));
                    }
                } catch (e) {
                    // Suppress - expected during cleanup
                } finally {
                    // Always attempt to close the WebSocket
                    try {
                        ws.close();
                    } catch (e) {
                        // Suppress - expected during cleanup
                    }
                }
            }

            wsRef.current = null;
            stopMicrophone();
            stopCamera();
            stopAudio();
        };
    }, [user]);

    // Handle camera toggle - start/stop camera when toggled
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
        } else if (wsRef.current?.readyState === WebSocket.OPEN) {
            startMicrophone((audioData) => {
                if (wsRef.current?.readyState === WebSocket.OPEN && !isMuted) {
                    const base64 = btoa(String.fromCharCode(...new Uint8Array(audioData.buffer)));
                    wsRef.current.send(JSON.stringify({ type: 'audio', data: base64 }));
                }
            }).catch(console.error);
        }
    }, [isMuted]);

    const handleLogout = () => {
        if (wsRef.current) {
            wsRef.current.send(JSON.stringify({ type: 'close' }));
            wsRef.current.close();
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
