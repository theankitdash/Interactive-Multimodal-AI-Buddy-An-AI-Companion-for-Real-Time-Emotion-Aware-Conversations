import { useState } from 'react';
import { useApp } from '../context/AppContext';
import { useCamera } from '../hooks/useCamera';
import { BACKEND_URL, FACE_CAPTURE_COUNT, FACE_CAPTURE_DELAY } from '../config/constants';
import styles from './AuthScreen.module.css';

export function AuthScreen() {
    const { mode, setMode, setUser, showStatus } = useApp();
    const { videoRef, startCamera, stopCamera, captureFrame } = useCamera();

    const [username, setUsername] = useState('');
    const [fullname, setFullname] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [captureStatus, setCaptureStatus] = useState('');

    const handleRegister = async () => {
        if (!username.trim() || !fullname.trim()) {
            showStatus('Please enter both username and full name');
            return;
        }

        setIsLoading(true);
        try {
            // Start camera first
            setCaptureStatus('Starting camera...');
            await startCamera();

            // Capture face embeddings
            const embeddings = [];
            for (let i = 0; i < FACE_CAPTURE_COUNT; i++) {
                setCaptureStatus(`Capturing face sample ${i + 1}/${FACE_CAPTURE_COUNT}...`);
                await new Promise(resolve => setTimeout(resolve, FACE_CAPTURE_DELAY));
                const frameData = await captureFrame();
                if (!frameData) {
                    throw new Error('Failed to capture frame');
                }

                const response = await fetch(`${BACKEND_URL}/api/auth/capture-face`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_data: frameData })
                });

                const result = await response.json();
                if (result.success && result.embedding) {
                    embeddings.push(result.embedding);
                } else {
                    throw new Error(result.message || 'Face not detected');
                }
            }

            // Stop camera after capture
            stopCamera();
            setCaptureStatus('Processing registration...');

            // Register user with face embeddings in one call
            const registerResponse = await fetch(`${BACKEND_URL}/api/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    username,
                    fullname,
                    face_embeddings: embeddings
                })
            });

            const user = await registerResponse.json();
            setUser(user);
            setMode('assistant');
            showStatus(`Welcome, ${user.fullname}!`);
        } catch (error: any) {
            stopCamera(); // Ensure camera stops on error
            showStatus(error.message || 'Registration failed');
        } finally {
            setIsLoading(false);
            setCaptureStatus('');
        }
    };

    const handleLogin = async () => {
        setIsLoading(true);
        try {
            // Start camera first
            setCaptureStatus('Starting camera...');
            await startCamera();

            // Capture single face sample for login (faster authentication)
            setCaptureStatus('Capturing your face...');
            await new Promise(resolve => setTimeout(resolve, 500)); // Small delay for camera to stabilize

            const frameData = await captureFrame();
            if (!frameData) {
                throw new Error('Failed to capture frame');
            }

            const response = await fetch(`${BACKEND_URL}/api/auth/capture-face`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_data: frameData })
            });

            const result = await response.json();
            if (!result.success || !result.embedding) {
                throw new Error(result.message || 'Face not detected');
            }

            // Stop camera after capture
            stopCamera();
            setCaptureStatus('Verifying face...');

            // Login with single embedding
            const loginResponse = await fetch(`${BACKEND_URL}/api/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    face_embeddings: [result.embedding]
                })
            });

            if (!loginResponse.ok) {
                const error = await loginResponse.json();
                throw new Error(error.detail || 'Face not recognized');
            }

            const user = await loginResponse.json();
            setUser(user);
            setMode('assistant');
            showStatus(`Welcome back, ${user.fullname}!`);
        } catch (error: any) {
            stopCamera(); // Ensure camera stops on error
            showStatus(error.message || 'Login failed');
        } finally {
            setIsLoading(false);
            setCaptureStatus('');
        }
    };

    return (
        <div className={styles.container}>
            {/* Camera video - only visible during capture */}
            {isLoading && (
                <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className={styles.cameraBackground}
                />
            )}

            <div className={styles.authCard}>
                <h1 className={styles.title}>Deva</h1>
                <p className={styles.subtitle}>Your AI Companion</p>

                {/* Show capture status during loading */}
                {captureStatus && (
                    <p className={styles.captureStatus}>{captureStatus}</p>
                )}

                {mode === 'register' && (
                    <>
                        <input
                            type="text"
                            placeholder="Username"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            className={styles.input}
                            disabled={isLoading}
                        />
                        <input
                            type="text"
                            placeholder="Full Name"
                            value={fullname}
                            onChange={(e) => setFullname(e.target.value)}
                            className={styles.input}
                            disabled={isLoading}
                        />
                    </>
                )}

                <button
                    onClick={mode === 'login' ? handleLogin : handleRegister}
                    className={styles.actionButton}
                    disabled={isLoading}
                >
                    {isLoading ? 'Processing...' : mode === 'login' ? 'Login' : 'Register'}
                </button>

                <button
                    onClick={() => setMode(mode === 'login' ? 'register' : 'login')}
                    className={styles.switchButton}
                    disabled={isLoading}
                >
                    {mode === 'login' ? 'Switch to Register' : 'Switch to Login'}
                </button>
            </div>
        </div>
    );
}
