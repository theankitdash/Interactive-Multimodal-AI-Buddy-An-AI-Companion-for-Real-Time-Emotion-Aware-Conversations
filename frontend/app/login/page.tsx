'use client';

import { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function LoginPage() {
    const router = useRouter();
    const videoRef = useRef<HTMLVideoElement>(null);
    const [isRegistering, setIsRegistering] = useState(false);
    const [name, setName] = useState('');
    const [userId, setUserId] = useState('');
    const [isCameraReady, setIsCameraReady] = useState(false);
    const [verifying, setVerifying] = useState(false);

    useEffect(() => {
        startCamera();
        return () => {
            stopCamera();
        };
    }, []);

    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user' }
            });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                setIsCameraReady(true);
            }
        } catch (err) {
            console.error('Error accessing camera:', err);
        }
    };

    const stopCamera = () => {
        if (videoRef.current && videoRef.current.srcObject) {
            const stream = videoRef.current.srcObject as MediaStream;
            stream.getTracks().forEach(track => track.stop());
        }
    };

    const handleVerifyFace = async () => {
        setVerifying(true);

        // Simulate face verification API call
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Check if user exists (mock)
        const userExists = false; // Replace with actual API call

        if (userExists) {
            // User found, login
            localStorage.setItem('user', JSON.stringify({ name: 'User', userId: 'user123' }));
            router.push('/');
        } else {
            // User not found, show registration
            setIsRegistering(true);
            setVerifying(false);
        }
    };

    const handleRegister = async (e: React.FormEvent) => {
        e.preventDefault();
        setVerifying(true);

        // Simulate registration API call
        await new Promise(resolve => setTimeout(resolve, 1500));

        // Save user data
        localStorage.setItem('user', JSON.stringify({ name, userId }));
        router.push('/');
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-background p-4">
            <div className="w-full max-w-md fade-in">
                <div className="text-center mb-8">
                    <h1 className="text-5xl font-bold mb-2 gradient-text">Deva</h1>
                    <p className="text-foreground-secondary">Your AI Companion</p>
                </div>

                <div className="glass-card p-6 space-y-6">
                    {/* Camera Feed */}
                    <div className="relative aspect-video bg-background-secondary rounded-lg overflow-hidden">
                        <video
                            ref={videoRef}
                            autoPlay
                            playsInline
                            muted
                            className="w-full h-full object-cover"
                        />
                        {!isCameraReady && (
                            <div className="absolute inset-0 flex items-center justify-center">
                                <div className="text-foreground-secondary">Initializing camera...</div>
                            </div>
                        )}
                    </div>

                    {!isRegistering ? (
                        /* Face Verification */
                        <div className="space-y-4">
                            <p className="text-center text-foreground-secondary text-sm">
                                Position your face in the camera to verify your identity
                            </p>
                            <button
                                onClick={handleVerifyFace}
                                disabled={!isCameraReady || verifying}
                                className="w-full py-3 rounded-lg font-medium text-white bg-gradient-to-r from-accent-primary to-accent-secondary hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed glow"
                            >
                                {verifying ? 'Verifying...' : 'Verify Face'}
                            </button>
                        </div>
                    ) : (
                        /* Registration Form */
                        <form onSubmit={handleRegister} className="space-y-4">
                            <div className="text-center mb-4">
                                <p className="text-foreground-secondary text-sm">
                                    New user detected. Please register to continue.
                                </p>
                            </div>

                            <div>
                                <label htmlFor="name" className="block text-sm font-medium mb-2">
                                    Name
                                </label>
                                <input
                                    id="name"
                                    type="text"
                                    value={name}
                                    onChange={(e) => setName(e.target.value)}
                                    required
                                    className="w-full px-4 py-3 rounded-lg bg-background-secondary border border-glass-border focus:border-accent-primary focus:outline-none text-foreground"
                                    placeholder="Enter your name"
                                />
                            </div>

                            <div>
                                <label htmlFor="userId" className="block text-sm font-medium mb-2">
                                    User ID
                                </label>
                                <input
                                    id="userId"
                                    type="text"
                                    value={userId}
                                    onChange={(e) => setUserId(e.target.value)}
                                    required
                                    className="w-full px-4 py-3 rounded-lg bg-background-secondary border border-glass-border focus:border-accent-primary focus:outline-none text-foreground"
                                    placeholder="Choose a user ID"
                                />
                            </div>

                            <button
                                type="submit"
                                disabled={verifying}
                                className="w-full py-3 rounded-lg font-medium text-white bg-gradient-to-r from-accent-primary to-accent-secondary hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed glow"
                            >
                                {verifying ? 'Registering...' : 'Register & Login'}
                            </button>
                        </form>
                    )}
                </div>
            </div>
        </div>
    );
}
