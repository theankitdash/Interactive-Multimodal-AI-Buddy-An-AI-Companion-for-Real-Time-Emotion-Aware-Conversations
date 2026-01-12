'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';

type AIState = 'listening' | 'thinking' | 'speaking';

export default function Home() {
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [user, setUser] = useState<{ name: string; userId: string } | null>(null);
  const [aiState, setAIState] = useState<AIState>('listening');
  const [isMuted, setIsMuted] = useState(false);
  const [isCameraOn, setIsCameraOn] = useState(true);

  useEffect(() => {
    // Check if user is logged in
    const userData = localStorage.getItem('user');
    if (!userData) {
      router.push('/login');
      return;
    }
    setUser(JSON.parse(userData));

    // Start camera
    if (isCameraOn) {
      startCamera();
    }

    return () => {
      stopCamera();
    };
  }, [router]);

  useEffect(() => {
    if (isCameraOn) {
      startCamera();
    } else {
      stopCamera();
    }
  }, [isCameraOn]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error('Error accessing camera:', err);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  const handleLogout = () => {
    stopCamera();
    localStorage.removeItem('user');
    router.push('/login');
  };

  const getInitials = (name: string) => {
    return name
      .split(' ')
      .map(n => n[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  const getStateColor = (state: AIState) => {
    switch (state) {
      case 'listening':
        return 'bg-accent-primary';
      case 'thinking':
        return 'bg-warning';
      case 'speaking':
        return 'bg-success';
    }
  };

  const cycleState = () => {
    const states: AIState[] = ['listening', 'thinking', 'speaking'];
    const currentIndex = states.indexOf(aiState);
    setAIState(states[(currentIndex + 1) % states.length]);
  };

  if (!user) {
    return null; // Will redirect to login
  }

  return (
    <div className="min-h-screen bg-background flex flex-col relative overflow-hidden">
      {/* Top Left - Profile Circle */}
      <div className="absolute top-6 left-6 z-10 fade-in">
        <div className="flex items-center gap-3">
          <div
            className="w-14 h-14 rounded-full flex items-center justify-center text-white font-bold text-lg"
            style={{ background: 'var(--accent-gradient)' }}
          >
            {getInitials(user.name)}
          </div>
        </div>
      </div>

      {/* Bottom Left - Logout Button */}
      <div className="absolute bottom-6 left-6 z-10 fade-in">
        <button
          onClick={handleLogout}
          className="glass-card px-6 py-3 text-foreground hover:bg-glass-border transition-all"
        >
          Logout
        </button>
      </div>

      {/* Center - AI Agent */}
      <div className="flex-1 flex flex-col items-center justify-center px-4 fade-in">
        <div className="text-center space-y-8">
          {/* Deva Logo/Name */}
          <h1 className="text-7xl md:text-8xl font-bold gradient-text mb-4">
            Deva
          </h1>

          {/* AI State Indicator */}
          <div className="flex flex-col items-center gap-4">
            <div
              className="relative w-32 h-32 cursor-pointer"
              onClick={cycleState}
            >
              {/* Outer glow ring */}
              <div className={`absolute inset-0 rounded-full ${getStateColor(aiState)} opacity-20 blur-2xl pulse`}></div>

              {/* Main circle */}
              <div className={`absolute inset-0 rounded-full ${getStateColor(aiState)} opacity-80 pulse`}></div>

              {/* Inner circle */}
              <div className={`absolute inset-4 rounded-full ${getStateColor(aiState)}`}></div>
            </div>

            <div className="text-xl font-medium text-foreground capitalize">
              {aiState}
            </div>
          </div>

          {/* Media Controls */}
          <div className="flex items-center gap-4 justify-center pt-8">
            {/* Audio Toggle */}
            <button
              onClick={() => setIsMuted(!isMuted)}
              className={`glass-card w-16 h-16 rounded-full flex items-center justify-center hover:scale-110 transition-transform ${isMuted ? 'bg-error' : ''
                }`}
              title={isMuted ? 'Unmute' : 'Mute'}
            >
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                {isMuted ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                )}
              </svg>
            </button>

            {/* Camera Toggle */}
            <button
              onClick={() => setIsCameraOn(!isCameraOn)}
              className={`glass-card w-16 h-16 rounded-full flex items-center justify-center hover:scale-110 transition-transform ${!isCameraOn ? 'bg-error' : ''
                }`}
              title={isCameraOn ? 'Turn off camera' : 'Turn on camera'}
            >
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                {isCameraOn ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
                )}
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Bottom Right - User Video Feed */}
      <div className="absolute bottom-6 right-6 z-10 fade-in">
        <div className="glass-card p-2 w-64 h-48 rounded-lg overflow-hidden">
          {isCameraOn ? (
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover rounded-lg"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-background-secondary rounded-lg">
              <svg
                className="w-12 h-12 text-foreground-secondary"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
              </svg>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
