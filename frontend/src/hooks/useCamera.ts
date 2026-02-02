import { useState, useRef, useCallback } from 'react';
import { VIDEO_WIDTH, VIDEO_HEIGHT, IMAGE_QUALITY } from '../config/constants';

export function useCamera() {
    const [isActive, setIsActive] = useState(false);
    const streamRef = useRef<MediaStream | null>(null);
    const videoRef = useRef<HTMLVideoElement>(null);

    const startCamera = useCallback(async () => {
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { width: VIDEO_WIDTH, height: VIDEO_HEIGHT },
                audio: false
            });

            streamRef.current = mediaStream;

            if (videoRef.current) {
                videoRef.current.srcObject = mediaStream;

                // Wait for video to be ready (metadata loaded)
                await new Promise<void>((resolve, reject) => {
                    const video = videoRef.current;
                    if (!video) {
                        reject(new Error('Video element not found'));
                        return;
                    }

                    const onLoadedMetadata = () => {
                        console.log('[Camera] Video metadata loaded:', {
                            width: video.videoWidth,
                            height: video.videoHeight
                        });
                        video.removeEventListener('loadedmetadata', onLoadedMetadata);
                        resolve();
                    };

                    const onError = (e: Event) => {
                        console.error('[Camera] Video error:', e);
                        video.removeEventListener('error', onError);
                        reject(new Error('Video failed to load'));
                    };

                    // If metadata is already loaded
                    if (video.readyState >= 2) {
                        console.log('[Camera] Video already ready');
                        resolve();
                    } else {
                        video.addEventListener('loadedmetadata', onLoadedMetadata);
                        video.addEventListener('error', onError);
                    }
                });
            }

            setIsActive(true);
            console.log('[Camera] Camera started successfully');
            return mediaStream;
        } catch (error) {
            console.error('Error accessing camera:', error);
            setIsActive(false);
            throw error;
        }
    }, []);

    const stopCamera = useCallback(() => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
            setIsActive(false);

            if (videoRef.current) {
                videoRef.current.srcObject = null;
            }
        }
    }, []); // No dependencies - uses refs to avoid memory leak

    const captureFrame = useCallback(async (): Promise<string | null> => {
        // Check if stream and video are ready (don't rely on isActive state due to async updates)
        if (!videoRef.current || !streamRef.current) {
            console.error('[Camera] Cannot capture: video/stream not ready', {
                hasVideo: !!videoRef.current,
                hasStream: !!streamRef.current,
                isActive
            });
            return null;
        }

        const video = videoRef.current;

        // Validate video dimensions are available
        if (!video.videoWidth || !video.videoHeight) {
            console.error('[Camera] Video dimensions not available:', {
                videoWidth: video.videoWidth,
                videoHeight: video.videoHeight,
                readyState: video.readyState
            });
            return null;
        }

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext('2d');
        if (!ctx) {
            console.error('[Camera] Failed to get canvas context');
            return null;
        }

        ctx.drawImage(video, 0, 0);
        const dataUrl = canvas.toDataURL('image/jpeg', IMAGE_QUALITY);

        console.log('[Camera] Frame captured successfully:', {
            size: `${canvas.width}x${canvas.height}`,
            dataLength: dataUrl.length
        });

        return dataUrl;
    }, [isActive]);

    return {
        isActive,
        videoRef,
        stream: streamRef.current,
        startCamera,
        stopCamera,
        captureFrame
    };
}
