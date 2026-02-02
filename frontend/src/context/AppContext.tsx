import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { AppContextType, AppMode, AIState, User } from '../types';

const AppContext = createContext<AppContextType | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
    const [mode, setMode] = useState<AppMode>('login');
    const [user, setUser] = useState<User | null>(null);
    const [aiState, setAiState] = useState<AIState>('listening');
    const [isMuted, setIsMuted] = useState(false);
    const [isCameraOn, setIsCameraOn] = useState(false); // Camera off by default
    const [statusMessage, setStatusMessage] = useState('');

    const toggleMute = useCallback(() => {
        setIsMuted(prev => !prev);
    }, []);

    const toggleCamera = useCallback(() => {
        setIsCameraOn(prev => !prev);
    }, []);

    const showStatus = useCallback((message: string, duration: number = 5000) => {
        setStatusMessage(message);
        setTimeout(() => {
            setStatusMessage('');
        }, duration);
    }, []);

    const value: AppContextType = {
        mode,
        setMode,
        user,
        setUser,
        aiState,
        setAiState,
        isMuted,
        toggleMute,
        isCameraOn,
        toggleCamera,
        statusMessage,
        showStatus
    };

    return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
    const context = useContext(AppContext);
    if (context === undefined) {
        throw new Error('useApp must be used within an AppProvider');
    }
    return context;
}
