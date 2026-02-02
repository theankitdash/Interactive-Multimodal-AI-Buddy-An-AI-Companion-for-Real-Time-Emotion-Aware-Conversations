export interface User {
    username: string;
    fullname: string;
    initials: string;
}

export interface FaceCaptureResponse {
    success: boolean;
    message: string;
    embedding?: number[];
}

export type AIState = 'listening' | 'thinking' | 'speaking';

export type AppMode = 'login' | 'register' | 'assistant';

export interface AppContextType {
    mode: AppMode;
    setMode: (mode: AppMode) => void;

    user: User | null;
    setUser: (user: User | null) => void;

    aiState: AIState;
    setAiState: (state: AIState) => void;

    isMuted: boolean;
    toggleMute: () => void;

    isCameraOn: boolean;
    toggleCamera: () => void;

    statusMessage: string;
    showStatus: (message: string, duration?: number) => void;
}

export interface StreamMessage {
    type: 'audio' | 'video' | 'text' | 'audio_reply' | 'close';
    data: string;
    sample_rate?: number;
}
