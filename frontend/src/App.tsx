import React from 'react';
import { useApp } from './context/AppContext';
import { AuthScreen } from './components/AuthScreen';
import { AssistantScreen } from './components/AssistantScreen';
import './index.css';

export function App() {
    const { mode, statusMessage } = useApp();

    return (
        <>
            {mode === 'assistant' ? <AssistantScreen /> : <AuthScreen />}

            {/* Status Message Toast */}
            {statusMessage && (
                <div style={{
                    position: 'fixed',
                    bottom: '20px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    padding: '12px 24px',
                    background: 'var(--bg-glass)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 'var(--radius-md)',
                    color: 'var(--text-primary)',
                    fontSize: '14px',
                    zIndex: 'var(--z-modal)',
                    backdropFilter: 'blur(10px)',
                    WebkitBackdropFilter: 'blur(10px)',
                    animation: 'fadeIn 0.3s ease'
                }}>
                    {statusMessage}
                </div>
            )}
        </>
    );
}
