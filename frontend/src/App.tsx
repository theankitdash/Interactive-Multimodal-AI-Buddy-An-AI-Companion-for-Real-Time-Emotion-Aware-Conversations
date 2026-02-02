import { useApp } from './context/AppContext';
import { AuthScreen } from './components/AuthScreen';
import { AssistantScreen } from './components/AssistantScreen';
import './index.css';
import AnimatedBackground from './components/AnimatedBackground';

export function App() {
    const { mode, statusMessage } = useApp();

    return (
        <>
            <AnimatedBackground />
            {mode === 'assistant' ? <AssistantScreen /> : <AuthScreen />}

            {/* Status Message Toast */}
            {statusMessage && (
                <div className="status-toast glass-card">
                    {statusMessage}
                </div>
            )}
        </>
    );
}
