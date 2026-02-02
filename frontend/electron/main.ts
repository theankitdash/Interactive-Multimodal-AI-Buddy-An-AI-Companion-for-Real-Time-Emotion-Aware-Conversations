import { app, BrowserWindow, ipcMain } from 'electron';
import { spawn, ChildProcess } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let mainWindow: BrowserWindow | null = null;
let pythonBackend: ChildProcess | null = null;

// Auto-launch on startup setting
function setAutoLaunch(enable: boolean) {
    app.setLoginItemSettings({
        openAtLogin: enable,
        openAsHidden: false,
        path: process.execPath
    });
}

// Start Python FastAPI backend
function startBackend() {
    const isDev = process.env.NODE_ENV === 'development';
    // In dev: from frontend/electron to project root backend
    // In prod: backend bundled in resources
    const backendPath = isDev
        ? path.join(__dirname, '..', '..', 'backend')  // Go up two levels from frontend/electron to project root, then into backend
        : path.join(process.resourcesPath, 'backend');

    console.log('[Electron] Starting Python backend from:', backendPath);

    // Use python without shell for better security
    pythonBackend = spawn('python', ['-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '8000'], {
        cwd: backendPath
    });

    pythonBackend.stdout?.on('data', (data) => {
        console.log(`[Backend] ${data}`);
    });

    pythonBackend.stderr?.on('data', (data) => {
        console.error(`[Backend Error] ${data}`);
    });

    pythonBackend.on('close', (code) => {
        console.log(`[Backend] Process exited with code ${code}`);
    });
}

// Stop Python backend
function stopBackend() {
    if (pythonBackend) {
        console.log('[Electron] Stopping Python backend...');
        pythonBackend.kill();
        pythonBackend = null;
    }
}

// Create the main window
function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1280,
        height: 800,
        minWidth: 800,
        minHeight: 600,
        title: 'AI Buddy',
        backgroundColor: '#0a0a0a',
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            nodeIntegration: false,
            contextIsolation: true,
            webSecurity: true
        },
        autoHideMenuBar: true,
        icon: path.join(__dirname, '..', 'public', 'icon.png')
    });

    // Load the app
    if (process.env.NODE_ENV === 'development') {
        mainWindow.loadURL('http://localhost:5173');
        mainWindow.webContents.openDevTools();
    } else {
        mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
    }

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

// Backend health check
async function waitForBackend(maxAttempts: number = 30): Promise<boolean> {
    for (let i = 0; i < maxAttempts; i++) {
        try {
            const response = await fetch('http://127.0.0.1:8000/health', {
                method: 'GET',
                signal: AbortSignal.timeout(3000)
            });

            if (response.ok) {
                const data = await response.json();
                if (data.status === 'healthy' || data.status === 'ok') {
                    console.log('[Electron] Backend is ready');
                    return true;
                }
            }
        } catch (error) {
            // Backend not ready yet
        }

        // Wait 1 second before next attempt
        await new Promise(resolve => setTimeout(resolve, 1000));
    }

    console.error('[Electron] Backend health check timeout');
    return false;
}

// App lifecycle
app.whenReady().then(async () => {
    const isDev = process.env.NODE_ENV === 'development';

    // Only start backend in production mode
    // In dev mode, backend is already running via npm run dev
    if (!isDev) {
        startBackend();

        // Wait for backend to be ready with proper health check
        const backendReady = await waitForBackend();

        if (!backendReady) {
            // Show error dialog if backend fails to start
            const { dialog } = require('electron');
            dialog.showErrorBox(
                'Backend Error',
                'Failed to start the backend server. Please check if Python and all dependencies are installed.'
            );
            app.quit();
            return;
        }
    } else {
        // In dev mode, just verify backend is already running
        console.log('[Electron] Development mode - expecting backend to be running on port 8000');
        const backendReady = await waitForBackend();

        if (!backendReady) {
            const { dialog } = require('electron');
            dialog.showErrorBox(
                'Backend Not Running',
                'Backend is not running. Please start it with: npm run dev'
            );
            app.quit();
            return;
        }
    }

    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        // Only stop backend if we started it (production mode)
        if (process.env.NODE_ENV !== 'development') {
            stopBackend();
        }
        app.quit();
    }
});

app.on('before-quit', () => {
    // Only stop backend if we started it (production mode)
    if (process.env.NODE_ENV !== 'development') {
        stopBackend();
    }
});

// IPC Handlers
ipcMain.handle('get-backend-url', () => {
    return 'http://127.0.0.1:8000';
});

ipcMain.handle('set-auto-launch', (_event, enable: boolean) => {
    setAutoLaunch(enable);
    return app.getLoginItemSettings().openAtLogin;
});

ipcMain.handle('get-auto-launch', () => {
    return app.getLoginItemSettings().openAtLogin;
});
