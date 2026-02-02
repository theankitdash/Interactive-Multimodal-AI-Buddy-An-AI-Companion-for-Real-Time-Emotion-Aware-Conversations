import { contextBridge, ipcRenderer } from 'electron';

// Expose protected methods that allow the renderer process to use
// ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electron', {
    // Get backend URL
    getBackendUrl: () => ipcRenderer.invoke('get-backend-url'),

    // Auto-launch settings
    setAutoLaunch: (enable: boolean) => ipcRenderer.invoke('set-auto-launch', enable),
    getAutoLaunch: () => ipcRenderer.invoke('get-auto-launch'),
});

// Type definitions for TypeScript
export interface ElectronAPI {
    getBackendUrl: () => Promise<string>;
    setAutoLaunch: (enable: boolean) => Promise<boolean>;
    getAutoLaunch: () => Promise<boolean>;
}

declare global {
    interface Window {
        electron: ElectronAPI;
    }
}
