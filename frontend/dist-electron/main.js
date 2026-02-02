"use strict";
const electron = require("electron");
const child_process = require("child_process");
const path = require("path");
const url = require("url");
var _documentCurrentScript = typeof document !== "undefined" ? document.currentScript : null;
const __filename$1 = url.fileURLToPath(typeof document === "undefined" ? require("url").pathToFileURL(__filename).href : _documentCurrentScript && _documentCurrentScript.tagName.toUpperCase() === "SCRIPT" && _documentCurrentScript.src || new URL("main.js", document.baseURI).href);
const __dirname$1 = path.dirname(__filename$1);
let mainWindow = null;
let pythonBackend = null;
function setAutoLaunch(enable) {
  electron.app.setLoginItemSettings({
    openAtLogin: enable,
    openAsHidden: false,
    path: process.execPath
  });
}
function startBackend() {
  const isDev = process.env.NODE_ENV === "development";
  const backendPath = isDev ? path.join(__dirname$1, "..", "..", "backend") : path.join(process.resourcesPath, "backend");
  console.log("[Electron] Starting Python backend from:", backendPath);
  pythonBackend = child_process.spawn("python", ["-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"], {
    cwd: backendPath
  });
  pythonBackend.stdout?.on("data", (data) => {
    console.log(`[Backend] ${data}`);
  });
  pythonBackend.stderr?.on("data", (data) => {
    console.error(`[Backend Error] ${data}`);
  });
  pythonBackend.on("close", (code) => {
    console.log(`[Backend] Process exited with code ${code}`);
  });
}
function stopBackend() {
  if (pythonBackend) {
    console.log("[Electron] Stopping Python backend...");
    pythonBackend.kill();
    pythonBackend = null;
  }
}
function createWindow() {
  mainWindow = new electron.BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    title: "AI Buddy",
    backgroundColor: "#0a0a0a",
    webPreferences: {
      preload: path.join(__dirname$1, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: true
    },
    autoHideMenuBar: true,
    icon: path.join(__dirname$1, "..", "public", "icon.png")
  });
  if (process.env.NODE_ENV === "development") {
    mainWindow.loadURL("http://localhost:5173");
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname$1, "..", "dist", "index.html"));
  }
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}
async function waitForBackend(maxAttempts = 30) {
  for (let i = 0; i < maxAttempts; i++) {
    try {
      const response = await fetch("http://127.0.0.1:8000/health", {
        method: "GET",
        signal: AbortSignal.timeout(3e3)
      });
      if (response.ok) {
        const data = await response.json();
        if (data.status === "healthy" || data.status === "ok") {
          console.log("[Electron] Backend is ready");
          return true;
        }
      }
    } catch (error) {
    }
    await new Promise((resolve) => setTimeout(resolve, 1e3));
  }
  console.error("[Electron] Backend health check timeout");
  return false;
}
electron.app.whenReady().then(async () => {
  const isDev = process.env.NODE_ENV === "development";
  if (!isDev) {
    startBackend();
    const backendReady = await waitForBackend();
    if (!backendReady) {
      const { dialog } = require("electron");
      dialog.showErrorBox(
        "Backend Error",
        "Failed to start the backend server. Please check if Python and all dependencies are installed."
      );
      electron.app.quit();
      return;
    }
  } else {
    console.log("[Electron] Development mode - expecting backend to be running on port 8000");
    const backendReady = await waitForBackend();
    if (!backendReady) {
      const { dialog } = require("electron");
      dialog.showErrorBox(
        "Backend Not Running",
        "Backend is not running. Please start it with: npm run dev"
      );
      electron.app.quit();
      return;
    }
  }
  createWindow();
  electron.app.on("activate", () => {
    if (electron.BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});
electron.app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    if (process.env.NODE_ENV !== "development") {
      stopBackend();
    }
    electron.app.quit();
  }
});
electron.app.on("before-quit", () => {
  if (process.env.NODE_ENV !== "development") {
    stopBackend();
  }
});
electron.ipcMain.handle("get-backend-url", () => {
  return "http://127.0.0.1:8000";
});
electron.ipcMain.handle("set-auto-launch", (_event, enable) => {
  setAutoLaunch(enable);
  return electron.app.getLoginItemSettings().openAtLogin;
});
electron.ipcMain.handle("get-auto-launch", () => {
  return electron.app.getLoginItemSettings().openAtLogin;
});
