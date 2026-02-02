"use strict";
const electron = require("electron");
electron.contextBridge.exposeInMainWorld("electron", {
  // Get backend URL
  getBackendUrl: () => electron.ipcRenderer.invoke("get-backend-url"),
  // Auto-launch settings
  setAutoLaunch: (enable) => electron.ipcRenderer.invoke("set-auto-launch", enable),
  getAutoLaunch: () => electron.ipcRenderer.invoke("get-auto-launch")
});
