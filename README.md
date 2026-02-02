# Interactive Multimodal AI Buddy

**An AI Companion for Real-Time Emotion-Aware Conversations**

A **desktop application** built with **Electron + React** frontend and **Python FastAPI** backend, providing an intelligent, multimodal AI companion with facial recognition, voice interaction, and personalized memory.

---

## 🚀 Overview

The **Interactive Multimodal AI Buddy** is an advanced AI companion designed to engage users in real-time, emotionally intelligent conversations. By integrating multimodal inputs—such as voice and facial expressions—this system adapts its responses based on the user's emotional state, fostering more natural and empathetic interactions.

This is a **desktop application** built with modern web technologies (Electron + React) for the UI and Python for AI processing.

---

## 🧠 Features

- **Facial Recognition Authentication**: Secure login using face embeddings with FaceNet
- **Emotion Recognition**: Utilizes facial expression analysis and voice tone detection to gauge user emotions
- **Multimodal Interaction**: Supports voice and video inputs for a comprehensive conversational experience
- **Contextual Awareness**: Remembers past interactions using LangChain and vector databases
- **Real-Time Processing**: Immediate feedback during conversations with Gemini 2.0 Flash
- **Modern Desktop UI**: Electron-based application with glassmorphism design and smooth animations
- **Auto-Launch Support**: Optionally start with your computer for instant AI companion access

---

## 🛠️ Technology Stack

### Frontend (Electron + React)
- **Desktop Framework**: Electron
- **UI Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **Styling**: CSS Modules with glassmorphism effects
- **Media**: WebRTC for camera/microphone access
- **Real-time**: WebSocket for AI streaming

### Backend (Python FastAPI)
- **API Framework**: FastAPI
- **Facial Detection**: OpenCV, FaceNet-PyTorch (MTCNN)
- **LLM Agent**: Gemini 2.0 Flash (Multimodal)
- **LLM Framework**: LangChain (For Personalization)
- **Database**: PostgreSQL (User data), ChromaDB (Vector DB for memory)
- **Audio**: sounddevice (Microphone and speaker I/O)

---

## 🔧 Installation

### Prerequisites
- **Node.js** 18+ and **npm**
- **Python** 3.8+
- **PostgreSQL** databas
- Webcam and microphone
- Windows/Linux/macOS

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/theankitdash/Interactive-Multimodal-AI-Buddy.git
   cd Interactive-Multimodal-AI-Buddy
   ```

2. **Install PostgreSQL**:
   - Download from [postgresql.org/download](https://www.postgresql.org/download/)
   - Install with default settings (port 5432)
   - Create database:
     ```sql
     CREATE DATABASE multimodal_buddy;
     ```

3. **Backend Setup**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On Linux/macOS
   pip install -r backend/requirements.txt
   ```

4. **Frontend Setup**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

5. **Configure environment variables**:
   
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   NVIDIA_API_KEY=your_nvidia_api_key_here
   DB_USER=postgres
   DB_PASSWORD=your_postgres_password
   DB_NAME=multimodal_buddy
   DB_HOST=localhost
   DB_PORT=5432
   ```

6. **Run the application**:

   **Development mode**:
   ```bash
   cd frontend
   npm run dev  # Starts both backend (FastAPI) and frontend (Vite)
   ```
   
   **Production build**:
   ```bash
   cd frontend
   npm run build:electron  # Creates distributable desktop app
   ```

---

## 📖 Usage

1. **First Time**: Use the **Register** mode to create an account with facial recognition
2. **Login**: The AI will recognize your face for secure authentication
3. **Interact**: Speak naturally - the AI responds with voice and maintains conversation context
4. **Controls**: 
   - Toggle microphone mute
   - Toggle camera on/off
   - Logout to switch users
5. **Auto-Launch**: Enable in settings to start the app automatically with your computer

---

## 📁 Project Structure

```
Interactive-Multimodal-AI-Buddy/
├── backend/                 # Python FastAPI backend
│   ├── ai/                 # AI handlers (Gemini, LangChain)
│   ├── routes/             # API routes (auth, assistant, media)
│   ├── utils/              # Utilities (face recognition, database)
│   ├── main.py             # FastAPI app entry point
│   ├── models.py           # Pydantic models
│   ├── config.py           # Configuration
│   └── requirements.txt    # Python dependencies
├── frontend/               # Electron + React frontend
│   ├── src/               # React frontend source
│   │   ├── components/    # React components
│   │   ├── context/       # State management
│   │   ├── hooks/         # Custom hooks (camera, mic, audio)
│   │   ├── types/         # TypeScript types
│   │   └── main.tsx       # React entry point
│   ├── electron/          # Electron main process
│   │   ├── main.ts        # Main process (window, backend spawn)
│   │   └── preload.ts     # Preload script (IPC bridge)
│   ├── package.json       # Node.js dependencies
│   └── vite.config.ts     # Vite configuration
├── chroma/                 # Vector database storage
├── .venv/                  # Python virtual environment
├── .env                    # Environment variables (create this)
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

---

## 🚀 Development

### Run in Development Mode
```bash
cd frontend
npm run dev
```
This starts:
- Python FastAPI backend on `http://127.0.0.1:8000`
- Vite dev server on `http://localhost:5173`
- Opens in your browser (use `npm run dev:electron` for Electron window)

### Build for Production
```bash
cd frontend
npm run build:electron
```
Creates a distributable desktop app in `frontend/release/` directory.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is open source and available under the MIT License.

