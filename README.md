# Interactive Multimodal AI Buddy

**An AI Companion for Real-Time Emotion-Aware Conversations**

A **desktop application** built with **Electron + React** frontend and **Python FastAPI** backend, providing an intelligent, multimodal AI companion with facial recognition, voice interaction, and personalized memory.

---

## 🚀 Overview

The **Interactive Multimodal AI Buddy** is an advanced AI companion designed to engage users in real-time, emotionally intelligent conversations. By integrating multimodal inputs—such as voice and facial expressions—this system adapts its responses based on the user's emotional state, fostering more natural and empathetic interactions.

This is a **desktop application** built with modern web technologies (Electron + React) for the UI and Python for AI processing.

---

## 🧠 Features

- **Meet Deva**: A distinct AI personality that remembers you and evolves with conversation
- **Facial Recognition Authentication**: Secure login using face embeddings (FaceNet)
- **Multimodal Interaction**: Real-time voice and video processing (Gemini Live API)
- **Emotional Intelligence**:  Detects emotions from facial expressions and voice tone
- **Long-term Memory**: Remembers preferences and events using PostgreSQL + pgvector
- **Agentic Workflow**: Uses **LangGraph** for sequential reasoning and response generation
- **Modern Desktop UI**: Glassmorphism design with smooth, reactive animations
- **Privacy First**: Local database processing for user data

---

## 🛠️ Technology Stack

### Frontend (Electron + React)
- **Desktop Framework**: Electron
- **UI Framework**: React 18 + TypeScript
- **Styling**: CSS Modules with glassmorphism effects
- **Media**: Web Audio API & MediaStream (Camera/Mic)
- **Real-time**: WebSocket for low-latency AI streaming

### Backend (Python FastAPI)
- **API Framework**: FastAPI (Async)
- **AI Core**: Google Gemini 2.0 Flash (Live API)
- **Agent Orchestrator**: **LangGraph** (Reasoning -> Generation flow)
- **Vision/Auth**: OpenCV, FaceNet-PyTorch
- **Database**: PostgreSQL (User data + vector embeddings with `pgvector`)
- **Infrastructure**: Async connection pooling (`asyncpg`)

---

## 🔧 Installation

### Prerequisites
- **Node.js** 22+ and **npm**
- **Python** 3.12+
- **PostgreSQL** 13+ (with `vector` extension)
- Webcam and microphone
- Windows/Linux/macOS

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/theankitdash/Interactive-Multimodal-AI-Buddy.git
   cd Interactive-Multimodal-AI-Buddy
   ```

2. **Database Setup**:
   - Install PostgreSQL
   - Create database and enable pgvector:
     ```sql
     CREATE DATABASE multimodal_buddy;
     \c multimodal_buddy
     CREATE EXTENSION vector;
     ```

3. **Backend Setup**:
   ```bash
   # Create virtual environment
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/macOS
   
   # Install dependencies
   pip install -r backend/requirements.txt
   ```

4. **Frontend Setup**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

5. **Configure environment variables**:
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_key
   NVIDIA_API_KEY=your_nvidia_key (for reasoning node)
   
   # Database
   DB_USER=postgres
   DB_PASSWORD=your_password
   DB_NAME=multimodal_buddy
   DB_HOST=localhost
   DB_PORT=5432
   ```

6. **Run the application**:

   **Development (Recommended)**:
   ```bash
   cd frontend
   npm run dev
   ```
   *Starts FastAPI backend (port 8000) and React frontend (port 5173)*

   **Production Build**:
   ```bash
   cd frontend
   npm run build:electron
   ```

---

## 📖 Usage

1. **Registration**: 
   - New users must register with face data.
   - Look at the camera to capture face embeddings.
   
2. **Login**: 
   - Seamless hands-free login using facial recognition.
   
3. **Chat with Deva**: 
   - Speak naturally! Deva listens and responds with voice.
   - She sees you through the camera to understand context.
   
4. **Controls**:
   - **Mute/Unmute**: Toggle microphone privacy.
   - **Camera**: Toggle video input.
   - **Logout**: Securely end session.

---

## 📁 Project Structure

```
Interactive-Multimodal-AI-Buddy/
├── backend/                 # Python FastAPI backend
│   ├── ai/                 # Real-time AI (Gemini Live handler)
│   ├── graphs/             # LangGraph workflows (Agent logic)
│   ├── nodes/              # Graph nodes (Reasoning, Generation)
│   ├── routes/             # API endpoints (Auth, Assistant, Media)
│   ├── utils/              # Shared utilities (DB, Memory, Face)
│   ├── main.py             # App entry point & lifespan manager
│   └── requirements.txt    # Python dependencies
├── frontend/               # Electron + React frontend
│   ├── src/               
│   │   ├── components/    # UI Components (AssistantScreen, etc.)
│   │   ├── hooks/         # Custom hooks (useAudio, useCamera)
│   │   └── context/       # Global state (User, WebSocket)
│   └── electron/          # Main process integration
├── .venv/                  # Virtual environment
├── .env                    # Config (API Keys, DB)
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

