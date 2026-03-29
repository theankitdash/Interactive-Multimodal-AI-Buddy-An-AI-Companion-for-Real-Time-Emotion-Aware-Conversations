# Interactive Multimodal AI Buddy

**An AI Companion for Real-Time Emotion-Aware Conversations**

A **desktop application** built with **Electron + React** frontend and **Python FastAPI** backend, providing an intelligent, multimodal AI companion with facial recognition, voice interaction, real-time vision understanding, and personalized long-term memory.

---

## 🚀 Overview

The **Interactive Multimodal AI Buddy** is an advanced AI companion designed to engage users in real-time, emotionally intelligent conversations. By integrating multimodal inputs—voice, facial expressions, and live camera vision—the system adapts its responses based on the user's emotional state and visual context, fostering natural and empathetic interactions.

This is a **desktop application** built with modern web technologies (Electron + React) for the UI and Python for AI processing, featuring a **dual-socket architecture** that separates real-time audio streaming from cognitive reasoning.

---

## 🧠 Features

- **Meet Deva**: A distinct AI personality that remembers you and evolves with conversation
- **Facial Recognition Authentication**: Secure hands-free login using face embeddings (FaceNet, multi-sample registration)
- **Real-Time Voice Conversation**: Bidirectional audio streaming via Gemini Live API (native audio)
- **Vision Understanding**: Periodic scene analysis using Gemini 2.5 Flash — Deva sees and understands your environment
- **Intelligent Reasoning**: Intent classification (Chat / Fact / Event) via locally fine-tuned Mistral 7B (LoRA + DPO) through LangGraph
- **Continuous RL Improvement**: Automatic feedback collection from interactions, periodic DPO training for model improvement
- **Long-Term Memory**: Stores preferences, memories, and events using PostgreSQL + pgvector with semantic vector search
- **Context Injection**: Retrieves stored knowledge and upcoming events, injecting them into Gemini's live audio context
- **Emotional Intelligence**: Detects emotions from facial expressions and adapts conversational tone
- **Dual-Socket Architecture**: Separates audio streaming (low-latency) from cognitive processing (reasoning + memory)
- **Modern Desktop UI**: Glassmorphism design with animated backgrounds and reactive controls

---

## 🏗️ Architecture

The system uses a **dual-WebSocket architecture** bridged by a `SessionRegistry`:

```
┌──────────────────────────────────────────────────────────────────┐
│                        Frontend (Electron + React)               │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ AuthScreen   │  │ AssistantScreen│  │ AnimatedBackground      │ │
│  │ (FaceNet)    │  │ (Voice + Video)│  │ (Glassmorphism)         │ │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────────────┘ │
│         │                 │                                       │
│         │    ┌────────────┼──────────────┐                        │
│         │    │ useAudio   │ useCamera    │ useMicrophone           │
│         │    └────────────┼──────────────┘                        │
└─────────┼────────────────┼───────────────────────────────────────┘
          │                │
     REST API        Two WebSockets
          │          ┌─────┴──────┐
          │          │            │
┌─────────▼──────────▼────────────▼────────────────────────────────┐
│                     Backend (Python FastAPI)                      │
│                                                                  │
│  ┌──────────┐  ┌──────────────────┐  ┌─────────────────────────┐ │
│  │ /api/auth │  │ /ws/assistant     │  │ /ws/cognition           │ │
│  │ (REST)    │  │ (Audio Socket)    │  │ (Cognition Socket)      │ │
│  └──────────┘  └────────┬─────────┘  └────────────┬────────────┘ │
│                         │                          │              │
│                         │    SessionRegistry       │              │
│                         │◄────────────────────────►│              │
│                         │    (bridges both)        │              │
│                         │                          │              │
│              ┌──────────▼──────────┐    ┌──────────▼──────────┐   │
│              │ Gemini Live API     │    │ LangGraph Pipeline   │   │
│              │ (Audio Streaming)   │    │ Reasoning → Generation│  │
│              │ + VisionAnalyzer    │    │ (Local Mistral 7B)   │   │
│              └─────────────────────┘    └──────────┬──────────┘   │
│                                                    │              │
│                                         ┌──────────▼──────────┐   │
│                                         │ PostgreSQL + pgvector│   │
│                                         │ (Memory & Events)    │   │
│                                         └─────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. User speaks → Audio Socket streams to **Gemini Live API** → Audio response streamed back
2. Gemini transcribes user speech → forwarded to **Cognition Socket** via `SessionRegistry`
3. Cognition runs **LangGraph** pipeline: **Reasoning** (local Mistral 7B classifies intent, extracts facts/events) → **Generation** (retrieves memories, builds context)
4. Generated context is **injected back** into Gemini's live session for personalized responses
5. **VisionAnalyzer** periodically analyzes camera frames via Gemini 2.5 Flash and injects scene descriptions

---

## 🛠️ Technology Stack

### Frontend (Electron + React)
| Layer | Technology |
|-------|-----------|
| Desktop Framework | Electron 40 |
| UI Framework | React 18 + TypeScript |
| Build Tool | Vite 7 |
| Styling | CSS Modules with glassmorphism effects |
| Media | Web Audio API & MediaStream (Camera/Mic) |
| Real-time | Dual WebSocket connections (Audio + Cognition) |

### Backend (Python FastAPI)
| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI (Async, WebSocket) |
| Voice AI | Google Gemini 2.5 Flash (Native Audio Live API) |
| Vision AI | Google Gemini 2.5 Flash (Scene analysis) |
| Reasoning & Generation | Local Mistral 7B (`mistralai/Mistral-7B-Instruct-v0.3`, 4-bit quantized via bitsandbytes + PEFT/LoRA) |
| Continuous RL | DPO (Direct Preference Optimization) via TRL, with automatic feedback collection |
| Agent Orchestrator | LangGraph (Conditional Reasoning → Generation flow) |
| Embeddings | Sentence Transformers (`all-mpnet-base-v2`, 768 dims) |
| Face Auth | OpenCV + FaceNet-PyTorch (512-dim embeddings) |
| Database | PostgreSQL 13+ with `pgvector` (cosine similarity search) |
| Connection Pooling | `asyncpg` (5–20 connections, auto-init schema) |

---

## 🔧 Installation

### Prerequisites
- **Node.js** 22+ and **npm**
- **Python** 3.12+
- **PostgreSQL** 13+ (with `vector` extension)
- **NVIDIA GPU** with ≥ 8 GB VRAM (for local Mistral 7B inference)
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
     CREATE EXTENSION pgcrypto;
     ```
   > **Note:** The backend auto-initializes all tables, indexes, enums, and triggers on startup via `db_connect.init_db()`.

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
   Create a `.env` file in the `backend/` directory:
   ```env
   GEMINI_API_KEY=your_gemini_key

   # Local Mistral 7B (auto-downloads from HuggingFace on first run)
   LOCAL_MODEL_PATH=mistralai/Mistral-7B-Instruct-v0.3

   # Database
   DB_USER=postgres
   DB_PASSWORD=your_password
   DB_NAME=multimodal_buddy
   DB_HOST=localhost
   DB_PORT=5432
   ```
   > **Note:** First startup downloads ~4 GB model from HuggingFace Hub (cached at `~/.cache/huggingface/`). Subsequent starts load from cache (~30s).

6. **Run the application**:

   **Development (Recommended)**:
   ```bash
   cd frontend
   npm run dev
   ```
   *Starts FastAPI backend (port 8000) and Vite dev server (port 5173) concurrently*

   **Production Build**:
   ```bash
   cd frontend
   npm run build:electron
   ```

---

## 📖 Usage

1. **Registration**: 
   - New users register with face data (multi-sample capture for accuracy).
   - Look at the camera to capture 50 face samples for robust embeddings.
   
2. **Login**: 
   - Hands-free login using facial recognition (cosine similarity matching).
   
3. **Chat with Deva**: 
   - Speak naturally! Deva listens and responds with voice in real-time.
   - He sees you through the camera to understand visual context.
   - He remembers your preferences, past conversations, and scheduled events.
   
4. **Controls**:
   - **Mute/Unmute**: Toggle microphone privacy.
   - **Camera**: Toggle video input (vision context updates accordingly).
   - **Logout**: Securely end session.

---

## 📁 Project Structure

```
Interactive-Multimodal-AI-Buddy/
├── backend/                     # Python FastAPI backend
│   ├── ai/                     # AI model clients
│   │   ├── gemini_handler.py   # Gemini Live API (bidirectional audio streaming)
│   │   ├── local_mistral.py    # Local Mistral 7B client (4-bit quantized, LangChain-compatible)
│   │   └── vision_analyzer.py  # Real-time scene analysis (Gemini 2.5 Flash vision)
│   ├── graphs/                 # LangGraph workflows
│   │   └── agent_graph.py      # Conditional Reasoning → Generation pipeline
│   ├── nodes/                  # Graph nodes
│   │   ├── reasoning.py        # Intent classification + fact/event extraction (Local Mistral)
│   │   └── generation.py       # Context-enriched response generation (Local Mistral)
│   ├── routes/                 # API endpoints
│   │   ├── auth.py             # Face registration & recognition (REST)
│   │   ├── assistant.py        # Audio WebSocket (Gemini Live streaming)
│   │   ├── cognition.py        # Cognition WebSocket (reasoning + memory pipeline)
│   │   └── media.py            # Media utilities
│   ├── utils/                  # Shared utilities
│   │   ├── db_connect.py       # PostgreSQL pool + auto schema initialization
│   │   ├── face_utils.py       # FaceNet embedding extraction
│   │   ├── feedback_collector.py # Interaction logging for continuous RL (DPO training data)
│   │   └── memory.py           # Vector knowledge store + semantic retrieval
│   ├── training/               # Continuous RL improvement pipeline
│   │   ├── config/dpo_config.yaml  # LoRA + DPO hyperparameters
│   │   ├── export_feedback.py  # Export feedback_logs → DPO preference pairs
│   │   ├── train_dpo.py        # DPO fine-tuning with LoRA adapters
│   │   ├── merge_and_deploy.py # Merge LoRA weights → production model
│   │   └── evaluate.py         # Benchmark intent accuracy & response quality
│   ├── config.py               # Centralized configuration (model paths, params)
│   ├── models.py               # Pydantic request/response models
│   ├── session_registry.py     # Dual-socket session bridge (Audio ↔ Cognition)
│   ├── main.py                 # App entry point & lifespan manager
│   └── requirements.txt        # Python dependencies
├── frontend/                    # Electron + React frontend
│   ├── src/
│   │   ├── components/         # UI Components
│   │   │   ├── AssistantScreen.tsx  # Main conversation interface
│   │   │   ├── AuthScreen.tsx       # Face registration & login
│   │   │   └── AnimatedBackground.tsx # Animated glassmorphism backdrop
│   │   ├── hooks/              # Custom React hooks
│   │   │   ├── useAudio.ts     # WebSocket audio streaming & playback
│   │   │   ├── useCamera.ts    # Camera stream & frame capture
│   │   │   └── useMicrophone.ts # Mic capture & PCM encoding
│   │   ├── context/            # Global state
│   │   │   └── AppContext.tsx   # App-wide state (auth, mode, status)
│   │   ├── config/             # Frontend configuration
│   │   ├── types/              # TypeScript type definitions
│   │   └── utils/              # Frontend utilities
│   ├── electron/               # Electron main process
│   ├── public/                 # Static assets
│   └── package.json            # Dependencies & scripts
├── .gitignore
└── README.md
```

---

## 🗄️ Database Schema

The backend auto-creates the following schema on startup:

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `user_details` | User profiles | `username`, `name`, `face_embedding` (vector 512) |
| `user_knowledge` | Long-term memory (facts) | `fact`, `category` (preference/memory/skill/habit), `embedding` (vector 768) |
| `events` | Scheduled events & reminders | `description`, `event_time`, `type`, `status`, `priority` |
| `feedback_logs` | RL training data (DPO) | `prompt`, `response`, `node_type`, `intent_parse_success`, `response_quality_signal` |

Custom enum types: `knowledge_category`, `event_type`, `event_status`

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

### Available Scripts
| Script | Description |
|--------|-------------|
| `npm run dev` | Start backend + frontend concurrently |
| `npm run dev:electron` | Launch Electron desktop window |
| `npm run build:electron` | Build distributable desktop app |
| `npm run typecheck` | TypeScript type checking |
| `npm run lint` | ESLint code linting |
| `npm run format` | Prettier code formatting |

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
