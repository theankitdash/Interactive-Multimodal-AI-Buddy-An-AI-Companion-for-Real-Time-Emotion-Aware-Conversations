# Quick Start

## 1. Install Dependencies

**From the `frontend/` directory:**
```bash
npm install
```

**From the `backend/` directory:**
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r ../requirements-backend.txt
```

## 2. Setup Environment

Ensure `.env` file exists in **project root** with:
```env
GEMINI_API_KEY=your_key_here
NVIDIA_API_KEY=your_key_here
DATABASE_URL=postgresql://user:pass@localhost:5432/ai_buddy
```

## 3. Run Development Mode

**From the `frontend/` directory:**
```bash
npm run dev
```

This starts:
- Backend (FastAPI) on `http://127.0.0.1:8000`
- Frontend (Vite) on `http://localhost:5173`
- Electron window automatically

## 4. Build Production App

**From the `frontend/` directory:**
```bash
npm run build:electron
```

Output: `frontend/release/` directory
