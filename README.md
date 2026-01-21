# Interactive Multimodal AI Buddy

**An AI Companion for Real-Time Emotion-Aware Conversations**

A **Kivy-based desktop application** that provides an intelligent, multimodal AI companion with facial recognition, voice interaction, and personalized memory.

---

## 🚀 Overview

The **Interactive Multimodal AI Buddy** is an advanced AI companion designed to engage users in real-time, emotionally intelligent conversations. By integrating multimodal inputs—such as voice and facial expressions—this system adapts its responses based on the user's emotional state, fostering more natural and empathetic interactions.

This is a **desktop application** built with Kivy for Python, designed to run locally on your system with full camera and microphone access.

---

## 🧠 Features

- **Facial Recognition Authentication**: Secure login using face embeddings with FaceNet
- **Emotion Recognition**: Utilizes facial expression analysis and voice tone detection to gauge user emotions
- **Multimodal Interaction**: Supports voice and video inputs for a comprehensive conversational experience
- **Contextual Awareness**: Remembers past interactions to provide contextually relevant responses using LangChain
- **Real-Time Processing**: Ensures immediate feedback during conversations with Gemini 2.0 Flash
- **Desktop UI**: Modern, animated Kivy interface with visual state indicators

---

## 🛠️ Technologies Used

- **Desktop Framework**: Kivy (Python-based UI)
- **Facial Detection**: OpenCV, FaceNet-PyTorch (MTCNN)
- **LLM Agent**: Gemini 2.0 Flash (Multimodal)
- **LLM Framework**: LangChain (For Personalization)
- **Database**: PostgreSQL (User data), Chroma (Vector DB for memory)
- **Audio**: sounddevice (Microphone and speaker I/O)

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- PostgreSQL database
- Webcam and microphone
- Windows/Linux/macOS

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/theankitdash/Interactive-Multimodal-AI-Buddy.git
   cd Interactive-Multimodal-AI-Buddy
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On Linux/macOS
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   NVIDIA_API_KEY=your_nvidia_api_key_here
   DATABASE_URL=postgresql://user:password@localhost:5432/ai_buddy
   ```

5. **Setup database**:
   
   Ensure PostgreSQL is running and create the required database and tables (check `utils/db_connect.py` for schema).

6. **Run the application**:
   ```bash
   python main.py
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

---

## 📁 Project Structure

```
Interactive-Multimodal-AI-Buddy/
├── ai/                    # AI handlers (Gemini, LangChain)
├── ui/                    # Kivy layout files
│   └── kv_layout.kv      # UI design
├── utils/                 # Utilities (face recognition, database)
├── chroma/               # Vector database storage
├── main.py               # Main Kivy application
├── .env                  # Environment variables (create this)
└── README.md             # This file
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is open source and available under the MIT License
