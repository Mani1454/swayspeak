## SwaySpeak: Interactive AI English Tutor 🗣️💡

A high-performance, ultra-low latency **interactive English conversation tutor** designed for real-time spoken practice.
Built for **natural conversation, real-time grammar coaching, and confidence building**.

Runs efficiently on a **standard laptop or cloud CPU** using the world's fastest voice and LLM APIs — no GPU required.

---

## 🚀 Try It Now (No Setup Required)

**Just launch the application to start speaking and practicing:**

👉 **[Live Web App (Render)](https://swayspeak.onrender.com)**  
👉 **[Local Server (Offline / Development)](http://127.0.0.1:8000)**

---

## ⚡ Key Features

* **Spoken Coaching Aloud** — Corrects grammar and phrasing naturally within spoken conversation, explaining native phrasing tips.
* **Visual English Coach Cards** — Real-time display showing "What You Should Say", sentence improvements, and tips in the UI.
* **Patient Turn Endpointing** — Natural speech pauses (up to 1.2s) without premature interruptions so you can articulate freely.
* **Sub-Second Voice Latency** — Streams responses almost instantly like a real human speaking partner.
* **Smart Natural Interruptions** — Speak anytime or continue thoughts smoothly.
* **Clean Session Privacy** — Fresh conversation memory per session with one-click context clearing.

---

## 🎮 How to Use

1. **Click the Link** above to open the web app.
2. **Tap the Glowing Orb** in the center.
3. **Allow Microphone Access** when asked.
4. **Speak Naturally** — Ask about my life, my superpower, or anything else!
5. **Interrupt Anytime** — If I'm talking too much, just speak over me.

---

## 📂 For Developers (Technical Details)

If you want to run this code yourself or understand how it works:

### **Core Backend**

| File                         | Role                                                                              |
| ---------------------------- | --------------------------------------------------------------------------------- |
| `server.py`                  | The FastAPI server. Manages WebSockets, routing, and audio buffers.               |
| `speech_pipeline_manager.py` | Orchestrates listening, thinking, and speaking using threads for low latency.     |
| `transcribe.py`              | Connects to Deepgram STT. Handles live transcription and end-of-speech detection. |
| `audio_in.py`                | Receives raw mic audio and prepares it safely for transcription.                  |
| `llm_module.py`              | LLM wrapper for Groq, OpenAI, MegaLLM with streaming text output.                 |
| `audio_module.py`            | Text-to-Speech module (Deepgram). Streams generated audio back to client.         |

### **Frontend (`/static`)**

| File                      | Purpose                                                      |
| ------------------------- | ------------------------------------------------------------ |
| `index.html`              | Main UI with grid layout + glowing core animation.           |
| `app.js`                  | Manages WebSockets, animations, audio context, and UI logic. |

---

## 💻 Run Locally (Windows/Mac/Linux)

Since the configuration is already set up, you can run this on your own computer easily!

### **Windows Users**
1. Download the code (Click "Code" -> "Download ZIP" and extract it).
2. Double-click the `start_windows.bat` file.
3. That's it! The bot will open in your browser.

### **Mac / Linux Users**
1. Open your terminal in the folder.
2. Run this command:
   ```sh
   ./start_unix.sh
   ```
3. The bot will launch instantly.

---

## 🛠 Manual Installation (For Developers)

---

## 👨‍💻 Developed By

**Manish Kumar**
🚀 Building the future of ambient AI.
