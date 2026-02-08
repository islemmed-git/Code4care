# M.O.H. - Integrated Cell Culture Monitoring System (CODE4CARE)

🏆 hashtag#HackInSaclay2026 – 2ᵉ place ! 

**M.O.H. (Monitor. Observe. Heal.)** is a comprehensive monitoring and decision-support system designed for cell culture laboratories, specifically tailored for skin cultivation therapies (e.g., for burn patients).

This application combines **real-time sensor simulation**, **automated health scoring**, and an **AI-powered RAG (Retrieval-Augmented Generation) chatbot** to assist lab technicians in maintaining optimal culture conditions and responding rapidly to critical events.

---

## 🚀 Features

### 📊 Live Dashboard

- **Real-time Monitoring**: Visualizes key environmental parameters:
  - Temperature (°C)
  - pH Level
  - Oxygen Saturation (% O₂)
  - Impedance (Cell Growth / Confluence Index)
- **Health Scoring**: Automatically calculates a global "Health Score" (0–100) and status (OK, WARNING, CRITICAL) based on parameter deviations.
- **Trend Analysis**: Interactive charts showing historical data for all sensors.

### 🤖 AI Playbook Assistant

- **RAG Chatbot**: Integrated AI assistant powered by Google Gemini and LangChain.
- **Context-Aware**: Answers troubleshooting questions using a strictly defined "Knowledge Base" of lab protocols (stored in `knowledge_base/`).
- **Source Citations**: Provides answers with direct references to source documents and links.

### 🚨 Smart Alerts & Notifications

- **Automated Detection**: Instantly flags "WARNING" or "CRITICAL" conditions.
- **Email Integration**: Sends automated emails via Resend to supervisors.
- **Progress Reports**: Scheduled updates on culture growth.
- **Critical Alerts**: Immediate notifications with AI-generated step-by-step solutions when issues occur.

### 🧪 Simulation Engine

- **Realistic Data**: Simulates sensor noise, drift, and logistic cell growth curves.
- **Scenario Testing**: "God Mode" allows users to inject faults (e.g., temperature spikes, contamination) to test system responses.

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/islemmed-git/Code4care.git
cd Code4care
```

## Install dependencies:

pip install -r requirements.txt

Environment Setup

Create a .env file in the root directory with the following keys:

# Required for AI Chatbot

GOOGLE_API_KEY=your_google_api_key

# Usage

    Start the Application
    streamlit run fintegrated_app_.py
