# Gemini Chatbot and Ollama Chatbot using LangChain

## Gemini Chatbot

## 📝 Project Summary
This project is a simple AI chatbot built using LangChain, Google Gemini API, and Streamlit. It allows users to ask questions and receive real-time AI-generated responses.

## 🔍 What it does
- Loads API key securely using `.env`
- Uses Gemini model (`gemini-3-flash-preview`) for generating responses
- Formats user input using prompt templates
- Processes input through LangChain pipeline
- Displays output using Streamlit UI

## ⚙️ Tech Stack
- LangChain
- Google Gemini API
- Streamlit
- Python

## 🔄 Workflow
User Input → Prompt Template → Gemini Model → Output Parser → Response

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run app.py

## 📸 Screenshots

### 🖥️ Chatbot UI
screenshots/Gemini_Chat_Bot_image.png

### 💬 User Input & Responsea
screenshots/even_odd1.png
screenshots/even_odd2.png
screenshots/even_odd3.png
screenshots/sample2.png

## Ollama Chatbot

## Ollama Chatbot using LangChain

## 📝 Project Summary
This project is a simple AI chatbot built using LangChain, Ollama (local LLM), and Streamlit. It allows users to ask questions and get responses from a locally running AI model.

## 🔍 What it does
- Uses Ollama to run LLM locally (no API required)
- Uses `gemma:2b` model for generating responses
- Formats user input using prompt templates
- Processes input through LangChain pipeline
- Displays chat interface using Streamlit

## ⚙️ Tech Stack
- LangChain
- Ollama (Local LLM)
- Streamlit
- Python

## 🔄 Workflow
User Input → Prompt Template → Ollama Model → Output Parser → Response

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run locallama.py

## 📸 Screenshots
### 🖥️ Chatbot UI
screenshots/Ollama_ChatBot.png

### 💬 User Input & Responsea
screenshots/Ollama_sample1.png
screenshots/Ollama_sample2.png
screenshots/Ollama_sample2.1.png

