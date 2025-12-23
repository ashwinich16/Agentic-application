# Agentic Application (FastAPI + LangGraph + Streamlit)

An **agentic multimodal AI application** that autonomously understands user intent and executes the correct task across **text, image, PDF, audio, and YouTube tancripts**.

The system uses **LangGraph** for orchestration, **FastAPI** for backend APIs, and **Streamlit** for a simple chatbot-style UI.

## Features
#### 📝 Text-based Tasks
- **Summarization**
  - 1-line summary
  - 3 bullet points
  - 5-sentence paragraph
- **Sentiment Analysis**
  - Sentiment label (Positive / Negative / Neutral)
  - Confidence score
  - One-line justification
- **Conversational Question Answering**
- **Code Explanation**
  - What the code does
  - Potential bugs or issues
  - Time complexity analysis

#### Image / PDF Processing
- Image OCR using **Tesseract**
  - Returns cleaned text + OCR confidence
- PDF text extraction using **pypdf**
  - Suitable for digitally generated PDFs

#### YouTube
- Detects YouTube URL anywhere in input
- Extracts video ID
- Fetches transcript or returns fallback message

#### Audio
- Offline audio transcription using **Whisper** ans summarization of the audio

#### Agent Capabilities
- Intent classification using LLM
- Autonomous task routing
- Follow-up questions when required inputs are missing
- Per-session in-memory context

## Architecture Overview
User (Text / File Upload)  
      🠋      
Intent Classifier (LLM)  
      🠋  
LangGraph State Machine  
      🠋  
Task Worker (audio_worker/code_worker/text_worker/extraction_worker/youtube_worker)   
      🠋  
Post-processing
      🠋  
Response  

## Tech Stack
| Layer              | Technology
|--------------------|-----------
| Backend API        | FastAPI 
| Orchestration      | LangGraph 
| Frontend           | Streamlit 
| LLM                |LLaMA via NVIDIA NIM
| OCR                | Tesseract
| PDF Parsing        | pypdf 
| Audio Transcription| Whisper (offline)
| Audio Processing   | FFmpeg

## System dependencies
Tessaract OCR
Download https://github.com/UB-Mannheim/tesseract/wiki 
Add to System Variables path:
C:\Program Files\Tesseract-OCR\
Download ffmpeg-8.0.1-essentials_build from https://www.gyan.dev/ffmpeg/builds/
exatract the zip file
Add to System Variables path:
C:\ffmpeg\ffmpeg-8.0.1-essentials_build\bin

conda create -n agent python=3.10
conda activate agent
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
streamlit run streamlit.py

## Sample Test Cases
## PDF Summarization
Upload a PDF
Input: Summarize the document
Output: structured summary
## Image OCR
Upload an image
Output: extracted text + OCR confidence
## Audio Lecture (5 min)
Upload audio file
Input: Summarize this audio
Output:
Summary 




