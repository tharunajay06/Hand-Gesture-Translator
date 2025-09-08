# Hand Gesture Translator

![Project Demo](assets/demo.gif)

## Overview
*Hand Gesture Translator* is an AI and Computer Vision-based system designed to help communication for people with speech or hearing impairments. It recognizes hand gestures in real-time and converts them into both text and speech output.

Built with *Python, OpenCV, MediaPipe, CNN models, and Text-to-Speech (TTS)*.

---

## Features
- Real-time hand gesture recognition using MediaPipe
- Gesture classification with a pre-trained CNN model
- Converts gestures into text and speech (TTS)
- Optimized for low-latency and embedded devices like Raspberry Pi
- "Clear" gesture to reset sentence construction
- Visual feedback with webcam overlay and progress bar

---

## Tech Stack
- *Python* – Core programming language
- *OpenCV* – Real-time video processing
- *MediaPipe* – Hand landmark detection
- *CNN (scikit-learn / joblib)* – Gesture classification
- *gTTS / pyttsx3* – Text-to-Speech

---

## Installation
1. Clone the repository:
```bash
git clone https://github.com/<your-username>/Hand-Gesture-Translator.git
cd Hand-Gesture-Translator
