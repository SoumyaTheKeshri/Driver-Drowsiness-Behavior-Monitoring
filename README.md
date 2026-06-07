# Driver-Drowsiness-Behavior-Monitoring
Real-time AI-based driver drowsiness and behavior monitoring system using computer vision, sensor fusion, and multi-level alerts.
AI Driver Monitoring System

A real-time driver safety system that detects drowsiness, 
distraction, and unsafe driving behaviour using computer 
vision and vehicle sensor fusion.

Built as a two-module system:
- Vision & AI Core — MediaPipe FaceMesh, EAR, MAR, 
  PERCLOS, head pose estimation
- Integration & Alerts — Sensor fusion, risk scoring, 
  multi-level alert engine, live dashboard, CSV logging

Tech Stack:
Python · OpenCV · MediaPipe · TensorFlow · Scikit-learn
NumPy · SciPy · Pygame

Features:
- Real-time face mesh with 468 landmarks
- Drowsiness scoring (Level 0–3)
- Vehicle sensor data integration (speed, braking, 
   acceleration)
- Unified risk score (0–1)
- 3-level alert system (INFO / WARNING / CRITICAL)
- Live web dashboard
- Per-session CSV logging & evaluation report

Academic Project — MCA Final Year | CSET312
