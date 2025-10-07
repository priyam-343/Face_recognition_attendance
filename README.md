Employee Authentication using Face Recognition

This is a robust, cross-platform backend application designed for automated employee attendance tracking using real-time facial recognition and liveness detection. It has been successfully optimized to run seamlessly on both Windows and macOS (Apple Silicon M1/M2/M3).

Features
This application provides a powerful and secure attendance system:

Real-time Biometric Check-in: Processes video streams to verify employee identity against stored facial embeddings.

Performance Stability: Uses asynchronous processing to run heavy recognition tasks (dlib, face encoding) in the background, ensuring a fluid video stream without freezing.

Liveness Detection: Requires the user to blink and move their head slightly to prevent spoofing using static images.

Instant Confirmation Service: Sends an automatic, personalized email confirmation upon successfully marking attendance.

Secure Access: Uses Flask-Login for secure administration access.

Data Management: Records attendance logs in a CSV file and manages employee data via an SQLite database.

Tech Stack
Python 3.8 (Recommended for stability)

Flask (Web Framework)

SQLAlchemy (Database ORM)

OpenCV & dlib (Computer Vision & Facial Landmarks)

face_recognition & DeepFace (Recognition & Core Models)

Flask-Mail (Email Service)

Pandas (Data Handling)

How to Run Locally
1. Prerequisites (Required for All Systems)
Python 3.8 (Recommended via Conda/Miniforge)

The facial landmark file: shape_predictor_68_face_landmarks.dat must be placed in the project's root directory.

Active Gmail Account (for sending emails)

2. Platform-Specific Installation

A. Setup for macOS (Apple Silicon M1/M2/M3)
This setup uses Miniforge and Homebrew to compile dependencies for the ARM architecture, ensuring native speed.

Create Environment & Install CMake:

# Create the virtual environment
conda create --name face_rec_app python=3.8 

# Activate the environment
conda activate face_rec_app

# Install CMake (C++ builder required by dlib)
brew install cmake 

Install Python Libraries:

pip install python-dotenv wheel dlib Flask flask_sqlalchemy Flask-Mail flask_login pandas face_recognition imutils opencv-python playsound deepface plotly plotly-express


B. Setup for Windows
This setup assumes you are using the standard Anaconda/Miniconda Python distribution.

Create Environment & Activate:

# Create the virtual environment
conda create --name face_rec_app python=3.8 

# Activate the environment
conda activate face_rec_app

Install Python Libraries:
The following single command will install all required libraries, including the compiled Windows versions of OpenCV and dlib.

pip install python-dotenv wheel dlib Flask flask_sqlalchemy Flask-Mail flask_login pandas face_recognition imutils opencv-python playsound deepface plotly plotly-express

3. Environment Variables (Required for Email)
Create a file named .env in the project root directory and add your login details.

MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=True
MAIL_USERNAME=your_sending_email@gmail.com
MAIL_PASSWORD=your_16_char_app_password
MAIL_SENDER=your_sending_email@gmail.com 

4. Final Step: Run the Application
Ensure your environment is active:

conda activate face_rec_app

Run the application from the project directory:

python app.py

The application will be accessible at http://127.0.0.1:7000.

Developed by (as group project)

1. Mahendar Singh Gurjar
2. Priyam Kumar
3. Swapnil
