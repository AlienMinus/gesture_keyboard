# 🖐️ Advanced Hand Gesture Detection with Onscreen Keyboard using OpenCV & MediaPipe

This project builds upon real-time hand gesture recognition by integrating a **virtual onscreen keyboard**. Users can "type" using finger gestures—simulating **Backspace**, **Shift**, **Space**, and **typing letters** via just their hands and a webcam!

## 🔍 Key Features

- 🎯 Real-time **dual-hand** tracking using MediaPipe
- ⌨️ Gesture-controlled **onscreen keyboard** with:
  - Shift for case toggling
  - Backspace for deletion
  - Spacebar support
- 👆 Pointer interaction using **index finger gesture**
- ✊ Recognizes multiple gestures like:
  - Thumbs Up 👍
  - Peace ✌️
  - Fist ✊
  - OK 👌
  - Call Me 🤙
  - Rock 🤘
  - Middle Finger 🖕
  - Two, Three, Four fingers
- 💬 Text display window showing typed content
- 🖼️ Fully centered and responsive onscreen keyboard

## 🧠 How It Works

- MediaPipe Hands detects 21 hand landmarks in real time.
- A function determines which fingers are up using landmark comparisons.
- If only the index finger is up, the system treats it as a pointer to "press" virtual keys.
- A debounce logic avoids repeated key entries.

## ⚙️ Tech Stack

- **Python 3.x**
- **OpenCV** – for computer vision & UI rendering
- **MediaPipe** – for hand tracking & gesture detection
- **NumPy** – for frame handling

## 🧭 Project Structure

```bash
gesture_keyboard/
├── keyboard.py     # Main Python file
├── README.md               # Project documentation
```

## 🚀 Getting Started
1. Clone the Repository
```bash
git clone https://github.com/AlienMinus/gesture_keyboard.git
cd gesture_keyboard
```
2. Install Dependencies
```bash
pip install opencv-python mediapipe numpy
```
3. Run the Application
```bash
python keyboard.py
```

## 📦 Future Improvements
- 🔊 Add Text-to-Speech for typed output
- 🧠 Train a gesture classification model using CNNs for better accuracy
- 🖥️ Control OS-level functions (e.g., open browser/app)
- 🖐️ Sign language to text translator

👨‍💻 Author
Manas Ranjan Das
ECE + AI-ML Enthusiast | Full-Stack & Embedded Developer
📧 dasmanasranjan2005@gmail.com