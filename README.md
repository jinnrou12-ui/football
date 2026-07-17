# Football Video Analysis Studio Tracker

A professional full-stack computer vision application for tactical soccer video analysis. The system consists of a Python FastAPI backend leveraging YOLOv8 object detection with customized OpenCV visual filters, and a dark-themed React.js tactical workspace.

---

## Features

- **Automated Object Detection**: Detects players (`class 0`) and soccer balls (`class 32`) in high-fidelity video frames using YOLOv8.
- **Euclidean Possession Proximity Tracker**: Compares the bottom-center coordinate of each player's bounding box (their feet) to the center of the ball. If within 50 pixels, that player is flagged as the ball possessor.
- **Background Privacy Blur**: Smoothly applies a heavy `cv2.GaussianBlur` over the bounding box areas of all players *not* currently possessing the ball. The possessing player and the ball remain sharp and highlighted.
- **Dynamic Hex-to-BGR Styling**: Converts modern web color picker hex codes into OpenCV BGR structures to render custom-colored tactical tracking circles.

---

## 1. Backend Setup & Run

### Prerequisites
- Python 3.10+
- PyTorch (configured for GPU/CPU depending on hardware availability)

### Step-by-Step Installation
1. Open a terminal and navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows (cmd/powershell):
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the FastAPI development server:
   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```
   *Note: On its first run, the backend will automatically download the `yolov8n.pt` model weights.*

---

## 2. Frontend Setup & Run

### Prerequisites
- Node.js 18+
- npm (Node Package Manager)

### Step-by-Step Installation
1. Open a new terminal window/tab and navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install npm package dependencies:
   ```bash
   npm install
   ```
3. Launch the React development studio:
   ```bash
   npm start
   ```
   *This starts the application at `http://localhost:3000` with automated hot-reloading.*

---

## 3. Detailed Logic Flow

### Ball Possession Algorithm
The backend calculates distance using the Euclidean norm:
$$d = \sqrt{(x_{\text{ball}} - x_{\text{player}})^2 + (y_{\text{ball}} - y_{\text{player}})^2}$$
- **$x_{\text{player}}, y_{\text{player}}$**: Calculated as the bottom-middle of the detected person's bounding box:
  $$x_{\text{player}} = \frac{x_1 + x_2}{2}, \quad y_{\text{player}} = y_2$$
- **$x_{\text{ball}}, y_{\text{ball}}$**: Calculated as the center of the ball's bounding box:
  $$x_{\text{ball}} = \frac{x_1 + x_2}{2}, \quad y_{\text{ball}} = \frac{y_1 + y_2}{2}$$

If the minimum $d$ is less than $50$ pixels, that player is dynamically assigned the possession highlight, while others undergo Gaussian blurring.
