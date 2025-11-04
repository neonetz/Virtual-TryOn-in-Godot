# Virtual Try-On with Godot and Python

This project is a real-time virtual try-on application that uses a classical computer vision pipeline for face detection and integrates it with the Godot game engine for rendering. The backend is built with Python and OpenCV, while the frontend is a Godot 4 application. The two parts communicate over a UDP network connection.

## Features

- **Real-time Face Detection**: Uses a pipeline of Haar Cascades, ORB features, Bag of Visual Words (BoVW), and a Support Vector Machine (SVM) classifier to detect faces in a webcam feed.
- **Virtual Mask Overlay**: Overlays different Halloween-themed masks on the detected faces.
- **Dynamic Mask Selection**: Allows the user to change masks in real-time from the Godot UI.
- **Godot Integration**: The processed video feed from the Python backend is streamed to a Godot application for display.
- **UDP Communication**: Efficiently streams video data and commands between the Python backend and the Godot frontend.

## Architecture

The project is divided into two main components:

1.  **Backend (`backend/`)**: A Python application responsible for:
    - Capturing video from the webcam.
    - Performing face detection using a custom-trained SVM model.
    - Overlaying the selected mask on the detected faces.
    - Encoding the final video frame as a JPEG image.
    - Streaming the image data to the Godot client via UDP.
    - Listening for commands from the Godot client (e.g., to change the mask).

2.  **Frontend (`godot/`)**: A Godot 4 application that:
    - Receives the JPEG image data from the Python backend via UDP.
    - Decodes the image and displays it in real-time.
    - Provides a user interface for selecting different masks.
    - Sends commands to the backend to change the currently displayed mask.

### Communication Protocol

-   **Video Stream (Python -> Godot)**: The backend sends JSON objects over UDP to port `5555`. Each object contains the `image` (as a base64 encoded JPEG), `frame_id`, `fps`, and the number of `faces` detected.
-   **Commands (Godot -> Python)**: The Godot client sends JSON commands to the backend on port `5556`. The primary command is `change_mask`, which includes the filename of the desired mask.

## How it Works

### Backend Pipeline

The face detection process in the backend follows these steps:

1.  **ROI Proposal**: A Haar Cascade classifier is used to propose Regions of Interest (ROIs) where faces are likely to be.
2.  **Feature Extraction**: For each ROI, Oriented FAST and Rotated BRIEF (ORB) keypoints and descriptors are extracted.
3.  **Bag of Visual Words (BoVW)**: The ORB descriptors are encoded into a histogram using a pre-trained BoVW "codebook" (a KMeans model). This converts the variable-length descriptors into a fixed-size feature vector.
4.  **SVM Classification**: The feature vector is fed into a pre-trained Support Vector Machine (SVM) classifier, which determines if the ROI contains a face.
5.  **Mask Overlay**: If a face is detected, the selected mask is overlaid onto the face region. The system can also detect the angle of the eyes to rotate the mask for a better fit.
6.  **UDP Streaming**: The final frame is encoded and sent to Godot.

### Frontend (Godot)

The Godot application is relatively simple:

1.  **UDP Listener**: A `PacketPeerUDP` is set up to listen for incoming data on the specified port.
2.  **Frame Processing**: When a packet is received, the JSON data is parsed. The base64 image string is decoded into a JPEG buffer.
3.  **Texture Update**: The JPEG buffer is loaded into an `Image`, which is then used to create an `ImageTexture`. This texture is displayed in a `TextureRect` node, effectively showing the video stream.
4.  **UI Interaction**: A separate UI scene allows the user to click on different mask icons. Clicking a mask sends a UDP command to the backend to switch the active mask.

## How to Run

### 1. Backend

First, you need to have Python and the required libraries installed.

```bash
# Navigate to the backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Run the UDP server
python webcam_udp_server.py
```

The backend will start capturing the webcam, performing face detection, and streaming the output to `127.0.0.1:5555`.

### 2. Frontend

You need Godot 4 to run the frontend.

1.  Open the Godot Engine.
2.  Click "Import" and navigate to the `godot/` directory in this project. Select the `project.godot` file.
3.  Once the project is imported, open it.
4.  Run the main scene (`MainMenu.tscn`) by pressing F5.

The application will open, and you should see the video feed from the Python backend. You can then navigate to the "Try On" scene to see the virtual masks and select different ones from the right-hand menu.
