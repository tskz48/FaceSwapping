Face Swapping AI for Videos

A Python-based AI application for performing face swapping on videos using SimSwap and a FastAPI backend.

The project takes one or more source face images and a target video, runs face-swapping inference through SimSwap, and generates a processed output video. It also supports specifying a particular target face when working with videos containing multiple people.

Project Description

This project demonstrates how deep-learning-based face swapping can be integrated into a backend application for video processing.

The application uses SimSwap as the underlying face-swapping model. A Python/FastAPI layer prepares the source images, target video, output path, and optional target-person image before running the face-swapping pipeline. The project also incorporates hair segmentation and masking to provide more control over which regions are preserved or modified in the processed video.

The overall pipeline is:

Source face image(s)
        +
Target video
        ↓
FastAPI / Python backend
        ↓
SimSwap inference
        ↓
Face detection and identity transfer
        ↓
Processed video
        ↓
result.mp4

The goal of the project is to provide a simple application-level implementation of AI video face swapping while keeping the model inference logic separated from the API/backend layer.

Features

Video face swapping using a pretrained deep learning model

Support for multiple source face images

Support for selecting a specific target person in a video

Hair segmentation to identify hair regions

Masking to control which regions are preserved or modified during face swapping

FastAPI backend

SimSwap integration through Python subprocess

Automatic generation of an output video

CORS support for connecting the backend to a frontend application

Local file-based input and output handling

Technologies Used

Python

FastAPI

SimSwap

PyTorch

OpenCV

InsightFace

MoviePy

NumPy

Uvicorn

How It Works

1. Source Faces

The application uses source images containing the identities that will be transferred to faces in the target video.

The current project configuration uses files such as:

source1.jpg
source2.jpg
source3.jpg
source4.jpg

The paths are prepared in Python and passed to the face-swapping pipeline.

2. Target Video

The video to be processed is provided as the target video.

For example:

target_fast2.mp4

SimSwap processes the video frame by frame and transfers the identity from the source image to the detected target face.

3. Specific Target Face

For videos containing multiple people, the project can use a reference image such as:

target_person.jpg

to identify which person in the target video should receive the swapped face.

Hair Segmentation and Masking

The project also includes hair segmentation and masking as part of the video-processing pipeline.

Hair segmentation is used to identify the hair region separately from the rest of the face. A mask can then be applied so that selected regions are preserved or excluded from the face-swapping operation.

This helps provide greater control over the final composition by separating facial identity transfer from surrounding regions such as hair.

Conceptually:

Video frame
    ↓
Hair segmentation
    ↓
Hair / non-hair mask
    ↓
Face-swapping process
    ↓
Masked composition
    ↓
Final processed frame

The masking stage can be used to combine the swapped facial region with preserved regions from the original frame before the final video is generated.

4. SimSwap Inference

The FastAPI backend runs the SimSwap video inference script from the SimSwap directory using Python's subprocess module.

Conceptually:

FastAPI request
      ↓
Code.py
      ↓
subprocess
      ↓
SimSwap/test_video_swapsingle.py
      ↓
Face-swapped video

This keeps the API logic separate from the deep learning inference implementation.

5. Output

After processing is complete, the generated face-swapped video is stored as:

result.mp4

The backend can then return or serve this video to the client.

Project Structure

A simplified structure of the project is:

FaceSwapping/
│
├── Code.py
│
├── source1.jpg
├── source2.jpg
├── source3.jpg
├── source4.jpg
│
├── target_fast2.mp4
├── target_person.jpg
│
├── result.mp4
│
├── SimSwap/
│   ├── models/
│   ├── options/
│   ├── insightface_func/
│   ├── parsing_model/
│   ├── test_video_swapsingle.py
│   └── ...
│
└── README.md

Some input/output media files may be excluded from the Git repository through .gitignore.

Installation

1. Clone the repository

git clone https://github.com/tskz48/FaceSwapping.git
cd FaceSwapping

2. Create a virtual environment

python3 -m venv venv

Activate it on macOS/Linux:

source venv/bin/activate

On Windows:

venv\Scripts\activate

3. Install the backend dependencies

Install FastAPI and Uvicorn:

pip install fastapi uvicorn

The SimSwap portion of the project also requires its own deep learning and video-processing dependencies, including packages such as:

torch
torchvision
opencv-python
pillow
numpy
imageio
moviepy
insightface
timm

Because SimSwap has model- and platform-specific requirements, refer to the upstream SimSwap setup instructions when configuring its pretrained models and dependencies.

SimSwap Model Setup

This project relies on the pretrained models required by SimSwap.

The necessary model files must be placed in the locations expected by the SimSwap code before inference can run successfully.

Upstream project:

https://github.com/neuralchen/SimSwap

Follow the upstream repository's preparation instructions for the required checkpoints and face-recognition models.

Running the Backend

From the main project directory, run the FastAPI application with Uvicorn:

uvicorn Code:app --reload

By default, the backend is available at:

http://127.0.0.1:8000

FastAPI's automatically generated API documentation can normally be viewed at:

http://127.0.0.1:8000/docs

CORS

The FastAPI application can be configured to allow requests from a local frontend development server such as:

http://localhost:5173

This is useful when connecting a React/Vite frontend to the Python backend during development.

Face-Swapping Workflow

1. Select source face image(s)
                ↓
2. Select the target video
                ↓
3. Optionally provide a target-person image
                ↓
4. Send/run the face-swap request
                ↓
5. FastAPI launches the processing pipeline
                ↓
6. Target faces are detected
                ↓
7. Hair segmentation and masking are applied
                ↓
8. Source identity is transferred to the target face
                ↓
9. Masked regions are combined with the processed frame
                ↓
10. Output video is generated
                ↓
11. result.mp4 is returned/served

Example Inputs and Output

Source

source1.jpg

Target

target_fast2.mp4

Optional target-person reference

target_person.jpg

Output

result.mp4

AI Model

The face-swapping functionality is provided by SimSwap: An Efficient Framework for High Fidelity Face Swapping.

SimSwap is designed to perform arbitrary face swapping using a single trained model. In this project, SimSwap is used as the inference engine while the surrounding Python application handles application-level input, execution, and output delivery.

Responsible Use

Face-swapping technology can create realistic synthetic media. Use this project only with appropriate permission and for legitimate educational, research, or development purposes.

Do not use generated content to impersonate people, deceive others, violate privacy, or misrepresent synthetic media as authentic.

Attribution

This project uses SimSwap, an open-source face-swapping framework licensed under CC BY-NC 4.0.

Repository

https://github.com/tskz48/FaceSwapping
