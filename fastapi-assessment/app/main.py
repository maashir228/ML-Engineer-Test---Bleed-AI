# app/main.py
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from passlib.context import CryptContext
from typing import List
import os
from app.models import Base, engine, SessionLocal
from app.crud import create_user, get_user_by_id, update_user_name, search_users_by_name
import mediapipe as mp
import cv2
import numpy as np
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import base64
import io
# Initialize the app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your needs
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the database
Base.metadata.create_all(bind=engine)

class UserBase(BaseModel):
    name: str

class UserUpdate(BaseModel):
    name: str

class ImageData(BaseModel):
    file: UploadFile

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def base64_to_image(base64_string, output_path):
    # Decode the base64 string to binary data
    binary_data = base64.b64decode(base64_string)

    # Create PIL image from binary data
    image = Image.open(BytesIO(binary_data))

    # Save the image to a file
    image.save(output_path)

# Endpoints
@app.post("/users/", response_model=int)
def create_new_user(user: UserBase, db: Session = Depends(get_db)):
    user = create_user(db, user.name)
    return user.id

@app.get("/users/{user_id}")
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": user.id, "name": user.name}

@app.put("/users/{user_id}")
def update_user(user_id: int, user: UserUpdate, db: Session = Depends(get_db)):
    user_in_db = get_user_by_id(db, user_id)
    if not user_in_db:
        raise HTTPException(status_code=404, detail="User not found")
    updated_user = update_user_name(db, user_id, user.name)
    return {"id": updated_user.id, "name": updated_user.name}

@app.get("/search/")
def search_users(query: str, db: Session = Depends(get_db)):
    results = search_users_by_name(db, query)
    return [{"id": user.id, "name": user.name} for user in results]

# Image Processing with MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

@app.post("/process-image/")
async def process_image(File: UploadFile = File(...)):
    # Read the image file
    image_file = await File.read()
    print(image_file)
    # Convert the image file to a format that MediaPipe can process
    image = Image.open(BytesIO(image_file))
    image = np.array(image)

    # Use MediaPipe to detect faces in the image
    with mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # If a face is detected, get the facial boundaries and landmarks
    if results.detections:
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(box.xmin * iw), int(box.ymin * ih), int(box.width * iw), int(box.height * ih)

            # Crop the image
            cropped_image = image[y:y+h, x:x+w]

            # Convert the cropped image to a base64 string
            pil_img = Image.fromarray(cropped_image)
            byte_arr = io.BytesIO()
            pil_img.save(byte_arr, format='PNG')
            byte_arr = byte_arr.getvalue()
            base64_str = base64.b64encode(byte_arr).decode()

            # Extract the landmarks
            landmarks = []
            for landmark in detection.location_data.relative_keypoints:
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y
                })
            base64_to_image(base64_str, "cropped_image.png")
            # Return the facial boundaries, landmarks, and cropped image in the response
            return {"box": {"x": x, "y": y, "width": w, "height": h}, "landmarks": landmarks, "cropped_image": base64_str}

    return {"error": "No face detected"}