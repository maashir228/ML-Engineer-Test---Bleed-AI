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

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Endpoints
@app.post("/users/", response_model=int)
def create_new_user(name: str, db: Session = Depends(get_db)):
    user = create_user(db, name)
    return user.id

@app.get("/users/{user_id}")
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": user.id, "name": user.name}

@app.put("/users/{user_id}")
def update_user(user_id: int, name: str, db: Session = Depends(get_db)):
    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    updated_user = update_user_name(db, user_id, name)
    return {"id": updated_user.id, "name": updated_user.name}

@app.get("/search/")
def search_users(query: str, db: Session = Depends(get_db)):
    results = search_users_by_name(db, query)
    return [{"id": user.id, "name": user.name} for user in results]

# Image Processing with MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

@app.post("/process-image/")
def process_image(file: UploadFile = File(...)):
    image_data = file.file.read()
    np_image = np.frombuffer(image_data, dtype=np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(img)

        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box

            # Calculate the bounding box
            h, w, _ = img.shape
            x_min = int(bbox.xmin * w)
            y_min = int(bbox.ymin * h)
            x_max = int((bbox.xmin + bbox.width) * w)
            y_max = int((bbox.ymin + bbox.height) * h)

            # Crop the face
            face_crop = img[y_min:y_max, x_min:x_max]

            # Draw landmarks
            for detection in results.detections:
                mp_drawing.draw_detection(img, detection)

            # Encode the image to be returned
            _, encoded_img = cv2.imencode('.jpg', face_crop)
            encoded_landmarks, _ = cv2.imencode('.jpg', img)

            return {
                "cropped_image": encoded_img.tobytes(),
                "landmarked_image": encoded_landmarks.tobytes(),
            }

    raise HTTPException(status_code=404, detail="No face detected")
