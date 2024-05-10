FROM python:3.12

# Copy the entire app directory into the container
COPY ./app /app

RUN pip install fastapi uvicorn sqlalchemy mediapipe pydantic python-jose passlib 

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "15400"]