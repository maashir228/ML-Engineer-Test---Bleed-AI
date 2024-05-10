# ML-Engineer-Test---Bleed-AI
# FastAPI Application with JWT Authorization and MediaPipe Image Processing

This FastAPI application provides basic CRUD operations with JWT-based authorization and includes an endpoint for image processing using MediaPipe for facial detection. The project uses SQLite for database operations and includes secure endpoints with JWT tokens.

## Setup and Installation
To set up the project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/maashir228/ML-Engineer-Test---Bleed-AI.git
   cd ML-Engineer-Test---Bleed-AI

##Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # For Windows, use venv\Scripts\activate

## Install Project Dependencies:
pip install -r requirements.txt

## Run the FastAPI Server:
uvicorn app.main:app --host 0.0.0.0 --port 8000

## Build docker:
docker build -t fastapiassignment .

## Build using docker-compose.yml:
docker-compose up -d

#Overview of API Endpoints
Here is an overview of the key API endpoints provided by the application:

##Authorization Endpoint:

POST /token: This endpoint generates a JWT token for user authentication. Provide username and password in the request body (x-www-form-urlencoded).
User Endpoints:

POST /users/: Create a new user. Requires JWT authorization.
GET /users/{user_id}: Retrieve a user by their ID.
PUT /users/{user_id}: Update a user's name by ID.
GET /search/: Search for users by name.
Image Processing Endpoint:

POST /process-image/: Upload an image to detect and process facial boundaries using MediaPipe.

#Contributing to the Project
Contributions are welcomed to this project. If you'd like to contribute, please follow these guidelines:

Fork the Repository: Create a fork from the main repository to work on your changes.
Create a New Branch: Use a descriptive name for your branch, indicating the feature or bug you're addressing.
Make Your Changes: Ensure your code is formatted with Black and includes appropriate type hints.
Submit a Pull Request: Once you've made your changes, submit a pull request for review.
