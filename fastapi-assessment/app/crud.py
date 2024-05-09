# app/crud.py
from sqlalchemy.orm import Session
from app.models import User

# Create a new user
def create_user(db: Session, name: str) -> User:
    new_user = User(name=name)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# Retrieve a user by ID
def get_user_by_id(db: Session, user_id: int) -> User:
    return db.query(User).filter(User.id == user_id).first()

# Update a user's name by their ID
def update_user_name(db: Session, user_id: int, new_name: str) -> User:
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.name = new_name
        db.commit()
        db.refresh(user)
    return user

# Search for users by name (case-insensitive)
def search_users_by_name(db: Session, query: str):
    return db.query(User).filter(User.name.ilike(f"%{query}%")).all()
