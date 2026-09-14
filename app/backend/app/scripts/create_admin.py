import os
import sys

# Add the apps/api directory to sys.path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.db.database import SessionLocal, engine, Base
from app.db.models import User
from app.core.auth import get_password_hash

import uuid

def create_admin(username, password):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    
    user = db.query(User).filter(User.username == username).first()
    if user:
        print(f"User '{username}' already exists.")
        db.close()
        return

    hashed_password = get_password_hash(password)
    admin_user = User(
        id=str(uuid.uuid4()),
        username=username,
        hashed_password=hashed_password,
        role="admin"
    )
    
    db.add(admin_user)
    db.commit()
    db.close()
    print(f"Successfully created admin user: {username}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_admin.py <username> <password>")
        sys.exit(1)
        
    username = sys.argv[1]
    password = sys.argv[2]
    create_admin(username, password)
