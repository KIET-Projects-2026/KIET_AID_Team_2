# ===================== AUTHENTICATION MODULE =====================
# User authentication and management system with MongoDB

import os
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict
from pathlib import Path
from pydantic import BaseModel, EmailStr, Field
from fastapi import HTTPException, status
from pymongo.errors import DuplicateKeyError

# Import MongoDB connection
try:
    from database import get_users_collection, get_sessions_collection
    USE_MONGODB = True
except ImportError:
    USE_MONGODB = False
    print("⚠️ MongoDB not available, using JSON file storage")

# ===================== CONFIGURATION =====================
USERS_DB_FILE = os.path.join(os.path.dirname(__file__), "users.json")
SESSIONS_DB_FILE = os.path.join(os.path.dirname(__file__), "sessions.json")
SESSION_EXPIRY_DAYS = 30

# ===================== PYDANTIC MODELS =====================

class UserCreate(BaseModel):
    """User registration model"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None
    phone: str = Field(..., min_length=5, max_length=20)
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(...)
    allergies: str = Field(...)
    emergencyContact: str = Field(...)

class UserLogin(BaseModel):
    """User login model"""
    username: str
    password: str

class UserResponse(BaseModel):
    """User response model"""
    user_id: str
    username: str
    email: str
    full_name: Optional[str]
    phone: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    allergies: Optional[str] = None
    emergencyContact: Optional[str] = None
    emergencyEmail: Optional[str] = None
    created_at: str
    updated_at: Optional[str] = None

class AuthResponse(BaseModel):
    """Authentication response model"""
    token: str
    user: UserResponse
    message: str

# ===================== DATABASE MANAGEMENT =====================

class Database:
    """Dual storage: MongoDB (primary) with JSON file fallback"""
    
    @staticmethod
    def load_users() -> Dict:
        """Load users from MongoDB (DB-only). Raises if DB unavailable."""
        if USE_MONGODB:
            try:
                users_collection = get_users_collection()
                users_list = list(users_collection.find({}))
                # Convert MongoDB documents to dict with user_id as key
                return {user['user_id']: user for user in users_list}
            except Exception as e:
                # Fail fast - do not fall back to JSON when DB is intended to be the source of truth
                raise RuntimeError(f"🛑 MongoDB error loading users: {e}")
        
        # If MongoDB not used, fall back to JSON
        if not os.path.exists(USERS_DB_FILE):
            return {}
        try:
            with open(USERS_DB_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading users JSON: {e}")
    
    @staticmethod
    def save_users(users: Dict):
        """Save users to MongoDB (and only JSON if MongoDB is not enabled)"""
        if USE_MONGODB:
            try:
                users_collection = get_users_collection()
                for user_id, user_data in users.items():
                    users_collection.update_one(
                        {'user_id': user_id},
                        {'$set': user_data},
                        upsert=True
                    )
                return
            except Exception as e:
                # Fail fast - do not silently write to JSON
                raise RuntimeError(f"🛑 MongoDB error saving users: {e}")
        
        # If MongoDB not used, persist to JSON
        try:
            with open(USERS_DB_FILE, 'w') as f:
                json.dump(users, f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Error saving users JSON: {e}")
    
    @staticmethod
    def load_sessions() -> Dict:
        """Load sessions from MongoDB (DB-only). Raises if DB unavailable."""
        if USE_MONGODB:
            try:
                sessions_collection = get_sessions_collection()
                sessions_list = list(sessions_collection.find({}))
                # Convert MongoDB documents to dict with token as key
                return {session['token']: session for session in sessions_list}
            except Exception as e:
                raise RuntimeError(f"🛑 MongoDB error loading sessions: {e}")
        
        # Fallback to JSON
        if not os.path.exists(SESSIONS_DB_FILE):
            return {}
        try:
            with open(SESSIONS_DB_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading sessions JSON: {e}")
    
    @staticmethod
    def save_sessions(sessions: Dict):
        """Save sessions to MongoDB (and only JSON if MongoDB is not enabled)"""
        if USE_MONGODB:
            try:
                sessions_collection = get_sessions_collection()
                for token, session_data in sessions.items():
                    session_data['token'] = token  # Ensure token is in the document
                    sessions_collection.update_one(
                        {'token': token},
                        {'$set': session_data},
                        upsert=True
                    )
                return
            except Exception as e:
                raise RuntimeError(f"🛑 MongoDB error saving sessions: {e}")

        # If MongoDB not used, persist to JSON
        try:
            with open(SESSIONS_DB_FILE, 'w') as f:
                json.dump(sessions, f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Error saving sessions JSON: {e}")
    
    @staticmethod
    def delete_session(token: str):
        """Delete a session from MongoDB (DB-only)"""
        if USE_MONGODB:
            try:
                sessions_collection = get_sessions_collection()
                sessions_collection.delete_one({'token': token})
                return
            except Exception as e:
                raise RuntimeError(f"🛑 MongoDB error deleting session: {e}")

        # If MongoDB not used, delete from JSON
        sessions = Database.load_sessions()
        if token in sessions:
            del sessions[token]
            try:
                with open(SESSIONS_DB_FILE, 'w') as f:
                    json.dump(sessions, f, indent=2)
            except Exception as e:
                raise RuntimeError(f"Error saving sessions JSON: {e}")

# ===================== AUTHENTICATION MANAGER =====================

class AuthManager:
    """Manages user authentication and sessions"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def generate_token() -> str:
        """Generate secure random token"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def create_user(user_data: UserCreate) -> UserResponse:
        """Create new user account"""
        users = Database.load_users()
        
        # Check if username already exists
        if any(u['username'].lower() == user_data.username.lower() for u in users.values()):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        # Check if email already exists
        if any(u['email'].lower() == user_data.email.lower() for u in users.values()):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already exists"
            )
        
        # Create user
        user_id = secrets.token_hex(16)
        user = {
            'user_id': user_id,
            'username': user_data.username,
            'email': user_data.email,
            'password_hash': AuthManager.hash_password(user_data.password),
            'full_name': user_data.full_name,
            'phone': user_data.phone,
            'age': user_data.age,
            'gender': user_data.gender,
            'allergies': user_data.allergies,
            'emergencyContact': user_data.emergencyContact,
            'emergencyEmail': user_data.emergencyEmail,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        users[user_id] = user
        Database.save_users(users)
        
        return UserResponse(
            user_id=user['user_id'],
            username=user['username'],
            email=user['email'],
            full_name=user.get('full_name'),
            phone=user.get('phone'),
            age=user.get('age'),
            gender=user.get('gender'),
            allergies=user.get('allergies'),
            emergencyContact=user.get('emergencyContact'),
            emergencyEmail=user.get('emergencyEmail'),
            created_at=user['created_at'],
            updated_at=user.get('updated_at')
        )
    
    @staticmethod
    def authenticate_user(login_data: UserLogin) -> tuple[str, UserResponse]:
        """Authenticate user and create session"""
        users = Database.load_users()
        
        # Find user
        user = None
        for u in users.values():
            if u['username'].lower() == login_data.username.lower():
                user = u
                break
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Verify password
        password_hash = AuthManager.hash_password(login_data.password)
        if user['password_hash'] != password_hash:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Create session
        token = AuthManager.generate_token()
        sessions = Database.load_sessions()
        
        session = {
            'user_id': user['user_id'],
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(days=SESSION_EXPIRY_DAYS)).isoformat()
        }
        
        sessions[token] = session
        Database.save_sessions(sessions)
        
        user_response = UserResponse(
            user_id=user['user_id'],
            username=user['username'],
            email=user['email'],
            full_name=user.get('full_name'),
            phone=user.get('phone'),
            age=user.get('age'),
            gender=user.get('gender'),
            allergies=user.get('allergies'),
            emergencyContact=user.get('emergencyContact'),
            emergencyEmail=user.get('emergencyEmail'),
            created_at=user['created_at'],
            updated_at=user.get('updated_at')
        )
        
        return token, user_response
    
    @staticmethod
    def verify_token(token: str) -> Optional[str]:
        """Verify token and return user_id if valid"""
        sessions = Database.load_sessions()
        
        session = sessions.get(token)
        if not session:
            return None
        
        # Check expiry
        expires_at = datetime.fromisoformat(session['expires_at'])
        if datetime.now() > expires_at:
            # Delete expired session
            del sessions[token]
            Database.save_sessions(sessions)
            return None
        
        return session['user_id']
    
    @staticmethod
    def get_user_by_id(user_id: str) -> Optional[UserResponse]:
        """Get user by ID"""
        users = Database.load_users()
        user = users.get(user_id)
        
        if not user:
            return None
        
        return UserResponse(
            user_id=user['user_id'],
            username=user['username'],
            email=user['email'],
            full_name=user.get('full_name'),
            phone=user.get('phone'),
            age=user.get('age'),
            gender=user.get('gender'),
            allergies=user.get('allergies'),
            emergencyContact=user.get('emergencyContact'),
            emergencyEmail=user.get('emergencyEmail'),
            created_at=user['created_at'],
            updated_at=user.get('updated_at')
        )
    
    @staticmethod
    def logout(token: str):
        """Logout user by removing session"""
        Database.delete_session(token)

auth_manager = AuthManager()
