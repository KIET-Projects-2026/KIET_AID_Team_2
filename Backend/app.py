# ===================== HEALTHCARE CHATBOT - COMPLETE FASTAPI BACKEND =====================
# Production-ready FastAPI backend with full voice and text support
# All functionalities included: REST endpoints, WebSocket, audio processing, logging, caching

# ===================== 1. REQUIREMENTS =====================
# pip install fastapi uvicorn torch transformers peft accelerate python-multipart
# pip install SpeechRecognition pydub librosa numpy scipy
# pip install python-dotenv

# ===================== 2. IMPORTS =====================
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, BackgroundTasks, Body, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
class ProfileUpdateRequest(BaseModel):
    full_name: str = None
    age: int = None
    gender: str = None
    allergies: str = None
    emergencyEmail: str = None

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import os
import json
from pathlib import Path
import asyncio
from typing import Optional, List, Dict
import io
import speech_recognition as sr
import hashlib
import secrets
import uuid
from fastapi import Form, Depends, Header, status
from pydub import AudioSegment
import logging
from datetime import datetime, timedelta
import time
from functools import wraps
import numpy as np

# Import authentication module
from auth import auth_manager, UserCreate, UserLogin, AuthResponse, UserResponse

# ===================== 3. LOGGING SETUP =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================== 4. INITIALIZE FASTAPI =====================
app = FastAPI(
    title="Healthcare Chatbot API with Voice Support",
    description="AI-powered healthcare chatbot with complete voice and text support",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ===================== 5. CORS CONFIGURATION =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific URLs in production: ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== 6. CONFIGURATION =====================
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "model"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25 MB
SUPPORTED_AUDIO_FORMATS = ['wav', 'mp3', 'flac', 'ogg', 'webm']
REQUEST_TIMEOUT = 30

logger.info(f"🚀 Configuration: Device={DEVICE}, Model Path={MODEL_PATH}")

# ===================== 7. PYDANTIC MODELS =====================

class TextInput(BaseModel):
    """Request model for text input"""
    text: str = Field(..., min_length=1, max_length=1000)
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None

class VoiceResponse(BaseModel):
    """Response model for voice/text input"""
    status: str
    input_type: str
    input_text: str
    response: str
    confidence: float = 0.95
    timestamp: Optional[str] = None
    processing_time: Optional[float] = None
    conversation_id: Optional[str] = None

class ErrorResponse(BaseModel):
    """Error response model"""
    status: str = "error"
    error: str
    message: str
    timestamp: Optional[str] = None

class BatchRequest(BaseModel):
    """Batch processing request"""
    requests: List[TextInput]

class ChatHistory(BaseModel):
    """Chat history item"""
    user_input: str
    bot_response: str
    input_type: str
    timestamp: str

# ===================== 8. MODEL MANAGER =====================

class ModelManager:
    """Manages model loading and inference with error handling"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = DEVICE
        self.is_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load trained model and tokenizer with fallback options"""
        try:
            logger.info("🔄 Loading model...")
            
            # Try to load from checkpoint first
            if os.path.exists(MODEL_PATH):
                logger.info(f"📂 Loading from checkpoint: {MODEL_PATH}")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
                    self.is_loaded = True
                    logger.info("✅ Model loaded successfully from checkpoint")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load from checkpoint: {e}")
                    self._load_base_model()
            else:
                logger.warning(f"⚠️ Model path not found: {MODEL_PATH}")
                self._load_base_model()
            
            if self.model is not None:
                self.model = self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True
                logger.info(f"✅ Model ready on device: {self.device}")
        
        except Exception as e:
            logger.error(f"❌ Critical error loading model: {e}")
            self.is_loaded = False
            raise
    
    def _load_base_model(self):
        """Fallback: Load base model"""
        logger.info("🔄 Loading base FLAN-T5 model...")
        base_model = "google/flan-t5-base"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
            logger.info("✅ Base model loaded (not fine-tuned)")
        except Exception as e:
            logger.error(f"❌ Failed to load base model: {e}")
            raise
    
    def generate_response(
        self,
        input_text: str,
        max_length: int = 256,
        temperature: float = 0.7,
        num_beams: int = 4,
        top_p: float = 0.9
    ) -> str:
        """
        Generate response for given input text
        
        Args:
            input_text: User's input question
            max_length: Maximum response length
            temperature: Sampling temperature (0.0-1.0)
            num_beams: Beam search size
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated response string
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            start_time = time.time()
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    num_beams=num_beams,
                    do_sample=True,
                    top_p=top_p,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=2
                )
            
            # Decode output
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            processing_time = time.time() - start_time
            logger.info(f"⏱️ Model inference took {processing_time:.4f}s")
            
            return response, processing_time
        
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            raise

# Initialize model manager
try:
    model_manager = ModelManager()
except Exception as e:
    logger.error(f"❌ Failed to initialize model manager: {e}")
    model_manager = None

# ===================== 9. AUDIO PROCESSOR =====================

class AudioProcessor:
    """Handles audio conversion and speech recognition"""
    
    SUPPORTED_FORMATS = ['wav', 'mp3', 'flac', 'ogg', 'webm']
    MAX_SIZE = 25 * 1024 * 1024  # 25 MB
    
    @staticmethod
    def validate_audio(filename: str, file_size: int) -> tuple[bool, str]:
        """Validate audio file"""
        ext = filename.split('.')[-1].lower()
        
        if ext not in AudioProcessor.SUPPORTED_FORMATS:
            return False, f"Unsupported format: {ext}. Supported: {', '.join(AudioProcessor.SUPPORTED_FORMATS)}"
        
        if file_size > AudioProcessor.MAX_SIZE:
            return False, f"File too large: {file_size / 1024 / 1024:.1f}MB. Max: {AudioProcessor.MAX_SIZE / 1024 / 1024:.0f}MB"
        
        return True, ""
    
    @staticmethod
    def transcribe_audio(audio_bytes: bytes, audio_format: str = "wav") -> tuple[str, float]:
        """
        Transcribe audio bytes to text using SpeechRecognition
        
        Args:
            audio_bytes: Audio file content as bytes
            audio_format: Audio format (wav, mp3, etc.)
        
        Returns:
            Tuple of (transcribed text, confidence)
        """
        try:
            start_time = time.time()
            
            logger.info(f"🎤 Transcribing audio ({audio_format})...")
            
            # Load audio from bytes
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
            
            # Convert to WAV format for speech recognition
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
            wav_io.seek(0)
            
            # Use SpeechRecognition
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_io) as source:
                audio_data = recognizer.record(source)
            
            # Try Google Speech Recognition API
            text = recognizer.recognize_google(audio_data)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Transcribed: '{text}' (took {processing_time:.2f}s)")
            
            return text, 0.95  # Google API confidence
        
        except sr.UnknownValueError:
            logger.warning("⚠️ Could not understand audio")
            return "", 0.0
        except sr.RequestError as e:
            logger.error(f"❌ Speech recognition service error: {e}")
            return "", 0.0
        except Exception as e:
            logger.error(f"❌ Error transcribing audio: {e}")
            return "", 0.0

audio_processor = AudioProcessor()

# ===================== 10. CACHE MANAGER =====================

class CacheManager:
    """Simple response caching"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[str]:
        """Get cached response"""
        return self.cache.get(key)
    
    def set(self, key: str, value: str):
        """Cache response"""
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.info("🗑️ Cache full, removed oldest entry")
        
        self.cache[key] = value
        logger.info(f"💾 Cached response for: {key[:50]}")

cache_manager = CacheManager()

# ===================== 11. STATISTICS TRACKER =====================

class Statistics:
    """Track API statistics"""
    
    def __init__(self):
        self.total_requests = 0
        self.text_requests = 0
        self.voice_requests = 0
        self.errors = 0
        self.avg_response_time = 0
        self.start_time = datetime.now()
    
    def record_request(self, request_type: str, response_time: float, error: bool = False):
        """Record API request"""
        self.total_requests += 1
        
        if request_type == "text":
            self.text_requests += 1
        elif request_type == "voice":
            self.voice_requests += 1
        
        if error:
            self.errors += 1
        
        # Update average response time
        self.avg_response_time = (
            (self.avg_response_time * (self.total_requests - 1) + response_time) / 
            self.total_requests
        )
    
    def get_stats(self) -> dict:
        """Get current statistics"""
        uptime = datetime.now() - self.start_time
        return {
            "total_requests": self.total_requests,
            "text_requests": self.text_requests,
            "voice_requests": self.voice_requests,
            "errors": self.errors,
            "avg_response_time": f"{self.avg_response_time:.4f}s",
            "uptime": str(uptime).split('.')[0],
            "cache_size": len(cache_manager.cache)
        }

statistics = Statistics()

# ===================== 12. CHAT LOGGER =====================

class ChatLogger:
    """Chat logger that stores conversation history in MongoDB (DB-only). Uses lazy initialization so startup can connect to DB during on_startup."""
    def __init__(self):
        self.chat_logs = None

    def _ensure_connected(self):
        try:
            from database import get_chat_logs_collection, mongodb, init_mongodb
            if not mongodb.is_connected:
                # attempt to initialize connection (useful during startup)
                init_mongodb()
            if not mongodb.is_connected:
                raise RuntimeError("MongoDB is not connected. Chat history requires MongoDB.")
            if self.chat_logs is None:
                self.chat_logs = get_chat_logs_collection()
                logger.info("✅ ChatLogger connected to MongoDB")
        except Exception as e:
            logger.error(f"❌ ChatLogger failed to connect: {e}")
            raise

    def add_log(self, user_input: str, bot_response: str, input_type: str = "text", user_id: Optional[str] = None, conversation_id: Optional[str] = None, **metadata):
        """Add conversation to MongoDB log. If no conversation_id is provided, generate one and return it."""
        if user_id:
            metadata['user_id'] = user_id
        # Use provided conversation_id or generate a new one
        conv_id = conversation_id or metadata.get('conversation_id') or str(uuid.uuid4())
        metadata['conversation_id'] = conv_id
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "bot_response": bot_response,
            "input_type": input_type,
            "user_id": user_id,
            "conversation_id": conv_id,
            "metadata": metadata
        }
        try:
            self._ensure_connected()
            result = self.chat_logs.insert_one(log_entry)
            logger.debug(f"💾 Chat log saved to MongoDB: {result.inserted_id} (conv={conv_id})")
            return conv_id
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Error saving chat log to MongoDB: {e}")
            raise HTTPException(status_code=500, detail="Failed to save chat log")

    def get_user_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get chat history for a specific user from MongoDB"""
        try:
            self._ensure_connected()
            cursor = self.chat_logs.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)
            history = list(cursor)
            for item in history:
                if '_id' in item:
                    item['_id'] = str(item['_id'])
            return history
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Error getting user history from MongoDB: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch chat history")

chat_logger = ChatLogger()

# ===================== 12.5 USERS & AUTHENTICATION =====================

class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    email: Optional[str] = None
    full_name: Optional[str] = None
    phone: str = Field(..., min_length=5, max_length=20)
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(...)
    allergies: str = Field(...)
    emergencyContact: str = Field(...)

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)

class UserInfo(BaseModel):
    id: str
    username: str
    email: Optional[str] = None

class AuthResponse(BaseModel):
    status: str
    token: Optional[str] = None
    user: Optional[UserInfo] = None
    message: Optional[str] = None

class MongoDBUsersManager:
    """MongoDB-backed user manager (DB-only) with lazy connection"""
    def __init__(self):
        # Lazy initialization - collections set on first use
        self.users_collection = None
        self.sessions_collection = None

    def _ensure_connected(self):
        try:
            from database import get_users_collection, get_sessions_collection, mongodb, init_mongodb
            if not mongodb.is_connected:
                # try to initialize connection (startup might not have finished yet)
                init_mongodb()
            if not mongodb.is_connected:
                raise RuntimeError("MongoDB is not connected. This backend requires MongoDB for user storage.")
            if self.users_collection is None or self.sessions_collection is None:
                self.users_collection = get_users_collection()
                self.sessions_collection = get_sessions_collection()
                logger.info("✅ UsersManager connected to MongoDB")
        except Exception as e:
            logger.error(f"❌ UsersManager failed to connect: {e}")
            raise

    def _hash_password(self, password: str, salt: str) -> str:
        hashed = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return hashed.hex()

    def _hash_password(self, password: str, salt: str) -> str:
        hashed = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return hashed.hex()

    def create_user(self, username: str, password: str, email: Optional[str] = None, full_name: Optional[str] = None, phone: str = '', age: int = 0, gender: str = '', allergies: str = '', emergencyContact: str = '') -> dict:
        """Create new user in MongoDB (DB-only). Ensures `id` field and avoids inserting null email. Extended for extra fields."""
        try:
            self._ensure_connected()
            # Check if username or email already exists (case-insensitive)
            if self.users_collection.find_one({'username': {'$regex': f'^{username}$', '$options': 'i'}}):
                raise ValueError('Username already exists')
            if email and self.users_collection.find_one({'email': {'$regex': f'^{email}$', '$options': 'i'}}):
                raise ValueError('Email already exists')

            user_id = str(uuid.uuid4())
            salt = secrets.token_hex(16)
            password_hash = self._hash_password(password, salt)

            # Build user document - omit `email` entirely if not provided to avoid unique-null conflicts
            user = {
                'user_id': user_id,
                'id': user_id,
                'username': username,
                'full_name': full_name,
                'password_hash': password_hash,
                'salt': salt,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'phone': phone,
                'age': age,
                'gender': gender,
                'allergies': allergies,
                'emergencyContact': emergencyContact
            }
            if email:
                user['email'] = email

            # Insert with DuplicateKeyError handling for safety
            try:
                from pymongo.errors import DuplicateKeyError
                self.users_collection.insert_one(user)
            except DuplicateKeyError as dk:
                msg = str(dk).lower()
                # If the duplicate key references email, it's often due to a non-partial unique index
                # on the `email` field that treats missing emails as duplicates. Attempt an automatic
                # repair (recreate partial/sparse index) and retry once before giving up.
                if 'email' in msg:
                    try:
                        from database import init_mongodb
                        logger.warning("⚠️ DuplicateKeyError on email detected. Attempting to repair email index and retry insert...")
                        # Re-initialize MongoDB indexes (this will drop and recreate email index if needed)
                        init_mongodb()
                        # Refresh collection handle in case indexes/connection changed
                        self.users_collection = get_users_collection()
                        # Retry insert once
                        try:
                            self.users_collection.insert_one(user)
                            logger.info("✅ Insert succeeded after index repair")
                        except DuplicateKeyError:
                            # Still failing: surface a friendly error
                            raise ValueError('Email already exists')
                        return user
                    except Exception as e:
                        logger.error(f"❌ Failed to auto-repair email index: {e}")
                        raise ValueError('Email already exists')
                if 'username' in msg:
                    raise ValueError('Username already exists')
                raise

            logger.info(f"✅ Created user in MongoDB: {username} ({user_id})")
            logger.info(f"Returning user from create_user: {user}")
            return user

        except ValueError:
            # propagate validation errors
            raise
        except Exception as e:
            logger.error(f"❌ MongoDB error creating user: {e}")
            raise HTTPException(status_code=500, detail="Failed to create user")

    def verify_user(self, username: str, password: str) -> Optional[dict]:
        """Verify user credentials from MongoDB (DB-only)"""
        try:
            self._ensure_connected()
            user = self.users_collection.find_one({'username': {'$regex': f'^{username}$', '$options': 'i'}})
            if not user:
                return None
            expected = user['password_hash']
            derived = self._hash_password(password, user['salt'])
            if secrets.compare_digest(derived, expected):
                # Ensure 'id' field exists for compatibility with existing code
                if 'id' not in user:
                    user['id'] = user.get('user_id', str(user.get('_id', '')))
                return user
            return None
        except Exception as e:
            logger.error(f"❌ MongoDB error verifying user: {e}")
            raise HTTPException(status_code=500, detail="Failed to verify user")

    def create_token(self, user_id: str) -> str:
        """Create authentication token in MongoDB (DB-only)"""
        token = secrets.token_urlsafe(32)
        try:
            self._ensure_connected()
            session = {
                'token': token,
                'user_id': user_id,
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(days=30)).isoformat()
            }
            self.sessions_collection.insert_one(session)
            logger.info(f"✅ Created token in MongoDB for user: {user_id}")
            return token
        except Exception as e:
            logger.error(f"❌ MongoDB error creating token: {e}")
            raise HTTPException(status_code=500, detail="Failed to create session token")

    def revoke_token(self, token: str):
        """Revoke authentication token from MongoDB (DB-only)"""
        try:
            self._ensure_connected()
            self.sessions_collection.delete_one({'token': token})
            logger.info(f"✅ Revoked token in MongoDB")
        except Exception as e:
            logger.error(f"❌ MongoDB error revoking token: {e}")
            raise HTTPException(status_code=500, detail="Failed to revoke token")

    def get_user_by_token(self, token: str) -> Optional[dict]:
        """Get user by authentication token from MongoDB (DB-only)"""
        try:
            self._ensure_connected()
            session = self.sessions_collection.find_one({'token': token})
            if not session:
                return None
            # Check expiry
            expires_at = datetime.fromisoformat(session['expires_at'])
            if datetime.now() > expires_at:
                self.sessions_collection.delete_one({'token': token})
                return None
            user = self.users_collection.find_one({'user_id': session['user_id']})
            if user and 'id' not in user:
                user['id'] = user.get('user_id', str(user.get('_id', '')))
            # Ensure emergencyEmail is present (for legacy users)
            if user is not None and 'emergencyEmail' not in user:
                user['emergencyEmail'] = user.get('emergency_email', None)
            return user
        except Exception as e:
            logger.error(f"❌ MongoDB error getting user by token: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve user by token")

    def get_user_by_id(self, user_id: str) -> Optional[dict]:
        """Get user by ID from MongoDB (DB-only)"""
        try:
            self._ensure_connected()
            user = self.users_collection.find_one({'user_id': user_id})
            if user and 'id' not in user:
                user['id'] = user.get('user_id', str(user.get('_id', '')))
            # Ensure emergencyEmail is present (for legacy users)
            if user is not None and 'emergencyEmail' not in user:
                user['emergencyEmail'] = user.get('emergency_email', None)
            return user
        except Exception as e:
            logger.error(f"❌ MongoDB error getting user by ID: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve user by id")

users_manager = MongoDBUsersManager()

# Dependency to get current user from Authorization header (Bearer <token>)
async def get_current_user(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    if not authorization:
        return None
    if not authorization.startswith('Bearer '):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid authorization header')
    token = authorization.split(' ', 1)[1]
    user = users_manager.get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid or expired token')
    user['_token'] = token
    return user

# ===================== 13. API ENDPOINTS =====================

# PATCH endpoint to update user profile (except username, email, phone, emergencyContact)
@app.patch('/api/auth/me')
async def update_profile(
    data: ProfileUpdateRequest = Body(...),
    current_user: dict = Depends(get_current_user)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    # Only allow editing specific fields
    update_fields = {}
    for field in ['full_name', 'age', 'gender', 'allergies', 'emergencyEmail']:
        value = getattr(data, field, None)
        if value is not None:
            update_fields[field] = value
    if not update_fields:
        raise HTTPException(status_code=400, detail="No valid fields to update")
    # Update in DB (MongoDB or JSON)
    user_id = current_user.get('user_id') or current_user.get('id')
    # MongoDB
    if hasattr(users_manager, 'users_collection'):
        users_manager.users_collection.update_one(
            {'user_id': user_id},
            {'$set': update_fields}
        )
        user = users_manager.users_collection.find_one({'user_id': user_id})
    else:
        # JSON fallback
        from auth import Database
        users = Database.load_users()
        user = users.get(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        user.update(update_fields)
        from auth import Database
        Database.save_users(users)
    # Return updated user profile (same as /api/auth/me)
    user_profile = {
        "user_id": user.get("user_id"),
        "username": user.get("username"),
        "email": user.get("email"),
        "full_name": user.get("full_name"),
        "phone": user.get("phone"),
        "age": user.get("age"),
        "gender": user.get("gender"),
        "allergies": user.get("allergies"),
        "emergencyContact": user.get("emergencyContact"),
        "emergencyEmail": user.get("emergencyEmail"),
        "created_at": user.get("created_at"),
        "updated_at": user.get("updated_at")
    }
    return {"status": "success", "user": user_profile}

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "status": "active",
        "name": "Healthcare Chatbot API",
        "version": "2.0.0",
        "device": DEVICE,
        "model_loaded": model_manager.is_loaded if model_manager else False,
        "endpoints": {
            "text": "/api/chat/text",
            "voice": "/api/chat/voice",
            "batch": "/api/chat/batch",
            "health": "/api/health",
            "stats": "/api/stats",
            "logs": "/api/logs",
            "ws": "/ws/chat"
        },
        "docs": "/docs"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint with detailed status including MongoDB connection"""
    mongodb_status = {"connected": False, "message": "Not configured"}
    
    try:
        from database import mongodb
        mongodb_status = mongodb.check_connection()
    except Exception as e:
        mongodb_status = {"connected": False, "message": f"Error: {str(e)}"}
    
    return {
        "status": "healthy" if model_manager.is_loaded else "degraded",
        "device": DEVICE,
        "model_loaded": model_manager.is_loaded if model_manager else False,
        "mongodb": mongodb_status,
        "timestamp": datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }

@app.get("/api/stats")
async def get_stats():
    """Get API statistics"""
    stats = statistics.get_stats()
    return {
        "status": "success",
        "data": stats,
        "timestamp": datetime.now().isoformat()
    }

# ===================== AUTH ENDPOINTS =====================

@app.post('/api/auth/signup', response_model=AuthResponse)
async def signup(payload: SignupRequest):
    """Create a new user account"""
    try:
        user = users_manager.create_user(
            payload.username,
            payload.password,
            email=payload.email,
            full_name=payload.full_name,
            phone=payload.phone,
            age=payload.age,
            gender=payload.gender,
            allergies=payload.allergies,
            emergencyContact=payload.emergencyContact
        )
        logger.debug(f"Created user object: {user}")
        if 'id' not in user:
            logger.error(f"Signup returned user without 'id': {user}")
            raise HTTPException(status_code=500, detail="Internal error: created user missing id")
        token = users_manager.create_token(user['id'])
        return AuthResponse(
            status='success', 
            token=token,
            user=UserInfo(id=user['id'], username=user['username'], email=user.get('email')),
            message='Account created successfully'
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/auth/login', response_model=AuthResponse)
async def login(payload: LoginRequest):
    """Authenticate user and return token"""
    try:
        user = users_manager.verify_user(payload.username, payload.password)
        if not user:
            raise HTTPException(status_code=401, detail='Invalid username or password')
        token = users_manager.create_token(user['id'])
        return AuthResponse(
            status='success', 
            token=token,
            user=UserInfo(id=user['id'], username=user['username']),
            message='Login successful'
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/auth/exists')
async def username_exists(username: Optional[str] = None):
    """Check if a username exists (case-insensitive). Returns {"exists": bool}."""
    if not username or not username.strip():
        raise HTTPException(status_code=400, detail='`username` query parameter is required')
    try:
        # Use users_manager to fetch by username
        # users_manager.verify_user expects a password; so query DB directly using users_manager helper
        user = users_manager.get_user_by_id(username) if False else None  # placeholder to satisfy linters
        # Query MongoDB directly for availability
        users_manager._ensure_connected()
        doc = users_manager.users_collection.find_one({'username': {'$regex': f'^{username}$', '$options': 'i'}})
        exists = doc is not None
        return {"exists": exists}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking username existence: {e}")
        raise HTTPException(status_code=500, detail='Failed to check username')

@app.post('/api/auth/logout')
async def logout(authorization: Optional[str] = Header(None)):
    """Invalidate a user's token"""
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing or invalid Authorization header')
    token = authorization.split(' ', 1)[1]
    users_manager.revoke_token(token)
    return {"status": "success", "message": "Logged out"}

@app.get('/api/auth/me')
async def me(current_user: dict = Depends(get_current_user)):
    """Get current user full profile info"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    # Return all relevant fields for the profile page
    user_profile = {
        "user_id": current_user.get("user_id"),
        "username": current_user.get("username"),
        "email": current_user.get("email"),
        "full_name": current_user.get("full_name"),
        "phone": current_user.get("phone"),
        "age": current_user.get("age"),
        "gender": current_user.get("gender"),
        "allergies": current_user.get("allergies"),
        "emergencyContact": current_user.get("emergencyContact"),
        "emergencyEmail": current_user.get("emergencyEmail"),
        "created_at": current_user.get("created_at"),
        "updated_at": current_user.get("updated_at")
    }
    return {"status": "success", "user": user_profile}
@app.post("/api/chat/text", response_model=VoiceResponse)
async def text_chat(request: TextInput, background_tasks: BackgroundTasks, current_user: Optional[dict] = Depends(get_current_user)):
    """
    Process text input and return AI response
    
    Request body:
    {
        "text": "What are symptoms of diabetes?",
        "user_id": "optional_user_id"
    }
    """
    start_time = time.time()
    
    try:
        # Validate input
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        input_text = request.text.strip()
        
        # Check cache
        cached_response = cache_manager.get(input_text)
        if cached_response:
            logger.info("⚡ Cache hit!")
            response_time = time.time() - start_time
            statistics.record_request("text", response_time)
            # Determine conversation id (generate if not provided)
            conv_id = request.conversation_id if getattr(request, 'conversation_id', None) else str(uuid.uuid4())
            # Attach user_id if authenticated
            if current_user:
                background_tasks.add_task(chat_logger.add_log, input_text, cached_response, "text", user_id=current_user['id'], conversation_id=conv_id, cached=True)
            else:
                background_tasks.add_task(chat_logger.add_log, input_text, cached_response, "text", conversation_id=conv_id, cached=True)
            
            return VoiceResponse(
                status="success",
                input_type="text",
                input_text=input_text,
                response=cached_response,
                confidence=0.95,
                timestamp=datetime.now().isoformat(),
                processing_time=response_time,
                conversation_id=conv_id
            )
        
        # Generate response
        if not model_manager or not model_manager.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        response, inference_time = model_manager.generate_response(input_text)
        
        # Cache the response
        cache_manager.set(input_text, response)
        
        response_time = time.time() - start_time
        statistics.record_request("text", response_time)
        # Determine conversation id: prefer client-provided, else generate one now so we can return it to the client
        conv_id = request.conversation_id if getattr(request, 'conversation_id', None) else str(uuid.uuid4())
        if current_user:
            background_tasks.add_task(chat_logger.add_log, input_text, response, "text", user_id=current_user['id'], conversation_id=conv_id)
        else:
            background_tasks.add_task(chat_logger.add_log, input_text, response, "text", conversation_id=conv_id)
        
        logger.info(f"✅ Text response: {input_text[:50]}... → {response[:50]}...")
        
        return VoiceResponse(
            status="success",
            input_type="text",
            input_text=input_text,
            response=response,
            confidence=0.95,
            timestamp=datetime.now().isoformat(),
            processing_time=response_time,
            conversation_id=conv_id
        )
    
    except HTTPException:
        statistics.record_request("text", time.time() - start_time, error=True)
        raise
    except Exception as e:
        logger.error(f"❌ Text chat error: {e}")
        statistics.record_request("text", time.time() - start_time, error=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/history")
async def chat_history(limit: int = 50, current_user: dict = Depends(get_current_user)):
    """Get authenticated user's chat history (latest first)"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        history = chat_logger.get_user_history(current_user['id'], limit=limit)
        return {"status": "success", "data": history}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/chat/conversations')
async def list_conversations(limit: int = 50, current_user: dict = Depends(get_current_user)):
    """List conversation threads for authenticated user, ordered newest first"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        chat_logger._ensure_connected()
        # Get both first and last messages per conversation (first = earliest, last = latest)
        pipeline = [
            {"$match": {"user_id": current_user['id']}},
            {"$sort": {"timestamp": 1}},  # ascending so $first is the earliest
            {"$group": {
                "_id": "$conversation_id",
                "first_timestamp": {"$first": "$timestamp"},
                "first_user_input": {"$first": "$user_input"},
                "first_bot_response": {"$first": "$bot_response"},
                "last_timestamp": {"$last": "$timestamp"},
                "last_user_input": {"$last": "$user_input"},
                "last_bot_response": {"$last": "$bot_response"},
                "count": {"$sum": 1}
            }},
            {"$sort": {"last_timestamp": -1}},
            {"$limit": limit}
        ]
        results = list(chat_logger.chat_logs.aggregate(pipeline))
        conversations = []
        for r in results:
            first_snip = (r.get('first_user_input') or r.get('first_bot_response')) or ''
            last_snip = (r.get('last_user_input') or r.get('last_bot_response')) or ''
            conversations.append({
                "conversation_id": r.get('_id'),
                "first_timestamp": r.get('first_timestamp'),
                "first_snippet": first_snip[:200] if first_snip else '',
                "last_timestamp": r.get('last_timestamp'),
                "last_snippet": last_snip[:200] if last_snip else '',
                "count": r.get('count', 0)
            })
        return {"status": "success", "data": conversations}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail="Failed to list conversations")


@app.get('/api/chat/conversations/{conversation_id}')
async def get_conversation(conversation_id: str, limit: int = 100, current_user: dict = Depends(get_current_user)):
    """Get messages for a conversation thread"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        chat_logger._ensure_connected()
        cursor = chat_logger.chat_logs.find({"user_id": current_user['id'], "conversation_id": conversation_id}).sort("timestamp", 1).limit(limit)
        messages = list(cursor)
        for m in messages:
            if '_id' in m:
                m['_id'] = str(m['_id'])
        return {"status": "success", "data": messages}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch conversation")


@app.post('/api/chat/conversations')
async def create_conversation(payload: Dict = None, current_user: dict = Depends(get_current_user)):
    """Create a new conversation for the authenticated user. Optional JSON body: { "initial_message": "..." }"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        body = payload or {}
        initial = body.get('initial_message') if isinstance(body, dict) else None
        # Generate conversation id
        conv_id = str(uuid.uuid4())
        chat_logger._ensure_connected()
        # Create an initial system/bot message so the conversation shows a first message
        if initial:
            # log initial user message; bot_response may be empty until model replies
            chat_logger.add_log(initial, "", "text", user_id=current_user['id'], conversation_id=conv_id)
        else:
            # create a starter bot message
            chat_logger.add_log("", "Conversation started", "system", user_id=current_user['id'], conversation_id=conv_id)
        return {"status": "success", "conversation_id": conv_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to create conversation")


@app.delete('/api/chat/conversations/{conversation_id}')
async def delete_conversation(conversation_id: str, current_user: dict = Depends(get_current_user)):
    """Delete all messages for a conversation (owned by the authenticated user)"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        chat_logger._ensure_connected()
        result = chat_logger.chat_logs.delete_many({"user_id": current_user['id'], "conversation_id": conversation_id})
        logger.info(f"🗑️ Deleted {result.deleted_count} messages for conversation {conversation_id}")
        return {"status": "success", "deleted": int(result.deleted_count)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete conversation")


@app.post('/api/chat/conversations/{conversation_id}/delete')
async def delete_conversation_post(conversation_id: str, current_user: dict = Depends(get_current_user)):
    """Fallback delete endpoint using POST for clients that cannot send DELETE. Calls same delete logic."""
    return await delete_conversation(conversation_id, current_user=current_user)


@app.options('/api/chat/conversations/{conversation_id}')
async def options_conversation(conversation_id: str):
    """Answer preflight OPTIONS for conversation routes."""
    from fastapi.responses import Response
    headers = {
        "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "*",
    }
    return Response(status_code=200, headers=headers)

@app.post("/api/chat/voice", response_model=VoiceResponse)
async def voice_chat(file: UploadFile = File(...), conversation_id: Optional[str] = Form(None), background_tasks: BackgroundTasks = None, current_user: Optional[dict] = Depends(get_current_user)):
    """
    Process voice input, transcribe, and return AI response
    
    Accepts: WAV, MP3, FLAC, OGG, WebM audio files
    Optional: conversation_id (form field) to attach message to a conversation
    """
    start_time = time.time()
    
    try:
        # Validate file
        if not file:
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        # Read file
        audio_bytes = await file.read()
        
        # Validate audio
        is_valid, error_msg = audio_processor.validate_audio(file.filename, len(audio_bytes))
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        audio_format = file.filename.split('.')[-1].lower()
        
        logger.info(f"📝 Processing voice input: {file.filename}")
        
        # Transcribe audio
        transcribed_text, confidence = audio_processor.transcribe_audio(audio_bytes, audio_format)
        
        if not transcribed_text:
            raise HTTPException(
                status_code=400,
                detail="Could not transcribe audio. Please try again with clearer audio."
            )
        
        logger.info(f"✅ Transcribed: {transcribed_text}")
        
        # Generate response
        if not model_manager or not model_manager.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        response, inference_time = model_manager.generate_response(transcribed_text)
        
        response_time = time.time() - start_time
        statistics.record_request("voice", response_time)
        # Ensure conversation id exists (client may provide or we generate one)
        conv_id = conversation_id or str(uuid.uuid4())
        if background_tasks:
            if current_user:
                background_tasks.add_task(
                    chat_logger.add_log,
                    transcribed_text,
                    response,
                    "voice",
                    user_id=current_user['id'],
                    conversation_id=conv_id,
                    audio_file=file.filename
                )
            else:
                background_tasks.add_task(
                    chat_logger.add_log,
                    transcribed_text,
                    response,
                    "voice",
                    conversation_id=conv_id,
                    audio_file=file.filename
                )
        
        logger.info(f"✅ Voice response: {transcribed_text[:50]}... → {response[:50]}... (conv={conv_id})")

        return VoiceResponse(
            status="success",
            input_type="voice",
            input_text=transcribed_text,
            response=response,
            confidence=float(confidence),
            timestamp=datetime.now().isoformat(),
            processing_time=response_time,
            conversation_id=conv_id
        )
        
        return VoiceResponse(
            status="success",
            input_type="voice",
            input_text=transcribed_text,
            response=response,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            processing_time=response_time
        )
    
    except HTTPException:
        statistics.record_request("voice", time.time() - start_time, error=True)
        raise
    except Exception as e:
        logger.error(f"❌ Voice chat error: {e}")
        statistics.record_request("voice", time.time() - start_time, error=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/batch")
async def batch_chat(request: BatchRequest, background_tasks: BackgroundTasks):
    """
    Process multiple text inputs in batch
    
    Request body:
    {
        "requests": [
            {"text": "Question 1"},
            {"text": "Question 2"}
        ]
    }
    """
    try:
        if not model_manager or not model_manager.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        results = []
        
        for req in request.requests:
            try:
                response, _ = model_manager.generate_response(req.text.strip())
                results.append({
                    "input": req.text,
                    "response": response,
                    "status": "success"
                })
                background_tasks.add_task(chat_logger.add_log, req.text, response, "text", batch=True)
            except Exception as e:
                logger.error(f"Error processing batch item: {e}")
                results.append({
                    "input": req.text,
                    "error": str(e),
                    "status": "error"
                })
        
        logger.info(f"✅ Batch processing complete: {len(results)} items")
        
        return {
            "status": "complete",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"❌ Batch chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs")
async def get_logs(limit: int = 50, user_id: Optional[str] = None, current_user: Optional[dict] = Depends(get_current_user)):
    """Get recent chat logs from MongoDB. If authenticated, returns only the logs for the current user (unless user_id param equals the current user id)."""
    try:
        # Ensure ChatLogger is connected to DB
        try:
            chat_logger._ensure_connected()
        except Exception as e:
            logger.error(f"Error ensuring chat logger DB connection: {e}")
            raise HTTPException(status_code=503, detail="Chat logs database not available")

        if current_user:
            if user_id and user_id != current_user['id']:
                raise HTTPException(status_code=403, detail="Forbidden: cannot access other user's logs")

            # Fetch user's logs from DB
            history = chat_logger.get_user_history(current_user['id'], limit=limit)
            total = chat_logger.chat_logs.count_documents({"user_id": current_user['id']})

            return {
                "status": "success",
                "total_logs": total,
                "recent_logs": history,
                "user_id": current_user['id'],
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Unauthenticated - disallow user-specific queries
            if user_id:
                raise HTTPException(status_code=401, detail="Authentication required to view user-specific logs")

            # Fetch recent global logs
            cursor = chat_logger.chat_logs.find({}).sort("timestamp", -1).limit(limit)
            recent_logs = []
            for item in cursor:
                if '_id' in item:
                    item['_id'] = str(item['_id'])
                # sanitize metadata
                if 'metadata' in item and 'user_id' in item['metadata']:
                    item['metadata'] = {k: v for k, v in item['metadata'].items() if k != 'user_id'}
                recent_logs.append(item)

            total = chat_logger.chat_logs.count_documents({})
            return {
                "status": "success",
                "total_logs": total,
                "recent_logs": recent_logs,
                "timestamp": datetime.now().isoformat()
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat
    
    JSON message format:
    {
        "type": "message",
        "input_type": "text" or "voice",
        "data": "text input or base64 audio"
    }
    """
    await websocket.accept()
    logger.info("🔌 WebSocket connection established")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "message":
                input_type = data.get("input_type", "text")
                start_time = time.time()
                
                if input_type == "text":
                    text = data.get("data", "").strip()
                    if not text:
                        await websocket.send_json({
                            "status": "error",
                            "message": "Empty input"
                        })
                        continue
                    
                    try:
                        response, _ = model_manager.generate_response(text)
                        response_time = time.time() - start_time
                        statistics.record_request("text", response_time)
                        
                        await websocket.send_json({
                            "status": "success",
                            "input": text,
                            "response": response,
                            "input_type": "text",
                            "timestamp": datetime.now().isoformat(),
                            "processing_time": response_time
                        })
                    except Exception as e:
                        await websocket.send_json({
                            "status": "error",
                            "message": str(e)
                        })
                
                elif input_type == "voice":
                    import base64
                    audio_data = data.get("data", "")
                    try:
                        audio_bytes = base64.b64decode(audio_data)
                        transcribed, confidence = audio_processor.transcribe_audio(audio_bytes)
                        
                        if transcribed:
                            response, _ = model_manager.generate_response(transcribed)
                            response_time = time.time() - start_time
                            statistics.record_request("voice", response_time)
                            
                            await websocket.send_json({
                                "status": "success",
                                "input": transcribed,
                                "response": response,
                                "input_type": "voice",
                                "confidence": confidence,
                                "timestamp": datetime.now().isoformat(),
                                "processing_time": response_time
                            })
                        else:
                            await websocket.send_json({
                                "status": "error",
                                "message": "Could not transcribe audio"
                            })
                    except Exception as e:
                        logger.error(f"WebSocket voice error: {e}")
                        await websocket.send_json({
                            "status": "error",
                            "message": f"Error processing voice: {str(e)}"
                        })
    
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        logger.info("🔌 WebSocket connection closed")

# ===================== 14. STARTUP/SHUTDOWN EVENTS =====================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("🚀 Healthcare Chatbot API starting...")
    logger.info(f"🖥️  Device: {DEVICE}")
    logger.info(f"📁 Model path: {MODEL_PATH}")
    logger.info(f"✅ Model loaded: {model_manager.is_loaded if model_manager else False}")
    
    # Initialize MongoDB connection
    try:
        from database import init_mongodb, mongodb
        if init_mongodb():
            logger.info("✅ MongoDB connected successfully!")
            status = mongodb.check_connection()
            logger.info(f"📊 Database: {status['database']}")
            logger.info(f"📦 Collections: {status.get('collections', [])}")
            logger.info(f"📈 Document counts: {status.get('document_counts', {})}")
        else:
            logger.warning("⚠️ MongoDB connection failed - using JSON file storage as fallback")
    except Exception as e:
        logger.warning(f"⚠️ MongoDB initialization error: {e} - using JSON file storage as fallback")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Healthcare Chatbot API shutting down...")
    logger.info(f"📊 Final stats: {statistics.get_stats()}")
    
    # Close MongoDB connection
    try:
        from database import mongodb
        mongodb.disconnect()
    except:
        pass

# ===================== 15. MAIN EXECUTION =====================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("🏥 Healthcare Chatbot FastAPI Server - Complete Backend")
    print("=" * 70)
    print(f"🖥️  Device: {DEVICE}")
    print(f"📁 Model Path: {MODEL_PATH}")
    print(f"✅ Model Loaded: {model_manager.is_loaded if model_manager else False}")
    print("=" * 70)
    print("📚 API Documentation: http://localhost:8000/docs")
    print("📍 Server running on: http://localhost:8000")
    print("=" * 70)
    print("\n✨ Features:")
    print("  ✅ Text input endpoint")
    print("  ✅ Voice input endpoint")
    print("  ✅ Batch processing")
    print("  ✅ WebSocket real-time chat")
    print("  ✅ Response caching")
    print("  ✅ Chat logging")
    print("  ✅ Statistics tracking")
    print("  ✅ Health checks")
    print("  ✅ Error handling")
    print("  ✅ Audio processing")
    print("\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )