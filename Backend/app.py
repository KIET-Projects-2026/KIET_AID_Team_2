# ===================== HEALTHCARE CHATBOT - COMPLETE FASTAPI BACKEND =====================
# Production-ready FastAPI backend with full voice and text support
# All functionalities included: REST endpoints, WebSocket, audio processing, logging, caching

# ===================== 1. REQUIREMENTS =====================
# pip install fastapi uvicorn torch transformers peft accelerate python-multipart
# pip install SpeechRecognition pydub librosa numpy scipy
# pip install python-dotenv

# ===================== 2. IMPORTS =====================
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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
from datetime import datetime
import time
from functools import wraps
import numpy as np

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

class VoiceResponse(BaseModel):
    """Response model for voice/text input"""
    status: str
    input_type: str
    input_text: str
    response: str
    confidence: float = 0.95
    timestamp: Optional[str] = None
    processing_time: Optional[float] = None

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
    """Log all conversations"""
    
    def __init__(self, log_file: str = "chat_logs.json"):
        self.log_file = log_file
        self.logs = []
        self.load_logs()
    
    def load_logs(self):
        """Load existing logs"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    self.logs = json.load(f)
                logger.info(f"📖 Loaded {len(self.logs)} chat logs")
            except Exception as e:
                logger.error(f"Error loading logs: {e}")
                self.logs = []
    
    def save_logs(self):
        """Save logs to file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.logs, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving logs: {e}")
    
    def add_log(self, user_input: str, bot_response: str, input_type: str = "text", user_id: Optional[str] = None, **metadata):
        """Add conversation to log -- associates user_id when provided"""
        if user_id:
            metadata['user_id'] = user_id
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "bot_response": bot_response,
            "input_type": input_type,
            "metadata": metadata
        }
        self.logs.append(log_entry)
        self.save_logs()

chat_logger = ChatLogger()

# ===================== 12.5 USERS & AUTHENTICATION =====================

class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)

class AuthResponse(BaseModel):
    status: str
    token: Optional[str] = None
    user_id: Optional[str] = None

class UsersManager:
    """Simple file-backed user manager with token sessions"""
    def __init__(self, users_file: str = "users.json"):
        self.users_file = users_file
        self.users: Dict[str, dict] = {}
        self.tokens: Dict[str, str] = {}  # token -> user_id
        self.load_users()

    def load_users(self):
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    data = json.load(f)
                    self.users = {u['id']: u for u in data.get('users', [])}
                    self.tokens = data.get('tokens', {})
            except Exception as e:
                logger.error(f"Error loading users file: {e}")
                self.users = {}
                self.tokens = {}

    def save_users(self):
        try:
            data = {
                'users': list(self.users.values()),
                'tokens': self.tokens
            }
            with open(self.users_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving users file: {e}")

    def _hash_password(self, password: str, salt: str) -> str:
        hashed = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return hashed.hex()

    def create_user(self, username: str, password: str) -> dict:
        # Ensure username unique
        for u in self.users.values():
            if u['username'].lower() == username.lower():
                raise ValueError('Username already exists')

        user_id = str(uuid.uuid4())
        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)
        user = {
            'id': user_id,
            'username': username,
            'password_hash': password_hash,
            'salt': salt,
            'created_at': datetime.now().isoformat()
        }
        self.users[user_id] = user
        self.save_users()
        logger.info(f"✅ Created user: {username} ({user_id})")
        return user

    def verify_user(self, username: str, password: str) -> Optional[dict]:
        for u in self.users.values():
            if u['username'].lower() == username.lower():
                expected = u['password_hash']
                derived = self._hash_password(password, u['salt'])
                if secrets.compare_digest(derived, expected):
                    return u
        return None

    def create_token(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        self.tokens[token] = user_id
        self.save_users()
        return token

    def revoke_token(self, token: str):
        if token in self.tokens:
            del self.tokens[token]
            self.save_users()

    def get_user_by_token(self, token: str) -> Optional[dict]:
        user_id = self.tokens.get(token)
        if not user_id:
            return None
        return self.users.get(user_id)

    def get_user_by_id(self, user_id: str) -> Optional[dict]:
        return self.users.get(user_id)

users_manager = UsersManager()

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
    """Health check endpoint with detailed status"""
    return {
        "status": "healthy" if model_manager.is_loaded else "degraded",
        "device": DEVICE,
        "model_loaded": model_manager.is_loaded if model_manager else False,
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
        user = users_manager.create_user(payload.username, payload.password)
        return AuthResponse(status='success', user_id=user['id'])
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
        return AuthResponse(status='success', token=token, user_id=user['id'])
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    """Get current user info"""
    return {"status": "success", "user": {"id": current_user['id'], "username": current_user['username']}}
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
            # Attach user_id if authenticated
            if current_user:
                background_tasks.add_task(chat_logger.add_log, input_text, cached_response, "text", user_id=current_user['id'], cached=True)
            else:
                background_tasks.add_task(chat_logger.add_log, input_text, cached_response, "text", cached=True)
            
            return VoiceResponse(
                status="success",
                input_type="text",
                input_text=input_text,
                response=cached_response,
                confidence=0.95,
                timestamp=datetime.now().isoformat(),
                processing_time=response_time
            )
        
        # Generate response
        if not model_manager or not model_manager.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        response, inference_time = model_manager.generate_response(input_text)
        
        # Cache the response
        cache_manager.set(input_text, response)
        
        response_time = time.time() - start_time
        statistics.record_request("text", response_time)
        # Attach user metadata when available
        if current_user:
            background_tasks.add_task(chat_logger.add_log, input_text, response, "text", user_id=current_user['id'])
        else:
            background_tasks.add_task(chat_logger.add_log, input_text, response, "text")
        
        logger.info(f"✅ Text response: {input_text[:50]}... → {response[:50]}...")
        
        return VoiceResponse(
            status="success",
            input_type="text",
            input_text=input_text,
            response=response,
            confidence=0.95,
            timestamp=datetime.now().isoformat(),
            processing_time=response_time
        )
    
    except HTTPException:
        statistics.record_request("text", time.time() - start_time, error=True)
        raise
    except Exception as e:
        logger.error(f"❌ Text chat error: {e}")
        statistics.record_request("text", time.time() - start_time, error=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/voice", response_model=VoiceResponse)
async def voice_chat(file: UploadFile = File(...), background_tasks: BackgroundTasks = None, current_user: Optional[dict] = Depends(get_current_user)):
    """
    Process voice input, transcribe, and return AI response
    
    Accepts: WAV, MP3, FLAC, OGG, WebM audio files
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
        
        if background_tasks:
            if current_user:
                background_tasks.add_task(
                    chat_logger.add_log,
                    transcribed_text,
                    response,
                    "voice",
                    user_id=current_user['id'],
                    audio_file=file.filename
                )
            else:
                background_tasks.add_task(
                    chat_logger.add_log,
                    transcribed_text,
                    response,
                    "voice",
                    audio_file=file.filename
                )
        
        logger.info(f"✅ Voice response: {transcribed_text[:50]}... → {response[:50]}...")
        
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
    """Get recent chat logs. If authenticated, returns only the logs for the current user (unless user_id param equals the current user id)."""
    try:
        # If requester is authenticated, return only that user's logs (or allow same-id filter)
        if current_user:
            if user_id and user_id != current_user['id']:
                raise HTTPException(status_code=403, detail="Forbidden: cannot access other user's logs")
            # Filter logs for user
            filtered = [l for l in chat_logger.logs if l.get('metadata', {}).get('user_id') == current_user['id']]
            recent_logs = filtered[-limit:]
            return {
                "status": "success",
                "total_logs": len(filtered),
                "recent_logs": recent_logs,
                "user_id": current_user['id'],
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Unauthenticated - disallow user-specific queries
            if user_id:
                raise HTTPException(status_code=401, detail="Authentication required to view user-specific logs")
            recent_logs = chat_logger.logs[-limit:]
            # Return logs without revealing metadata user_ids in unauthenticated mode
            sanitized = []
            for l in recent_logs:
                copy = dict(l)
                # remove user id if present
                if 'metadata' in copy and 'user_id' in copy['metadata']:
                    copy['metadata'] = {k: v for k, v in copy['metadata'].items() if k != 'user_id'}
                sanitized.append(copy)
            return {
                "status": "success",
                "total_logs": len(chat_logger.logs),
                "recent_logs": sanitized,
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

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Healthcare Chatbot API shutting down...")
    logger.info(f"📊 Final stats: {statistics.get_stats()}")

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