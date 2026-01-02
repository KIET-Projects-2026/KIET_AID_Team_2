# ===================== HEALTHCARE CHATBOT - UTILS & INTEGRATION =====================
# Utility functions and example integration code for the healthcare chatbot system

# ===================== PYTHON UTILITIES =====================

import os
import json
import torch
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# ===================== 1. MODEL UTILITIES =====================

class ModelConfig:
    """Configuration for model inference"""
    
    DEFAULT_MAX_LENGTH = 256
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_NUM_BEAMS = 4
    DEFAULT_TOP_P = 0.9
    DEFAULT_REPETITION_PENALTY = 1.2
    DEFAULT_LANGUAGE = 'en-US'
    
    @staticmethod
    def get_config() -> dict:
        """Get default configuration"""
        return {
            "max_length": ModelConfig.DEFAULT_MAX_LENGTH,
            "temperature": ModelConfig.DEFAULT_TEMPERATURE,
            "num_beams": ModelConfig.DEFAULT_NUM_BEAMS,
            "top_p": ModelConfig.DEFAULT_TOP_P,
            "repetition_penalty": ModelConfig.DEFAULT_REPETITION_PENALTY
        }

# ===================== 2. AUDIO UTILITIES =====================

class AudioUtils:
    """Utilities for audio processing"""
    
    SUPPORTED_FORMATS = ['wav', 'mp3', 'flac', 'ogg', 'webm']
    MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25 MB
    
    @staticmethod
    def validate_audio_format(filename: str) -> bool:
        """
        Validate audio file format
        
        Args:
            filename: Name of audio file
        
        Returns:
            True if format is supported
        """
        ext = filename.split('.')[-1].lower()
        return ext in AudioUtils.SUPPORTED_FORMATS
    
    @staticmethod
    def validate_audio_size(file_size: int) -> bool:
        """
        Validate audio file size
        
        Args:
            file_size: Size of audio file in bytes
        
        Returns:
            True if size is acceptable
        """
        return file_size <= AudioUtils.MAX_AUDIO_SIZE
    
    @staticmethod
    def get_audio_format_from_content_type(content_type: str) -> str:
        """
        Extract audio format from content type
        
        Args:
            content_type: MIME type string
        
        Returns:
            Audio format string
        """
        format_map = {
            'audio/wav': 'wav',
            'audio/mpeg': 'mp3',
            'audio/flac': 'flac',
            'audio/ogg': 'ogg',
            'audio/webm': 'webm'
        }
        return format_map.get(content_type, 'wav')

# ===================== 3. TEXT PROCESSING UTILITIES =====================

class TextProcessor:
    """Text processing utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean input text
        
        Args:
            text: Raw input text
        
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters
        text = text.strip()
        return text
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 256) -> str:
        """
        Truncate text to maximum length
        
        Args:
            text: Input text
            max_length: Maximum length
        
        Returns:
            Truncated text
        """
        words = text.split()
        if len(words) > max_length:
            return ' '.join(words[:max_length])
        return text
    
    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        """
        Extract keywords from text (simple implementation)
        
        Args:
            text: Input text
        
        Returns:
            List of keywords
        """
        # Simple keyword extraction (can be improved)
        keywords = []
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'what', 'how', 'why', 'when'}
        
        words = text.lower().split()
        for word in words:
            if word not in stop_words and len(word) > 3:
                keywords.append(word)
        
        return list(set(keywords))[:5]  # Return top 5 unique keywords

# ===================== 4. RESPONSE UTILITIES =====================

class ResponseFormatter:
    """Format AI responses for different outputs"""
    
    @staticmethod
    def format_json_response(status: str, response: str, input_text: str, **kwargs) -> dict:
        """
        Format response as JSON
        
        Args:
            status: Response status (success/error)
            response: Generated response text
            input_text: Original input text
            **kwargs: Additional fields
        
        Returns:
            Formatted JSON response
        """
        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "input": input_text,
            "response": response,
            **kwargs
        }
    
    @staticmethod
    def format_text_response(response: str, include_meta: bool = False) -> str:
        """
        Format response as plain text
        
        Args:
            response: Generated response
            include_meta: Include metadata
        
        Returns:
            Formatted text response
        """
        if include_meta:
            return f"[{datetime.now().strftime('%H:%M:%S')}] {response}"
        return response
    
    @staticmethod
    def format_html_response(response: str) -> str:
        """
        Format response as HTML
        
        Args:
            response: Generated response
        
        Returns:
            HTML formatted response
        """
        return f"<p>{response}</p>"

# ===================== 5. LOGGING UTILITIES =====================

class ChatLogger:
    """Log chat conversations"""
    
    def __init__(self, log_file: str = "chat_logs.json"):
        self.log_file = log_file
        self.logs = []
        self.load_logs()
    
    def load_logs(self):
        """Load existing logs from file"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    self.logs = json.load(f)
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
    
    def add_log(self, user_input: str, bot_response: str, input_type: str = "text", **metadata):
        """
        Add conversation to log
        
        Args:
            user_input: User's input
            bot_response: Bot's response
            input_type: Type of input (text/voice)
            **metadata: Additional metadata
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "bot_response": bot_response,
            "input_type": input_type,
            "metadata": metadata
        }
        self.logs.append(log_entry)
        self.save_logs()
    
    def get_conversation_history(self, limit: int = 10) -> List[dict]:
        """Get recent conversation history"""
        return self.logs[-limit:]

# ===================== 6. VALIDATION UTILITIES =====================

class InputValidator:
    """Validate user inputs"""
    
    MIN_TEXT_LENGTH = 2
    MAX_TEXT_LENGTH = 1000
    
    @staticmethod
    def validate_text_input(text: str) -> tuple[bool, str]:
        """
        Validate text input
        
        Args:
            text: Input text
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not text or not text.strip():
            return False, "Input cannot be empty"
        
        if len(text) < InputValidator.MIN_TEXT_LENGTH:
            return False, f"Input must be at least {InputValidator.MIN_TEXT_LENGTH} characters"
        
        if len(text) > InputValidator.MAX_TEXT_LENGTH:
            return False, f"Input cannot exceed {InputValidator.MAX_TEXT_LENGTH} characters"
        
        return True, ""
    
    @staticmethod
    def validate_audio_input(file_size: int, file_format: str) -> tuple[bool, str]:
        """
        Validate audio input
        
        Args:
            file_size: Size of audio file
            file_format: Audio format
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not AudioUtils.validate_audio_format(file_format):
            return False, f"Unsupported audio format: {file_format}"
        
        if not AudioUtils.validate_audio_size(file_size):
            return False, f"Audio file too large (max {AudioUtils.MAX_AUDIO_SIZE / 1024 / 1024:.0f} MB)"
        
        return True, ""

# ===================== 7. PERFORMANCE UTILITIES =====================

import time
from functools import wraps

class PerformanceMonitor:
    """Monitor performance metrics"""
    
    metrics = {}
    
    @staticmethod
    def measure_time(func_name: str):
        """Decorator to measure function execution time"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                execution_time = end_time - start_time
                
                if func_name not in PerformanceMonitor.metrics:
                    PerformanceMonitor.metrics[func_name] = []
                
                PerformanceMonitor.metrics[func_name].append(execution_time)
                
                logger.info(f"{func_name} took {execution_time:.4f} seconds")
                return result
            return wrapper
        return decorator
    
    @staticmethod
    def get_metrics() -> dict:
        """Get performance metrics"""
        metrics_summary = {}
        for func_name, times in PerformanceMonitor.metrics.items():
            metrics_summary[func_name] = {
                "calls": len(times),
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times)
            }
        return metrics_summary

# ===================== 8. CACHE UTILITIES =====================

class ResponseCache:
    """Simple caching for responses"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[str]:
        """Get cached response"""
        return self.cache.get(key)
    
    def set(self, key: str, value: str):
        """Cache response"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
    
    def is_cached(self, key: str) -> bool:
        """Check if response is cached"""
        return key in self.cache

# ===================== 9. EXAMPLE USAGE =====================

if __name__ == "__main__":
    # Example 1: Text Processing
    print("=== Text Processing ===")
    text = "  What are  the   symptoms of diabetes?  "
    cleaned = TextProcessor.clean_text(text)
    print(f"Cleaned: {cleaned}")
    
    keywords = TextProcessor.extract_keywords(cleaned)
    print(f"Keywords: {keywords}")
    
    # Example 2: Input Validation
    print("\n=== Input Validation ===")
    is_valid, error = InputValidator.validate_text_input("What is fever?")
    print(f"Valid: {is_valid}, Error: {error}")
    
    # Example 3: Chat Logging
    print("\n=== Chat Logging ===")
    chat_logger = ChatLogger()
    chat_logger.add_log(
        user_input="What is diabetes?",
        bot_response="Diabetes is...",
        input_type="text",
        user_id="user123"
    )
    history = chat_logger.get_conversation_history(limit=5)
    print(f"Chat history: {len(history)} entries")
    
    # Example 4: Response Formatting
    print("\n=== Response Formatting ===")
    response_json = ResponseFormatter.format_json_response(
        status="success",
        response="Diabetes is a chronic disease...",
        input_text="What is diabetes?",
        confidence=0.95
    )
    print(f"JSON Response: {json.dumps(response_json, indent=2)}")
    
    # Example 5: Cache Usage
    print("\n=== Cache Usage ===")
    cache = ResponseCache(max_size=100)
    cache.set("What is diabetes?", "Diabetes is a chronic disease...")
    cached = cache.get("What is diabetes?")
    print(f"Cached response: {cached[:50]}...")
    
    # Example 6: Model Configuration
    print("\n=== Model Configuration ===")
    config = ModelConfig.get_config()
    print(f"Default config: {config}")