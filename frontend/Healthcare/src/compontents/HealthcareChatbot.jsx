// ===================== HEALTHCARE CHATBOT - REACT FRONTEND (UPDATED) =====================
// Complete React component with PROPER voice input that sends to backend for AI processing
// Voice → Microphone → Record Audio → Send to Backend → AI Model → Response → Display

// ===================== 1. INSTALL DEPENDENCIES =====================
// npm install axios react-icons

// ===================== 2. HEALTHCARECHATBOT.JSX (COMPLETE WORKING VERSION) =====================
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { FiMic, FiSend, FiRefreshCw, FiTrash2, FiVolume2, FiStopCircle } from 'react-icons/fi';
import './HealthcareChatbot.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const HealthcareChatbot = () => {
  // ========== STATE MANAGEMENT ==========
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      text: 'Hello! I\'m your healthcare assistant. How can I help you today? You can type or use voice input.',
      timestamp: new Date(),
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voiceSupported, setVoiceSupported] = useState(true);
  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');

  // ====== AUTH ======
  const [token, setToken] = useState(localStorage.getItem('token') || null);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem('token'));

  // Apply token to axios default headers
  useEffect(() => {
    if (token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      localStorage.setItem('token', token);
    } else {
      delete axios.defaults.headers.common['Authorization'];
      localStorage.removeItem('token');
    }
  }, [token]);

  // ========== REFS ==========
  const messagesEndRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);

  // ========== CHECK BROWSER SUPPORT ==========
  useEffect(() => {
    const checkMicrophoneSupport = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach(track => track.stop());
        setVoiceSupported(true);
      } catch (err) {
        setVoiceSupported(false);
        console.warn('Microphone not supported:', err);
      }
    };

    checkMicrophoneSupport();

    // If token exists on mount, fetch history
    if (isAuthenticated) {
      fetchHistory();
    }
  }, [isAuthenticated]);

  // ========== AUTO SCROLL TO LATEST MESSAGE ==========
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ========== UTILITY FUNCTIONS ==========

  const generateId = () => Math.random().toString(36).substr(2, 9);

  useEffect(() => {
    if (error || successMessage) {
      const timer = setTimeout(() => {
        setError('');
        setSuccessMessage('');
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [error, successMessage]);

  // ========== SPEECH SYNTHESIS (Text-to-Speech) ==========
  const speakResponse = (text) => {
    if (!window.speechSynthesis) {
      setError('Speech synthesis not supported in this browser');
      return;
    }

    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1;
    utterance.pitch = 1;
    utterance.volume = 1;

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => {
      setError('Error during speech synthesis');
      setIsSpeaking(false);
    };

    window.speechSynthesis.speak(utterance);
  };

  // ========== API CALLS ==========

  // Send text message to backend
  const sendTextMessage = async (text) => {
    if (!text.trim()) return;

    setIsLoading(true);
    setError('');

    try {
      // Add user message to chat
      const userMessage = {
        id: generateId(),
        type: 'user',
        text: text.trim(),
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMessage]);
      setInputText('');

      console.log('📤 Sending text to backend:', text);

      // Send to API
      const response = await axios.post(
        `${API_BASE_URL}/api/chat/text`,
        { text: text.trim() },
        { timeout: 30000 }
      );

      console.log('📥 Received response:', response.data);

      if (response.data.status === 'success') {
        // Add bot response
        const botMessage = {
          id: generateId(),
          type: 'bot',
          text: response.data.response,
          timestamp: new Date(),
          inputType: 'text',
        };

        setMessages((prev) => [...prev, botMessage]);
        setSuccessMessage('✅ Response received!');
      } else {
        setError('❌ Failed to get response from server');
      }
    } catch (err) {
      console.error('❌ Error:', err);
      setError(
        err.response?.data?.detail ||
        err.message ||
        'Failed to send message. Please check if backend is running on ' + API_BASE_URL
      );
    } finally {
      setIsLoading(false);
    }
  };

  // ========== VOICE INPUT WITH AUDIO FILE SEND ==========
  const startVoiceRecording = async () => {
    setError('');
    
    try {
      console.log('🎤 Starting voice recording...');

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Determine supported MIME type
      let mimeType = 'audio/webm';
      if (MediaRecorder.isTypeSupported('audio/webm')) {
        mimeType = 'audio/webm';
      } else if (MediaRecorder.isTypeSupported('audio/ogg')) {
        mimeType = 'audio/ogg';
      } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
        mimeType = 'audio/mp4';
      }

      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        console.log('⏹️  Voice recording stopped');
        
        // Create audio blob with correct MIME type
        const audioBlob = new Blob(audioChunksRef.current, {
          type: mimeType,
        });

        // Determine file extension based on MIME type
        const extension = mimeType.split('/')[1].split(';')[0];

        // Stop all tracks
        stream.getTracks().forEach((track) => track.stop());
        setIsRecording(false);

        // Send audio to backend
        await sendVoiceAudioToBackend(audioBlob, extension);
      };

      mediaRecorder.start();
      mediaRecorderRef.current = mediaRecorder;
      setIsRecording(true);
      setSuccessMessage('🎙️ Recording... Speak now!');

    } catch (err) {
      console.error('❌ Microphone error:', err);
      setError('Microphone access denied. Please check browser permissions.');
      setVoiceSupported(false);
    }
  };

  const stopVoiceRecording = () => {
    if (mediaRecorderRef.current) {
      console.log('⏹️  Stopping recording...');
      mediaRecorderRef.current.stop();
    }
  };

  // Send audio file to backend for processing
  const sendVoiceAudioToBackend = async (audioBlob, extension = 'webm') => {
    setIsLoading(true);
    setError('');

    try {
      console.log('📤 Sending audio to backend...', { type: audioBlob.type, size: audioBlob.size, extension });

      // Create FormData for file upload
      const formData = new FormData();
      formData.append('file', audioBlob, `voice_input.${extension}`);

      // Send to voice endpoint
      const response = await axios.post(
        `${API_BASE_URL}/api/chat/voice`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 30000,
        }
      );

      console.log('📥 Voice response received:', response.data);

      if (response.data.status === 'success') {
        // Add user message with transcribed text
        const userMessage = {
          id: generateId(),
          type: 'user',
          text: response.data.input_text,
          timestamp: new Date(),
          isVoice: true,
        };

        setMessages((prev) => [...prev, userMessage]);

        // Add bot response
        const botMessage = {
          id: generateId(),
          type: 'bot',
          text: response.data.response,
          timestamp: new Date(),
          inputType: 'voice',
        };

        setMessages((prev) => [...prev, botMessage]);
        setSuccessMessage('✅ Voice message processed!');

        // Speak the response aloud
        speakResponse(response.data.response);

      } else {
        setError('❌ Failed to process voice message');
      }
    } catch (err) {
      console.error('❌ Error:', err);
      setError(
        err.response?.data?.detail ||
        err.message ||
        'Failed to process voice. Make sure backend is running.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  // ========== EVENT HANDLERS ==========

  const handleTextSubmit = (e) => {
    e.preventDefault();
    if (inputText.trim() && !isLoading) {
      sendTextMessage(inputText);
    }
  };

  const handleVoiceToggle = () => {
    if (!voiceSupported) {
      setError('🎤 Voice input not supported in your browser. Use Chrome or Edge.');
      return;
    }

    if (isRecording) {
      stopVoiceRecording();
    } else {
      startVoiceRecording();
    }
  };

  const handleClearChat = () => {
    if (window.confirm('Are you sure you want to clear chat history?')) {
      setMessages([
        {
          id: generateId(),
          type: 'bot',
          text: 'Chat cleared! How can I help you today?',
          timestamp: new Date(),
        },
      ]);
      setSuccessMessage('Chat cleared');
    }
  };

  const handleResetInput = () => {
    setInputText('');
  };

  // ====== AUTH HELPERS ======
  const signup = async (e) => {
    e.preventDefault();
    setError('');
    try {
      const resp = await axios.post(`${API_BASE_URL}/api/auth/signup`, { username, password });
      if (resp.data.status === 'success') {
        setSuccessMessage('Signup successful — you can login now');
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    }
  };

  const login = async (e) => {
    e.preventDefault();
    setError('');
    try {
      const resp = await axios.post(`${API_BASE_URL}/api/auth/login`, { username, password });
      if (resp.data.status === 'success') {
        setToken(resp.data.token);
        setIsAuthenticated(true);
        setSuccessMessage('Logged in');
        // fetch user chat history
        await fetchHistory();
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    }
  };

  const logout = async () => {
    try {
      await axios.post(`${API_BASE_URL}/api/auth/logout`);
    } catch (err) {
      // ignore
    }
    setToken(null);
    setIsAuthenticated(false);
    setSuccessMessage('Logged out');
    // clear to initial message
    setMessages([
      {
        id: generateId(),
        type: 'bot',
        text: 'Hello! I\'m your healthcare assistant. How can I help you today? You can type or use voice input.',
        timestamp: new Date(),
      },
    ]);
  };

  const fetchHistory = async () => {
    setIsLoading(true);
    setError('');
    try {
      const resp = await axios.get(`${API_BASE_URL}/api/logs`);
      if (resp.data.status === 'success') {
        const logs = resp.data.recent_logs || [];
        const msgs = [];
        logs.forEach((l) => {
          // user message
          msgs.push({ id: generateId(), type: 'user', text: l.user_input, timestamp: new Date(l.timestamp) });
          // bot message
          msgs.push({ id: generateId(), type: 'bot', text: l.bot_response, timestamp: new Date(l.timestamp) });
        });
        if (msgs.length) setMessages(msgs);
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setIsLoading(false);
    }
  };

  // ========== RENDER ==========
  return (
    <div className="chatbot-container">
      <div className="chatbot-header">
        <h1>🏥 Healthcare Chatbot</h1>
        <p className="subtitle">Voice & Text Support with AI Model</p>

        {/* Simple Auth UI */}
        <div style={{ marginTop: '10px' }}>
          {!isAuthenticated ? (
            <form style={{ display: 'flex', gap: '8px', alignItems: 'center' }} onSubmit={(e) => login(e)}>
              <input placeholder="username" value={username} onChange={(e) => setUsername(e.target.value)} />
              <input placeholder="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
              <button type="submit" className="btn">Login</button>
              <button type="button" onClick={(e) => signup(e)} className="btn">Signup</button>
            </form>
          ) : (
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <strong>Signed in</strong>
              <button className="btn" onClick={logout}>Logout</button>
            </div>
          )}
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="alert alert-error">
          {error}
        </div>
      )}

      {/* Success Message */}
      {successMessage && (
        <div className="alert alert-success">
          {successMessage}
        </div>
      )}

      {/* Messages Container */}
      <div className="messages-container">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`message ${msg.type === 'user' ? 'user-message' : 'bot-message'}`}
          >
            <div className="message-content">
              <p>{msg.text}</p>
              {msg.isVoice && (
                <span className="badge badge-voice">🎤 Voice Input</span>
              )}
              {msg.inputType === 'voice' && msg.type === 'bot' && (
                <button
                  className="speak-btn"
                  onClick={() => speakResponse(msg.text)}
                  disabled={isSpeaking}
                  title="Read message aloud"
                >
                  <FiVolume2 /> {isSpeaking ? 'Speaking...' : 'Speak'}
                </button>
              )}
            </div>
            <span className="message-time">
              {msg.timestamp.toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit',
              })}
            </span>
          </div>
        ))}
        {isLoading && (
          <div className="message bot-message">
            <div className="typing-indicator">
              <span></span><span></span><span></span>
            </div>
            <p style={{ marginTop: '10px', fontSize: '12px', color: '#666' }}>
              🤖 AI Model is generating response...
            </p>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="input-area">
        <form onSubmit={handleTextSubmit} className="input-form">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Type your health question..."
            disabled={isLoading || isRecording}
            className="text-input"
          />

          <button
            type="button"
            onClick={handleResetInput}
            disabled={isLoading || !inputText.trim()}
            className="btn btn-reset"
            title="Clear input"
          >
            <FiRefreshCw />
          </button>

          <button
            type="button"
            onClick={handleVoiceToggle}
            disabled={isLoading}
            className={`btn btn-voice ${isRecording ? 'recording' : ''}`}
            title={voiceSupported ? 'Click to record voice' : 'Voice not supported'}
          >
            {isRecording ? (
              <>
                <FiStopCircle /> Stop Recording
              </>
            ) : (
              <>
                <FiMic /> Voice Input
              </>
            )}
          </button>

          <button
            type="submit"
            disabled={isLoading || !inputText.trim()}
            className="btn btn-send"
            title="Send message"
          >
            <FiSend /> {isLoading ? 'Sending...' : 'Send'}
          </button>
        </form>

        <button
          onClick={handleClearChat}
          disabled={isLoading}
          className="btn btn-clear"
          title="Clear chat history"
        >
          <FiTrash2 /> Clear Chat
        </button>
      </div>

      {/* Browser Support Info */}
      <div className="footer">
        {!voiceSupported && (
          <p className="warning">
            ⚠️ Voice input requires Chrome/Edge browser and microphone permission. Use text input instead.
          </p>
        )}
        <p className="info">
          💡 <strong>How to use:</strong> Type a question OR click "Voice Input" to speak to the AI model
        </p>
        <p className="info">
          🔗 Backend: {API_BASE_URL}
        </p>
      </div>
    </div>
  );
};

export default HealthcareChatbot;