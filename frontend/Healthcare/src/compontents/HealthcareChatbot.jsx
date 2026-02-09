// ===================== HEALTHCARE CHATBOT - REACT FRONTEND (UPDATED) =====================
// Complete React component with PROPER voice input that sends to backend for AI processing
// Voice → Microphone → Record Audio → Send to Backend → AI Model → Response → Display

// ===================== 1. INSTALL DEPENDENCIES =====================
// npm install axios react-icons

// ===================== 2. HEALTHCARECHATBOT.JSX (COMPLETE WORKING VERSION) =====================
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { FiMic, FiSend, FiRefreshCw, FiTrash2, FiVolume2, FiStopCircle, FiLogOut, FiUser, FiMenu, FiChevronLeft } from 'react-icons/fi';
import { toast } from 'react-toastify';
import './HealthcareChatbot.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const HealthcareChatbot = ({ currentUser, onLogout }) => {
  // ========== STATE MANAGEMENT ==========
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      text: `Hello${currentUser?.username ? ' ' + currentUser.username : ''}! I'm your healthcare assistant. How can I help you today? You can type or use voice input.`,
      timestamp: new Date(),
    },
  ]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [conversationsLoading, setConversationsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voiceSupported, setVoiceSupported] = useState(true);



  // ========== REFS ==========
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
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

      // Fetch conversation list on mount if user is authenticated
    if (currentUser) {
      fetchConversations().then(async () => {
        // If user had an active conversation stored in localStorage, load it
        const saved = localStorage.getItem('activeConversationId');
        if (saved) {
          try {
            await loadConversation(saved);
            setActiveConversationId(saved);
            return;
          } catch (e) {
            // ignore and fallback to loading latest
            console.warn('Failed to load saved conversation:', e);
          }
        }

        // else auto-open the most recent conversation if any
        try {
          const resp = await axios.get(`${API_BASE_URL}/api/chat/conversations`);
          if (resp.data?.status === 'success' && resp.data.data && resp.data.data.length) {
            const latest = resp.data.data[0];
            loadConversation(latest.conversation_id);
          } else {
            // no existing conversations, keep default greeting
          }
        } catch (_) {}
      });
    }
  }, [currentUser]);

  // ========== AUTO SCROLL TO LATEST MESSAGE ==========
  useEffect(() => {
    // Only auto-scroll to bottom when the messages container is overflowing.
    // This avoids pinning a few messages to the bottom and leaving a large empty gap above.
    const container = messagesContainerRef.current;
    if (!container) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      return;
    }

    const isOverflowing = container.scrollHeight > container.clientHeight;
    if (isOverflowing) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    } else {
      // keep content at the top when there's no overflow
      container.scrollTop = 0;
    }
  }, [messages]);

  // ========== UTILITY FUNCTIONS ==========

  const generateId = () => Math.random().toString(36).substr(2, 9);



  // Persist active conversation id in localStorage whenever it changes
  useEffect(() => {
    if (activeConversationId) {
      localStorage.setItem('activeConversationId', activeConversationId);
    } else {
      localStorage.removeItem('activeConversationId');
    }
  }, [activeConversationId]);

  // ========== SPEECH SYNTHESIS (Text-to-Speech) ==========
  const speakResponse = (text) => {
    if (!window.speechSynthesis) {
      toast.error('Speech synthesis not supported in this browser');
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
      toast.error('Error during speech synthesis');
      setIsSpeaking(false);
    };

    window.speechSynthesis.speak(utterance);
  };

  // ========== API CALLS ==========

  // Fetch user's conversation list
  const fetchConversations = async () => {
    setConversationsLoading(true);
    try {
      const resp = await axios.get(`${API_BASE_URL}/api/chat/conversations`);
      if (resp.data?.status === 'success') {
        setConversations(resp.data.data || []);
      }
    } catch (e) {
      console.warn('Failed to load conversations:', e);
    } finally {
      setConversationsLoading(false);
    }
  };

  // Load a conversation's messages
  const loadConversation = async (conversationId) => {
    setIsLoading(true);
    try {
      const resp = await axios.get(`${API_BASE_URL}/api/chat/conversations/${conversationId}`);
      if (resp.data?.status === 'success') {
        const msgs = [];
        resp.data.data.forEach((m) => {
          if (m.user_input) msgs.push({ id: generateId(), type: 'user', text: m.user_input, timestamp: new Date(m.timestamp) });
          if (m.bot_response) msgs.push({ id: generateId(), type: 'bot', text: m.bot_response, timestamp: new Date(m.timestamp) });
        });
        setMessages(msgs.length ? msgs : [
          { id: generateId(), type: 'bot', text: 'Conversation started.', timestamp: new Date() }
        ]);
        setActiveConversationId(conversationId);
        localStorage.setItem('activeConversationId', conversationId);
        toast.success('📁 Conversation loaded');
      }
    } catch (err) {
      console.error('❌ Error loading conversation:', err);
      toast.error('Failed to load conversation');
      // If load failed (e.g., conversation deleted or unauthorized), clear saved id
      localStorage.removeItem('activeConversationId');
      setActiveConversationId(null);
    } finally {
      setIsLoading(false);
    }
  };

  // Create a new conversation on the server and load it
  const startNewConversation = async (initialMessage = null) => {
    setIsLoading(true);
    try {
      // Only include initial_message when it's a plain string (avoid passing DOM/event objects)
      const payload = (initialMessage && typeof initialMessage === 'string') ? { initial_message: initialMessage } : {};
      const resp = await axios.post(`${API_BASE_URL}/api/chat/conversations`, payload);
      if (resp.data?.status === 'success' && resp.data.conversation_id) {
        const convId = resp.data.conversation_id;
        setActiveConversationId(convId);
        // persist
        localStorage.setItem('activeConversationId', convId);
        // load conversation messages (should include initial message)
        await loadConversation(convId);
        // refresh list
        fetchConversations();
        toast.success('✨ New chat created');
      } else {
        toast.error('Failed to create new conversation');
      }
    } catch (err) {
      console.error('❌ Error creating conversation:', err);
      toast.error(err.response?.data?.detail || 'Failed to create conversation');
    } finally {
      setIsLoading(false);
    }
  };

  // Send text message to backend
  const sendTextMessage = async (text) => {
    if (!text.trim()) return;

    setIsLoading(true);

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

      console.log('📤 Sending text to backend:', text, 'conv=', activeConversationId);

      // If no active conversation, create one first
      let convId = activeConversationId;
      if (!convId) {
        const createResp = await axios.post(`${API_BASE_URL}/api/chat/conversations`, {});
        if (createResp.data?.status === 'success' && createResp.data.conversation_id) {
          convId = createResp.data.conversation_id;
          setActiveConversationId(convId);
          localStorage.setItem('activeConversationId', convId);
          fetchConversations();
        }
      }

      // Send to API (include conversation_id)
      const response = await axios.post(
        `${API_BASE_URL}/api/chat/text`,
        { text: text.trim(), conversation_id: convId },
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
        toast.success('✅ Response received!');

        // Save conversation id returned by backend for subsequent messages
        if (response.data.conversation_id) {
          setActiveConversationId(response.data.conversation_id);
          localStorage.setItem('activeConversationId', response.data.conversation_id);
          // Refresh conversation list
          fetchConversations();
        }
      } else {
        toast.error('Failed to get response from server');
      }
    } catch (err) {
      console.error('❌ Error:', err);
      toast.error(
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
      toast.info('🎙️ Recording... Speak now!');

    } catch (err) {
      console.error('❌ Microphone error:', err);
      toast.error('Microphone access denied. Please check browser permissions.');
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

    try {
      console.log('📤 Sending audio to backend...', { type: audioBlob.type, size: audioBlob.size, extension });

      // Create FormData for file upload
      const formData = new FormData();
      formData.append('file', audioBlob, `voice_input.${extension}`);

      // If no active conversation, create one on the server and attach it
      let convId = activeConversationId;
      if (!convId) {
        try {
          const createResp = await axios.post(`${API_BASE_URL}/api/chat/conversations`, {});
          if (createResp.data?.status === 'success' && createResp.data.conversation_id) {
            convId = createResp.data.conversation_id;
            setActiveConversationId(convId);
            localStorage.setItem('activeConversationId', convId);
            fetchConversations();
          }
        } catch (e) {
          console.warn('Failed to create conversation for voice:', e);
        }
      }

      if (convId) formData.append('conversation_id', convId);

      // Send to voice endpoint
      const response = await axios.post(
        `${API_BASE_URL}/api/chat/voice`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 30000,
        }
      );

      // Capture conversation id if backend provides it
      if (response.data?.conversation_id) {
        setActiveConversationId(response.data.conversation_id);
        localStorage.setItem('activeConversationId', response.data.conversation_id);
        fetchConversations();
      }

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
        toast.success('✅ Voice message processed!');

        // Speak the response aloud
        speakResponse(response.data.response);

      } else {
        toast.error('Failed to process voice message');
      }
    } catch (err) {
      console.error('❌ Error:', err);
      toast.error(
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
      toast.warning('🎤 Voice input not supported in your browser. Use Chrome or Edge.');
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
      // If user accepts, delete active conversation if set, otherwise clear local UI
      if (activeConversationId) {
        deleteConversation(activeConversationId);
      } else {
        setMessages([
          {
            id: generateId(),
            type: 'bot',
            text: 'Chat cleared! How can I help you today?',
            timestamp: new Date(),
          },
        ]);
        toast.success('🗑️ Chat cleared');
      }
    }
  };

  const handleResetInput = () => {
    setInputText('');
  };

  // Delete a conversation
  const deleteConversation = async (conversationId) => {
    if (!conversationId) return;
    setIsLoading(true);
    try {
      const resp = await axios.delete(`${API_BASE_URL}/api/chat/conversations/${conversationId}`);
      if (resp.data?.status === 'success') {
        toast.success('🗑️ Conversation deleted');
        // If we deleted the active conversation, clear UI
        if (activeConversationId === conversationId) {
          setActiveConversationId(null);
          localStorage.removeItem('activeConversationId');
          setMessages([
            { id: generateId(), type: 'bot', text: 'Conversation cleared. Start a new chat or select another.', timestamp: new Date() }
          ]);
        }
        fetchConversations();
      } else {
        toast.error('Failed to delete conversation');
      }
    } catch (err) {
      console.error('❌ Error deleting conversation:', err);
      // If server responded 405 Method Not Allowed, try POST fallback
      if (err.response?.status === 405) {
        try {
          const fallback = await axios.post(`${API_BASE_URL}/api/chat/conversations/${conversationId}/delete`);
          if (fallback.data?.status === 'success') {
            toast.success('🗑️ Conversation deleted');
            if (activeConversationId === conversationId) {
              setActiveConversationId(null);
              localStorage.removeItem('activeConversationId');
              setMessages([
                { id: generateId(), type: 'bot', text: 'Conversation cleared. Start a new chat or select another.', timestamp: new Date() }
              ]);
            }
            fetchConversations();
            return;
          }
        } catch (e) {
          console.error('❌ Fallback delete failed:', e);
        }
      }
      toast.error(err.response?.data?.detail || 'Failed to delete conversation');
    } finally {
      setIsLoading(false);
    }
  };

  // ====== FETCH CHAT HISTORY ======
  const fetchHistory = async () => {
    setIsLoading(true);
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
      toast.error(err.response?.data?.detail || err.message);
    } finally {
      setIsLoading(false);
    }
  };

  // ========== RENDER ==========
  return (
    <div className="chatbot-container">
      {/* Fixed left-edge hamburger for quickly opening/closing the chat history */}
      {/* <button
        className={`global-toggle ${sidebarOpen ? 'open' : ''}`}
        onClick={() => setSidebarOpen(!sidebarOpen)}
        aria-pressed={sidebarOpen}
        aria-label={sidebarOpen ? 'Close chats' : 'Open chats'}
        title={sidebarOpen ? 'Close chats' : 'Open chats'}
      >
        {sidebarOpen ? <FiChevronLeft /> : <FiMenu />}
      </button> */}
      <div className="chatbot-header">
        <div className="header-content">
          <div>
            <h1>🏥 Healthcare Chatbot</h1>
            <p className="subtitle">Voice & Text Support with AI Model</p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <button
              className="header-toggle"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              aria-pressed={sidebarOpen}
              title={sidebarOpen ? 'Close chats' : 'Open chats'}
            >
              {sidebarOpen ? <FiChevronLeft /> : <FiMenu />}
            </button>

            <div className="user-info">
              <div className="user-details">
                <FiUser className="user-icon" />
                <span className="username">{currentUser?.username || 'User'}</span>
              </div>
              <button className="logout-btn" onClick={onLogout} title="Logout">
                <FiLogOut />
                <span>Logout</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Toast notifications are handled by ToastContainer in App */}

      <div className={`chat-grid ${sidebarOpen ? '' : 'sidebar-closed'}`}>
        {/* Sidebar - Conversations */}
        <aside className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`} aria-hidden={!sidebarOpen}>
          <div className="sidebar-header">
            <h3>Chats</h3>
            <div className="sidebar-actions">
              <button className="btn new-chat" onClick={() => startNewConversation()}>New Chat</button>
              {/* <button className="btn toggle-sidebar" onClick={() => setSidebarOpen(!sidebarOpen)}>{sidebarOpen ? '◀' : '▶'}</button> */}
            </div>
          </div>

          <div className="conversations-list">
            {conversationsLoading && <div className="conv-loading">Loading...</div>}
            {(!conversationsLoading && conversations.length === 0) && (
              <div className="empty-convos">No conversations yet — start a new chat</div>
            )}
            {conversations.map((c) => (
              <div key={c.conversation_id} className={`conversation-item ${activeConversationId === c.conversation_id ? 'active' : ''}`} onClick={() => loadConversation(c.conversation_id)}>
                <div className="conv-title">{c.first_snippet || c.last_snippet || 'New Conversation'}</div>
                <div className="conv-sub">{c.last_snippet ? `Latest: ${c.last_snippet}` : ''}</div>
                <div className="conv-meta">{new Date(c.last_timestamp || c.first_timestamp).toLocaleString()} · {c.count} msgs</div>
                <div className="conv-actions">
                  <button className="btn small" onClick={(e) => { e.stopPropagation(); if (window.confirm('Delete this conversation?')) deleteConversation(c.conversation_id); }}>Delete</button>
                </div>
              </div>
            ))}
          </div>
        </aside>

        {/* Messages Container */}
        <div className="messages-container" ref={messagesContainerRef}>
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

        {/* Floating sidebar toggle */}
        
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

        
      </div>

      {/* Browser Support Info */}
      <div className="footer">
        {!voiceSupported && (
          <p className="warning">
            ⚠️ Voice input requires Chrome/Edge browser and microphone permission. Use text input instead.
          </p>
        )}
      </div>
    </div>
  );
};

export default HealthcareChatbot;