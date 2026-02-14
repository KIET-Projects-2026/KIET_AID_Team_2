// ===================== HEALTHCARE CHATBOT - REACT FRONTEND (UPDATED) =====================
// Complete React component with PROPER voice input that sends to backend for AI processing
// Voice → Microphone → Record Audio → Send to Backend → AI Model → Response → Display

// ===================== 1. INSTALL DEPENDENCIES =====================
// npm install axios react-icons

// ===================== 2. HEALTHCARECHATBOT.JSX (COMPLETE WORKING VERSION) =====================
import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { FiMic, FiSend, FiTrash2, FiVolume2, FiStopCircle, FiUser, FiMenu, FiChevronLeft, FiX, FiMicOff } from 'react-icons/fi';
import { toast } from 'react-toastify';
import './HealthcareChatbot.css';
import {
  buildUserContext,
  generateContextualSystemPrompt,
  getPersonalizedTips,
  generateWellnessReminder,
  generateFollowUpQuestions,
  getQuickActions,
  getEncouragementMessage,
  formatTipsSection,
  enhanceResponseWithContext,
  PersonalizedInsights,
} from './features';

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
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voiceSupported, setVoiceSupported] = useState(true);

  // ========== VOICE ASSISTANT MODE (Google Assistant-like) ==========
  const [voiceAssistantActive, setVoiceAssistantActive] = useState(false);
  const [assistantState, setAssistantState] = useState('idle'); // idle | listening | processing | speaking
  const [assistantTranscript, setAssistantTranscript] = useState('');
  const [assistantResponse, setAssistantResponse] = useState('');
  const [autoSpeak, setAutoSpeak] = useState(true); // auto-read bot replies
  const [continuousMode, setContinuousMode] = useState(true); // auto-listen after bot speaks
  const [audioLevel, setAudioLevel] = useState(0); // for visualizer animation
  const [selectedVoice, setSelectedVoice] = useState(null);

  // Profile menu (header) state + outside-click handler
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);
  const profileMenuRef = useRef(null);

  useEffect(() => {
    if (!profileMenuOpen) return;
    const onDocClick = (e) => {
      if (profileMenuRef.current && !profileMenuRef.current.contains(e.target)) {
        setProfileMenuOpen(false);
      }
    };
    const onKey = (e) => { if (e.key === 'Escape') setProfileMenuOpen(false); };
    document.addEventListener('mousedown', onDocClick);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDocClick);
      document.removeEventListener('keydown', onKey);
    };
  }, [profileMenuOpen]);

  // ========== WAKE WORD DETECTION ("Hey Voicebot") ==========
  const [wakeWordEnabled, setWakeWordEnabled] = useState(true); // background listening for wake word
  const [wakeWordListening, setWakeWordListening] = useState(false); // UI indicator

  // ========== UNIQUE FEATURES STATE ==========
  // 1. Emergency SOS Detection
  const [emergencyDetected, setEmergencyDetected] = useState(false);
  const [emergencyType, setEmergencyType] = useState('');
  const [showEmergencyEmailConfirm, setShowEmergencyEmailConfirm] = useState(false);
  const [emergencyEmailInput, setEmergencyEmailInput] = useState('');
  const [sendingEmergencyEmail, setSendingEmergencyEmail] = useState(false);
  const [autoSendNext, setAutoSendNext] = useState(false);
  const sentEmergencyRef = useRef(false); // prevents duplicate auto-sends during the same session



  // 3. Health Tips Carousel


  // Initialize WebSocket connection for real-time tips and chat (reconnects handled simply)
  useEffect(() => {
    const wsBase = API_BASE_URL.replace(/^http/, 'ws');
    const wsUrl = `${wsBase}/ws/chat`;

    let ws = null;
    try {
      ws = new WebSocket(wsUrl);
      ws.onopen = () => {
        console.info('WebSocket connected for tips');
        setTipsSource('server');
      };

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          console.debug('WS message:', msg);
          if ((msg.type && msg.type === 'tips') || msg.tips) {
            const tipsArr = Array.isArray(msg.tips) ? msg.tips : [];
            setServerTips(tipsArr);
            setCurrentTipIndex(0);
            setTipsSource('server');
            try { toast.success('Conversation tips received'); } catch (_) {}
            console.info('Tips updated from server:', tipsArr);
          }
        } catch (e) {
          console.warn('Failed to parse WS message', e);
        }
      };

      ws.onclose = () => {
        console.warn('WebSocket closed');
        setTipsSource('local');
      };

      ws.onerror = (e) => {
        console.error('WebSocket error', e);
        setTipsSource('local');
      };

      wsRef.current = ws;
    } catch (e) {
      console.warn('Failed to open WebSocket for tips', e);
      setTipsSource('local');
    }

    return () => {
      try {
        wsRef.current?.close();
      } catch (e) {}
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Helper: send tips request to server (WS) or fallback to REST/local DB
  const sendTipsRequest = async (overrideContext = {}, messagesOverride = null) => {
      const user_context = {
      topics: Object.fromEntries(getDashboardStats().topTopics || [])
    };

    // use provided messagesOverride (full conversation) or default to last 10
    const recent = (messagesOverride && Array.isArray(messagesOverride)
      ? messagesOverride.map(m => ({ type: m.type, text: m.text }))
      : messages.slice(-10).map(m => ({ type: m.type, text: m.text })));

    const payload = { type: 'tips_request', user_context: { ...user_context, ...overrideContext }, messages: recent };

    // notify user
    try { toast.info('Generating tips for your conversation...'); } catch {}

    // Try WS if connected
    if (wsRef.current && wsRef.current.readyState === 1) {
      try {
        wsRef.current.send(JSON.stringify(payload));
        return;
      } catch (e) {
        console.warn('WS send failed, falling back', e);
      }
    }

    // Fallback to REST generative endpoint
    try {
      const resp = await axios.post(`${API_BASE_URL}/api/generate/tips`, { user_context: payload.user_context, messages: payload.messages });
      if (resp.data?.status === 'success' && Array.isArray(resp.data.tips)) {
        setServerTips(resp.data.tips);
        setTipsSource('server-rest');
        setCurrentTipIndex(0);
        return;
      }
    } catch (e) {
      // ignore; we'll fallback locally
      console.warn('REST generate tips failed', e);
    }

    // Local fallback using tips DB
    try {
      const local = HEALTH_TIPS.map(t => t.tip);
      setServerTips(local);
      setTipsSource('local');
      setCurrentTipIndex(0);
    } catch (e) {
      console.warn('Local tips fallback failed', e);
    }
  }; 



  // 5. Chat Export
  const [exportingPdf, setExportingPdf] = useState(false);

  // 6. Health Dashboard
  const [showDashboard, setShowDashboard] = useState(false);

  // ========== REFS ==========
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);
  const speechRecognitionRef = useRef(null);
  const voiceAssistantActiveRef = useRef(false);
  const continuousModeRef = useRef(true);
  const wakeWordRecognitionRef = useRef(null);
  const wakeWordRunningRef = useRef(false); // guard against duplicate starts
  const wakeWordEnabledRef = useRef(true);
  const fetchConversationsInProgressRef = useRef(false); // guard against concurrent fetches
  const processQueryInProgressRef = useRef(false); // guard against concurrent voice queries

  // Mirror assistant state in a ref for cross-callback checks
  const assistantStateRef = useRef(assistantState);

  // Keep refs in sync with state for use inside callbacks
  useEffect(() => { voiceAssistantActiveRef.current = voiceAssistantActive; }, [voiceAssistantActive]);
  useEffect(() => { continuousModeRef.current = continuousMode; }, [continuousMode]);
  useEffect(() => { wakeWordEnabledRef.current = wakeWordEnabled; }, [wakeWordEnabled]);
  useEffect(() => { assistantStateRef.current = assistantState; }, [assistantState]);

  // ========== SELECT BEST TTS VOICE (prefer Google / natural voices) ==========
  useEffect(() => {
    const pickBestVoice = () => {
      const voices = window.speechSynthesis?.getVoices() || [];
      if (!voices.length) return;
      // Priority: Google US English > Google UK English > any English female > first English > default
      const preferred = [
        v => /google us english/i.test(v.name),
        v => /google uk english/i.test(v.name),
        v => v.lang.startsWith('en') && /female|samantha|zira|karen/i.test(v.name),
        v => v.lang.startsWith('en'),
      ];
      for (const test of preferred) {
        const match = voices.find(test);
        if (match) { setSelectedVoice(match); return; }
      }
      setSelectedVoice(voices[0]);
    };
    pickBestVoice();
    window.speechSynthesis?.addEventListener('voiceschanged', pickBestVoice);
    return () => window.speechSynthesis?.removeEventListener('voiceschanged', pickBestVoice);
  }, []);

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
    // Always scroll to the bottom when messages change so the latest reply is visible.
    // If the user has manually scrolled up we still jump to the bottom on new incoming messages
    // (this behavior mirrors most chat apps — change if you prefer a 'sticky' read position).
    const container = messagesContainerRef.current;
    try {
      if (messagesEndRef.current) {
        // prefer smooth scrolling for better UX
        messagesEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
      } else if (container) {
        container.scrollTop = container.scrollHeight;
      }
    } catch (e) {
      // fallback to direct assignment if browser doesn't support smooth
      if (container) container.scrollTop = container.scrollHeight;
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

  // ========== SPEECH SYNTHESIS (Text-to-Speech) - Enhanced for Assistant Mode ==========
  const speakResponse = useCallback((text, options = {}) => {
    if (!window.speechSynthesis) {
      toast.error('Speech synthesis not supported in this browser');
      return Promise.resolve();
    }

    window.speechSynthesis.cancel();

    return new Promise((resolve) => {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = options.rate || 1.0;
      utterance.pitch = options.pitch || 1.0;
      utterance.volume = options.volume || 1;

      // Use the selected best voice
      if (selectedVoice) {
        utterance.voice = selectedVoice;
      }

      utterance.onstart = () => {
        setIsSpeaking(true);
        if (voiceAssistantActiveRef.current) {
          setAssistantState('speaking');
        }
      };

      utterance.onend = () => {
        setIsSpeaking(false);
        if (voiceAssistantActiveRef.current) {
          setAssistantState('idle');
          // In continuous mode, auto-start listening again after bot finishes
          if (continuousModeRef.current) {
            setTimeout(() => {
              if (voiceAssistantActiveRef.current) {
                startAssistantListening();
              }
            }, 600);
          }
        }
        resolve();
      };

      utterance.onerror = (e) => {
        console.error('TTS error:', e);
        setIsSpeaking(false);
        if (voiceAssistantActiveRef.current) {
          setAssistantState('idle');
        }
        resolve();
      };

      window.speechSynthesis.speak(utterance);
    });
  }, [selectedVoice]);

  // ========== VOICE ASSISTANT - Speech Recognition (Live Transcription) ==========
  const startAssistantListening = useCallback(() => {
    if (!voiceAssistantActiveRef.current) return;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      toast.error('Speech recognition not supported. Use Chrome or Edge.');
      return;
    }

    // Stop any existing recognition
    if (speechRecognitionRef.current) {
      try { speechRecognitionRef.current.abort(); } catch (_) {}
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      setAssistantState('listening');
      setAssistantTranscript('');
      startAudioVisualizer();
    };

    recognition.onresult = (event) => {
      let interim = '';
      let final = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          final += transcript;
        } else {
          interim += transcript;
        }
      }
      setAssistantTranscript(final || interim);
    };

    recognition.onend = () => {
      stopAudioVisualizer();
      console.log('🎤 Voice assistant: recognition.onend fired');
      // Process the final transcript ONCE
      setAssistantTranscript((currentTranscript) => {
        // Only process if there's actual speech and we're not already processing
        if (currentTranscript && currentTranscript.trim() && voiceAssistantActiveRef.current) {
          console.log('📤 Processing transcript:', currentTranscript);
          processAssistantQuery(currentTranscript.trim());
        } else if (voiceAssistantActiveRef.current) {
          console.log('🤐 No speech detected, going idle');
          setAssistantState('idle');
        }
        // Clear transcript after processing to prevent re-processing
        return '';
      });
    };

    recognition.onerror = (event) => {
      stopAudioVisualizer();
      console.warn('Recognition error:', event.error);
      if (event.error === 'no-speech') {
        // No speech detected — go idle, allow user to tap again
        setAssistantState('idle');
      } else if (event.error === 'not-allowed') {
        toast.error('Microphone permission denied.');
        setVoiceAssistantActive(false);
        setAssistantState('idle');
      } else {
        setAssistantState('idle');
      }
    };

    speechRecognitionRef.current = recognition;
    recognition.start();
  }, []);

  const stopAssistantListening = useCallback(() => {
    if (speechRecognitionRef.current) {
      try { speechRecognitionRef.current.stop(); } catch (_) {}
    }
    stopAudioVisualizer();
  }, []);

  // ========== VOICE ASSISTANT - Process the spoken query ==========
  const processAssistantQuery = async (query) => {
    // Guard: prevent concurrent voice queries
    if (processQueryInProgressRef.current) {
      console.log('⏳ Voice query already in progress, skipping duplicate');
      return;
    }
    
    console.log('🎯 processAssistantQuery START:', query);
    console.log('📌 activeConversationId at start:', activeConversationId);
    console.log('📊 Available conversations:', conversations.length);
    processQueryInProgressRef.current = true;
    setAssistantState('processing');
    setAssistantResponse('');

    try {
      // Add user message to chat FIRST
      const userMessage = {
        id: generateId(),
        type: 'user',
        text: query,
        timestamp: new Date(),
        isVoice: true,
      };
      console.log('➕ Adding user message:', query);
      setMessages((prev) => {
        console.log(`📊 Messages before user: ${prev.length}`);
        return [...prev, userMessage];
      });

      // Get the conversation ID with smart fallback logic:
      // 1. Use activeConversationId if set (user selected a chat)
      // 2. Else use the first/most recent available conversation (chat is open)
      // 3. Else create a new one
      let convId = activeConversationId;
      
      if (!convId && conversations && conversations.length > 0) {
        // If no chat selected but chats exist, use the first (most recent) one
        convId = conversations[0].conversation_id;
        console.log('📂 No explicit selection - using first available chat:', convId);
        setActiveConversationId(convId);
        localStorage.setItem('activeConversationId', convId);
      } else if (!convId) {
        // Only create new conversation if no conversations exist at all
        console.log('⚠️ No chats available - creating new one');
        const createResp = await axios.post(`${API_BASE_URL}/api/chat/conversations`, {});
        
        if (createResp.data?.status === 'success' && createResp.data.conversation_id) {
          convId = createResp.data.conversation_id;
          console.log('✅ Created new conversation:', convId);
          setActiveConversationId(convId);
          localStorage.setItem('activeConversationId', convId);
        } else {
          throw new Error('Failed to create conversation');
        }
      } else {
        console.log('✅ Using selected conversation:', convId);
      }

      // Send message to API with the conversation ID
      console.log('📤 Sending to API - conversation:', convId, 'query:', query);
      const response = await axios.post(
        `${API_BASE_URL}/api/chat/text`,
        { text: query, conversation_id: convId },
        { timeout: 30000 }
      );

      if (response.data.status === 'success') {
        const botText = response.data.response;
        setAssistantResponse(botText);

        const botMessage = {
          id: generateId(),
          type: 'bot',
          text: botText,
          timestamp: new Date(),
          inputType: 'voice',
        };
        console.log('➕ Adding bot response:', botText.substring(0, 50) + '...');
        setMessages((prev) => {
          console.log(`📊 Messages before bot: ${prev.length}`);
          const updated = [...prev, botMessage];
          console.log(`📊 Messages after bot: ${updated.length}`);
          return updated;
        });

        // Refresh conversation list to update the sidebar
        console.log('🔄 Refreshing conversation list');
        await fetchConversations();

        // Speak the response
        await speakResponse(botText);
      } else {
        const errMsg = 'Sorry, I couldn\'t process that. Please try again.';
        setAssistantResponse(errMsg);
        setAssistantState('idle');
        await speakResponse(errMsg);
      }
    } catch (err) {
      console.error('❌ Assistant query error:', err);
      const errMsg = 'Sorry, something went wrong. Please try again.';
      setAssistantResponse(errMsg);
      setAssistantState('idle');
      await speakResponse(errMsg);
    } finally {
      console.log('🎯 processAssistantQuery END');
      processQueryInProgressRef.current = false;
    }
  };

  // ========== AUDIO VISUALIZER (for voice assistant animation) ==========
  const startAudioVisualizer = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = { analyser, audioContext, stream };

      const dataArray = new Uint8Array(analyser.frequencyBinCount);

      const updateLevel = () => {
        analyser.getByteFrequencyData(dataArray);
        const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
        setAudioLevel(Math.min(avg / 128, 1)); // normalize 0–1
        animationFrameRef.current = requestAnimationFrame(updateLevel);
      };
      updateLevel();
    } catch (err) {
      console.warn('Audio visualizer error:', err);
    }
  };

  const stopAudioVisualizer = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    if (analyserRef.current) {
      analyserRef.current.stream?.getTracks().forEach(t => t.stop());
      analyserRef.current.audioContext?.close().catch(() => {});
      analyserRef.current = null;
    }
    setAudioLevel(0);
  };

  // ========== VOICE ASSISTANT CONTROLS ==========
  const openVoiceAssistant = () => {
    // Stop wake word listener — assistant takes over the mic
    stopWakeWordListener();
    setVoiceAssistantActive(true);
    setAssistantState('idle');
    setAssistantTranscript('');
    setAssistantResponse('');
    // Small delay then start listening
    setTimeout(() => {
      startAssistantListening();
    }, 400);
  };

  const closeVoiceAssistant = () => {
    setVoiceAssistantActive(false);
    setAssistantState('idle');
    setAssistantTranscript('');
    setAssistantResponse('');
    stopAssistantListening();
    stopAudioVisualizer();
    window.speechSynthesis?.cancel();
    setIsSpeaking(false);
    // Restart wake word listener if enabled
    if (wakeWordEnabledRef.current) {
      setTimeout(() => startWakeWordListener(), 500);
    }
  };

  const handleAssistantTap = () => {
    if (assistantState === 'listening') {
      // Force stop listening and process what we have
      stopAssistantListening();
    } else if (assistantState === 'speaking') {
      // Stop speaking and go idle
      window.speechSynthesis?.cancel();
      setIsSpeaking(false);
      setAssistantState('idle');
    } else if (assistantState === 'idle') {
      // Start listening
      startAssistantListening();
    }
    // If processing, do nothing — wait for response
  };

  // ========== WAKE WORD LISTENER (background "voicebot" detection) ==========
  const startWakeWordListener = useCallback(() => {
    // Guard: don't start if already running, disabled, or assistant is open
    if (wakeWordRunningRef.current) {
      console.log('🛑 Wake word: Already running, skipping duplicate start');
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.log('Wake word: SpeechRecognition not supported');
      return;
    }
    // If the assistant is actively listening/processing, don't start the wake-word recognizer
    // but allow the wake-word to run when the assistant modal is open AND idle so a user
    // can say the wake word to start listening without tapping the orb.
    if (assistantStateRef.current === 'listening' || assistantStateRef.current === 'processing') {
      console.log('🛑 Wake word: Assistant busy, skipping start');
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = false;  // Stop after each utterance (more reliable)
    recognition.interimResults = false;  // Only final results
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

    // Wake word to match
    const WAKE_WORD = 'voice';
    let wakeWordTriggered = false;
    let listeningTimeout;

    const isWakeWordDetected = (transcript) => {
      const clean = transcript.toLowerCase().trim().replace(/[^\w\s]/g, '');
      console.log('🔎 Wake word heard:', clean);
      return clean.includes(WAKE_WORD);
    };

    recognition.onstart = () => {
      wakeWordRunningRef.current = true;
      setWakeWordListening(true);
      console.log('✅ Wake word listener STARTED — say "Voice" (10 seconds to speak)');
      
      // Auto-stop listening after 10 seconds to avoid hanging
      listeningTimeout = setTimeout(() => {
        if (!wakeWordTriggered) {
          console.log('⏱️ Wake word: 10 second timeout reached, stopping...');
          try { recognition.stop(); } catch (_) {}
        }
      }, 10000);
    };

    recognition.onresult = (event) => {
      // Log ALL detected speech in real-time
      console.log(`📊 Speech detection event (${event.results.length} results total)`);
      
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        const isFinal = result.isFinal;
        
        // Log ALL alternatives (not just the first one)
        for (let j = 0; j < result.length; j++) {
          const alt = result[j];
          const transcript = alt.transcript.trim();
          const confidence = (alt.confidence * 100).toFixed(1);
          
          if (transcript) {
            const status = isFinal ? '✅ FINAL' : '🔄 INTERIM';
            console.log(`${status}: "${transcript}" (${confidence}% confidence)`);
          }
        }
        
        // Check first alternative for wake word
        const mainTranscript = result[0].transcript;
        if (isWakeWordDetected(mainTranscript)) {
          wakeWordTriggered = true;
          wakeWordRunningRef.current = false;
          if (listeningTimeout) clearTimeout(listeningTimeout);
          try { recognition.stop(); } catch (_) {}
          setWakeWordListening(false);

          // Show what was detected
          toast.success(`🎯 Detected: "${mainTranscript.trim()}"`, { autoClose: 1200 });

          // If assistant modal is open, trigger assistant listening; otherwise open the assistant
          if (voiceAssistantActiveRef.current) {
            console.log('🚀 Wake word DETECTED while assistant open — starting assistant listening');
            toast.info('🎙️ Activating assistant (listening)...', { autoClose: 1200 });
            // only start assistant listening if it's idle
            if (assistantStateRef.current === 'idle') {
              setTimeout(() => startAssistantListening(), 200);
            }
          } else {
            console.log('🚀 Wake word DETECTED! Opening assistant...');
            toast.info('🎙️ "Voice" detected! Activating...', { autoClose: 1200 });
            setTimeout(() => openVoiceAssistant(), 300);
          }

          return;
        }
      }
    };

    recognition.onend = () => {
      if (listeningTimeout) clearTimeout(listeningTimeout);
      wakeWordRunningRef.current = false;
      setWakeWordListening(false);

      // If we triggered a wake-word, do not restart immediately — assistant will handle mic usage.
      if (wakeWordTriggered) return;

      console.log('⏹️ Wake word: Listening ended, will restart soon (if allowed)');

      // Auto-restart when wake-word is still enabled and assistant is not actively listening/processing
      if (wakeWordEnabledRef.current && assistantStateRef.current !== 'listening' && assistantStateRef.current !== 'processing') {
        setTimeout(() => {
          if (wakeWordEnabledRef.current && assistantStateRef.current !== 'listening' && !wakeWordRunningRef.current) {
            startWakeWordListener();
          }
        }, 500);
      }
    };

    recognition.onerror = (event) => {
      if (listeningTimeout) clearTimeout(listeningTimeout);
      wakeWordRunningRef.current = false;
      setWakeWordListening(false);
      console.warn('❌ Wake word error:', event.error);

      if (event.error === 'not-allowed') {
        setWakeWordEnabled(false);
        toast.error('🔒 Microphone permission denied. Wake word disabled.');
        return;
      }
      if (event.error === 'aborted') return; // intentional, onend handles restart

      // Log detailed error info for debugging
      if (event.error === 'no-speech') {
        console.warn('Wake word: No speech detected (will retry)');
      } else if (event.error === 'network') {
        console.error('Wake word: Network error');
      }

      // For no-speech, network, etc. — retry if wake-word still enabled and assistant not actively listening
      if (wakeWordEnabledRef.current && assistantStateRef.current !== 'listening' && assistantStateRef.current !== 'processing') {
        const delay = event.error === 'no-speech' ? 500 : 1500;
        setTimeout(() => {
          if (wakeWordEnabledRef.current && assistantStateRef.current !== 'listening' && !wakeWordRunningRef.current) {
            startWakeWordListener();
          }
        }, delay);
      }
    };

    wakeWordRecognitionRef.current = recognition;
    try {
      recognition.start();
      console.log('📍 Wake word: start() called');
    } catch (err) {
      console.error('❌ Wake word start() exception:', err);
      wakeWordRunningRef.current = false;
      setWakeWordListening(false);
      setTimeout(() => {
        if (wakeWordEnabledRef.current && assistantStateRef.current !== 'listening' && !wakeWordRunningRef.current) {
          startWakeWordListener();
        }
      }, 1500);
    }
  }, []);

  const stopWakeWordListener = useCallback(() => {
    wakeWordRunningRef.current = false;
    if (wakeWordRecognitionRef.current) {
      try { wakeWordRecognitionRef.current.abort(); } catch (_) {}
      wakeWordRecognitionRef.current = null;
    }
    setWakeWordListening(false);
  }, []);

  // Auto-start wake word listener on mount / when enabled changes
  useEffect(() => {
    // Use a small delay to avoid double-start from React strict mode or fast re-renders
    let timer;

    // Start wake-word listener whenever enabled as long as the assistant is NOT actively listening/processing.
    if (wakeWordEnabled && assistantState !== 'listening' && assistantState !== 'processing') {
      timer = setTimeout(() => {
        if (!wakeWordRunningRef.current) {
          startWakeWordListener();
        }
      }, 600);
    } else {
      stopWakeWordListener();
    }

    return () => {
      if (timer) clearTimeout(timer);
      stopWakeWordListener();
    };
  }, [wakeWordEnabled, assistantState, startWakeWordListener, stopWakeWordListener]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopAudioVisualizer();
      stopWakeWordListener();
      if (speechRecognitionRef.current) {
        try { speechRecognitionRef.current.abort(); } catch (_) {}
      }
      window.speechSynthesis?.cancel();
    };
  }, []);

  // ========== API CALLS ==========

  // Fetch user's conversation list
  const fetchConversations = async () => {
    // Prevent concurrent fetch calls
    if (fetchConversationsInProgressRef.current) {
      console.log('⏳ Conversation fetch already in progress, skipping duplicate');
      return;
    }
    
    fetchConversationsInProgressRef.current = true;
    setConversationsLoading(true);
    try {
      const resp = await axios.get(`${API_BASE_URL}/api/chat/conversations`);
      if (resp.data?.status === 'success') {
        const convList = resp.data.data || [];
        console.log(`📊 Fetched ${convList.length} conversations`);
        setConversations(convList);
      }
    } catch (e) {
      console.warn('Failed to load conversations:', e);
    } finally {
      setConversationsLoading(false);
      fetchConversationsInProgressRef.current = false;
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
      // Build user context for personalized responses
      const userContext = buildUserContext(null, null, messages, []);
      const systemPrompt = generateContextualSystemPrompt(userContext);
      const personalizedTips = getPersonalizedTips(userContext);
      const wellnessReminder = generateWellnessReminder(userContext);
      const encouragement = getEncouragementMessage(userContext);
      const followUpQuestions = generateFollowUpQuestions(userContext);
      const quickActions = getQuickActions(userContext);

      // Tips feature removed — no personalized or AI-suggested tips will be attached.
      const mergedTips = [];

      // Add user message to chat
      const userMessage = {
        id: generateId(),
        type: 'user',
        text: text.trim(),
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMessage]);
      setInputText('');

      console.log('📤 Sending text to backend with context:', { context: userContext });

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

      // Send to API with user context for personalized response
      const response = await axios.post(
        `${API_BASE_URL}/api/chat/text`,
        { 
          text: text.trim(), 
          conversation_id: convId,
          user_context: {
            symptoms: userContext.symptoms,
            system_prompt: systemPrompt,
          },
        },
        { timeout: 30000 }
      );

      console.log('📥 Received response:', response.data);

      if (response.data.status === 'success') {
        let responseText = response.data.response;
        
        // Enhance response with context-aware suggestions
        const enhancedResponse = enhanceResponseWithContext(responseText, userContext);
        const tipsSection = formatTipsSection(personalizedTips);

        // Add bot response with enhanced content
        const botMessage = {
          id: generateId(),
          type: 'bot',
          text: enhancedResponse,
          timestamp: new Date(),
          inputType: 'text',
          context: {
            wellnessReminder,
            encouragement,
            followUpQuestions,
            quickActions,
          },
        };

        setMessages((prev) => [...prev, botMessage]);
        toast.success('✅ Response received!');

        // Auto-speak the response if enabled
        if (autoSpeak) {
          speakResponse(enhancedResponse);
        }

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

  // Clear all conversations
  const clearAllChats = async () => {
    if (!conversations || conversations.length === 0) {
      toast.info('ℹ️ No chats to clear');
      return;
    }

    if (!window.confirm(`Are you sure? This will delete all ${conversations.length} chats. This cannot be undone.`)) {
      return;
    }

    setIsLoading(true);
    let deleted = 0;
    let failed = 0;

    try {
      for (const conv of conversations) {
        try {
          const resp = await axios.delete(`${API_BASE_URL}/api/chat/conversations/${conv.conversation_id}`);
          if (resp.data?.status === 'success') {
            deleted++;
          } else {
            failed++;
          }
        } catch (err) {
          console.warn('Failed to delete conversation:', conv.conversation_id);
          failed++;
        }
      }

      // Clear UI
      setActiveConversationId(null);
      localStorage.removeItem('activeConversationId');
      setMessages([
        { id: generateId(), type: 'bot', text: '🗑️ All chats cleared. Start a new conversation!', timestamp: new Date() }
      ]);
      setConversations([]);

      if (failed === 0) {
        toast.success(`✅ Deleted all ${deleted} chats`);
      } else {
        toast.warning(`⚠️ Deleted ${deleted} chats (${failed} failed)`);
      }
    } catch (err) {
      console.error('❌ Error clearing chats:', err);
      toast.error('Failed to clear all chats');
    } finally {
      setIsLoading(false);
    }
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

  // ===================== UNIQUE FEATURES =====================

  // ========== 1. EMERGENCY SOS DETECTION ==========
  const EMERGENCY_KEYWORDS = [
    'chest pain', 'heart attack', 'can\'t breathe', 'cannot breathe', 'difficulty breathing',
    'stroke', 'seizure', 'unconscious', 'bleeding heavily', 'severe bleeding',
    'choking', 'anaphylaxis', 'allergic reaction severe', 'suicide', 'suicidal',
    'overdose', 'poisoning', 'severe burn', 'drowning', 'collapsed',
    'head injury', 'not breathing', 'cardiac arrest', 'blood clot',
    'paralysis', 'fainting', 'severe pain chest', 'convulsions'
  ];

  const checkEmergency = useCallback((text) => {
    const lower = text.toLowerCase();
    for (const keyword of EMERGENCY_KEYWORDS) {
      if (lower.includes(keyword)) {
        setEmergencyDetected(true);
        setEmergencyType(keyword);
        return true;
      }
    }
    setEmergencyDetected(false);
    setEmergencyType('');
    return false;
  }, []);

  const dismissEmergency = () => {
    setEmergencyDetected(false);
    setEmergencyType('');
  };

  // ========== 3. HEALTH TIPS CAROUSEL ==========
  const HEALTH_TIPS = [
    { icon: '💧', tip: 'Drink at least 8 glasses of water daily to stay hydrated.', category: 'Hydration' },
    { icon: '🏃', tip: '30 minutes of moderate exercise daily can reduce heart disease risk by 35%.', category: 'Exercise' },
    { icon: '😴', tip: 'Adults need 7-9 hours of sleep. Poor sleep increases diabetes risk.', category: 'Sleep' },
    { icon: '🥗', tip: 'Eating 5 servings of fruits and vegetables daily boosts immunity.', category: 'Nutrition' },
    { icon: '🧘', tip: 'Just 10 minutes of daily meditation can reduce stress hormones by 20%.', category: 'Mental Health' },
    { icon: '🫁', tip: 'Deep breathing exercises can lower blood pressure in just 5 minutes.', category: 'Breathing' },
    { icon: '👁️', tip: 'Follow the 20-20-20 rule: every 20 min, look 20 ft away for 20 seconds.', category: 'Eye Care' },
    { icon: '🦷', tip: 'Brush twice daily and floss once. Gum disease links to heart disease.', category: 'Dental' },
    { icon: '☀️', tip: '15 minutes of sunlight daily helps your body produce Vitamin D.', category: 'Wellness' },
    { icon: '🧠', tip: 'Learning something new daily strengthens neural connections.', category: 'Brain Health' },
    { icon: '🫀', tip: 'Laughing for 15 minutes burns up to 40 calories and boosts heart health.', category: 'Heart Health' },
    { icon: '🍎', tip: 'An apple a day provides 14% of daily Vitamin C. Eat the skin for fiber!', category: 'Nutrition' },
  ];

  // Tip state + WebSocket ref (was removed earlier; restore to prevent runtime errors)
  const [serverTips, setServerTips] = useState([]);
  const [currentTipIndex, setCurrentTipIndex] = useState(0);
  const [tipsSource, setTipsSource] = useState('local');
  const wsRef = useRef(null);

  // Tip rotation uses the active tips list (server-provided when available)
  useEffect(() => {
    const getLength = () => (serverTips && serverTips.length ? serverTips.length : HEALTH_TIPS.length);
    // rotate index using current active length
    const tipTimer = setInterval(() => {
      const len = getLength();
      setCurrentTipIndex(prev => (prev + 1) % Math.max(1, len));
    }, 8000); // rotate every 8 seconds
    return () => clearInterval(tipTimer);
  }, [serverTips]);



  // ========== 5. EXPORT CHAT AS PDF ==========
  const exportChatPdf = () => {
    setExportingPdf(true);
    try {
      const chatContent = messages.filter(m => m.text).map(m => {
        const time = m.timestamp.toLocaleString();
        const sender = m.type === 'user' ? '🧑 You' : '🤖 Bot';
        return `${sender} [${time}]\n${m.text}\n`;
      }).join('\n' + '─'.repeat(50) + '\n\n');

      const report = `
╔══════════════════════════════════════════════════════╗
║          HEALTHCARE CHATBOT - MEDICAL REPORT         ║
╚══════════════════════════════════════════════════════╝

Patient: ${currentUser?.username || 'Anonymous'}
Date: ${new Date().toLocaleDateString()}
Time: ${new Date().toLocaleTimeString()}
Total Messages: ${messages.length}


════════════════════════════════════════════════════════
                    CONVERSATION LOG
════════════════════════════════════════════════════════

${chatContent}

════════════════════════════════════════════════════════
                      DISCLAIMER
════════════════════════════════════════════════════════

This report is generated by an AI-powered healthcare
chatbot. It is NOT a substitute for professional medical
advice, diagnosis, or treatment. Always seek the advice
of your physician or other qualified health provider.

Generated on: ${new Date().toLocaleString()}
Powered by: Healthcare AI Chatbot v2.0
      `.trim();

      // Create downloadable text file (works everywhere, no external library needed)
      const blob = new Blob([report], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Healthcare_Report_${new Date().toISOString().split('T')[0]}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast.success('📄 Medical report exported!');
    } catch (err) {
      console.error('Export failed:', err);
      toast.error('Failed to export report');
    } finally {
      setExportingPdf(false);
    }
  };

  // ========== 6. HEALTH DASHBOARD STATS ==========
  const getDashboardStats = () => {
    const userMsgs = messages.filter(m => m.type === 'user');
    const botMsgs = messages.filter(m => m.type === 'bot');
    const voiceMsgs = messages.filter(m => m.isVoice || m.inputType === 'voice');

    // Extract topic keywords from user messages
    const words = userMsgs.map(m => m.text.toLowerCase()).join(' ').split(/\s+/);
    const healthKeywords = ['pain', 'fever', 'headache', 'cold', 'cough', 'diabetes', 'heart', 'blood',
      'pressure', 'anxiety', 'depression', 'sleep', 'diet', 'exercise', 'vitamin', 'infection',
      'allergy', 'breathing', 'stomach', 'skin', 'cancer', 'stress', 'medication', 'treatment'];
    const topicCounts = {};
    healthKeywords.forEach(kw => {
      const count = words.filter(w => w.includes(kw)).length;
      if (count > 0) topicCounts[kw] = count;
    });
    const topTopics = Object.entries(topicCounts).sort((a, b) => b[1] - a[1]).slice(0, 6);

    return {
      totalMessages: messages.length,
      userMessages: userMsgs.length,
      botMessages: botMsgs.length,
      voiceMessages: voiceMsgs.length,
      totalConversations: conversations.length,
      topTopics,
    };
  };

  // Check emergency on every user message
  useEffect(() => {
    const lastMsg = messages[messages.length - 1];
    if (lastMsg && lastMsg.type === 'user') {
      checkEmergency(lastMsg.text);
    }
  }, [messages, checkEmergency]);

  // Auto-send emergency email if user opted in (runs when emergencyDetected changes)
  useEffect(() => {
    if (!emergencyDetected) {
      sentEmergencyRef.current = false; // reset for next session
      return;
    }
    // Only auto-send once per detection
    if (sentEmergencyRef.current) return;

    const contact = currentUser?.emergencyEmail;
    const auto = !!currentUser?.emergencyAutoSend;
    if (auto && contact) {
      (async () => {
        try {
          sentEmergencyRef.current = true;
          setSendingEmergencyEmail(true);
          const convId = activeConversationId || (conversations[0] && conversations[0].conversation_id) || null;
          const recent = messages.slice(-50).map(m => ({ type: m.type, text: m.text, timestamp: m.timestamp }));
          const resp = await axios.post(`${API_BASE_URL}/api/notify/emergency`, {
            conversation_id: convId,
            contact_email: contact,
            alert_message: emergencyType,
            messages: recent,
          });
          toast.success(resp.data?.message || 'Emergency email automatically queued');
          setShowEmergencyEmailConfirm(false);
        } catch (err) {
          console.error('❌ Auto emergency send failed', err);
          toast.error(err.response?.data?.detail || err.message || 'Failed to auto-send emergency email');
        } finally {
          setSendingEmergencyEmail(false);
        }
      })();
    }
  }, [emergencyDetected, currentUser, activeConversationId, conversations, messages, emergencyType]);

  // ========== RENDER ==========
  return (
    <>
      <div className={`chatbot-container ${voiceAssistantActive ? 'va-open' : ''}`}>
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
        <div className="chatbot-header-actions">
          <button className="menu-btn" title="Menu" aria-expanded={sidebarOpen} onClick={() => setSidebarOpen(!sidebarOpen)}>
            <FiMenu />
          </button>
        </div>
        <div className="chatbot-title">
          <span className="chatbot-logo"></span> Healthcare Chatbot
          <span className="chatbot-subtitle">Voice & Text Support with AI Model</span>
        </div>
        <div className="chatbot-header-actions">
          
          <span className="user-badge">{currentUser?.username || 'User'}</span>
          <div className="profile-menu-wrapper" ref={profileMenuRef}>
            <button
              className="btn profile-icon"
              title={profileMenuOpen ? 'Close menu' : 'Open profile menu'}
              aria-haspopup="menu"
              aria-expanded={profileMenuOpen}
              onClick={() => setProfileMenuOpen(p => !p)}
              aria-label="Open profile menu"
            >
              {currentUser?.username ? currentUser.username.charAt(0).toUpperCase() : <FiUser />}
            </button>

            <div className={`profile-menu ${profileMenuOpen ? 'open' : ''}`} role="menu" aria-hidden={!profileMenuOpen}>
              <button className="profile-menu__item" role="menuitem" onClick={() => { window.dispatchEvent(new CustomEvent('openProfile')); setProfileMenuOpen(false); }}>Profile</button>
              <button className="profile-menu__item" role="menuitem" onClick={() => { setShowDashboard(true); setProfileMenuOpen(false); }}>Dashboard</button>
              <div className="profile-menu__divider" />
              <button className="profile-menu__item" role="menuitem" onClick={() => { if (typeof onLogout === 'function') onLogout(); setProfileMenuOpen(false); }}>Logout</button>
            </div>
          </div>
        </div>
      </div>

      {/* ========== EMERGENCY SOS BANNER ========== */}
      {emergencyDetected && (
        <div className="emergency-banner">
          <div className="emergency-content">
            <span className="emergency-icon">🚨</span>
            <div className="emergency-text">
              <strong>EMERGENCY DETECTED: {emergencyType.toUpperCase()}</strong>
              <p>If this is a medical emergency, please call emergency services immediately!</p>
            </div>
            <div className="emergency-numbers">
              <a href="tel:911" className="emergency-call">📞 911 (US)</a>
              <a href="tel:108" className="emergency-call">📞 108 (India)</a>
              <a href="tel:112" className="emergency-call">📞 112 (EU)</a>
            </div>

            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <button
                className="toolbar-btn"
                onClick={() => {
                  setShowEmergencyEmailConfirm(true);
                  setEmergencyEmailInput(currentUser?.emergencyEmail || '');
                  setAutoSendNext(!!currentUser?.emergencyAutoSend);
                }}
                title="Send email to emergency contact"
              >
                ✉️ Send Email
              </button>

              <button
                className="toolbar-btn"
                onClick={async () => {
                  // Quick send without confirmation (uses saved emergencyEmail).
                  // If missing, try to refresh profile from server to pick up recent changes.
                  let contact = currentUser?.emergencyEmail;
                  if (!contact) {
                    try {
                      const token = localStorage.getItem('token');
                      if (!token) {
                        toast.error('Not authenticated. Please log in to use Quick Send.');
                        return;
                      }

                      const resp = await axios.get(`${API_BASE_URL}/api/auth/me`, { headers: { Authorization: `Bearer ${token}` } });
                      if (resp.data?.user?.emergencyEmail) {
                        contact = resp.data.user.emergencyEmail;
                        // Persist and notify other components
                        try {
                          localStorage.setItem('user', JSON.stringify(resp.data.user));
                          window.dispatchEvent(new CustomEvent('profileUpdated', { detail: resp.data.user }));
                        } catch (e) {
                          console.warn('Failed to persist refreshed profile locally', e);
                        }
                      }
                    } catch (e) {
                      console.warn('Failed to refresh user profile before quick send', e);
                      // If the request was unauthorized, let the user know
                      if (e?.response?.status === 401) {
                        toast.error('Session expired. Please sign in again to send emergency emails.');
                        return;
                      }
                    }
                  }

                  if (!contact) {
                    toast.error('No emergency email configured. Please set it in your profile or use Send Email to enter one.');
                    return;
                  }

                  try {
                    setSendingEmergencyEmail(true);
                    const convId = activeConversationId || (conversations[0] && conversations[0].conversation_id) || null;
                    const recent = messages.slice(-50).map(m => ({ type: m.type, text: m.text, timestamp: m.timestamp }));
                    const resp = await axios.post(`${API_BASE_URL}/api/notify/emergency`, {
                      conversation_id: convId,
                      contact_email: contact,
                      alert_message: emergencyType,
                      messages: recent,
                    });
                    toast.success(resp.data?.message || 'Emergency email queued');
                    dismissEmergency();
                  } catch (err) {
                    console.error('❌ Quick emergency send failed', err);
                    toast.error(err.response?.data?.detail || err.message || 'Failed to send emergency email');
                  } finally {
                    setSendingEmergencyEmail(false);
                  }
                }}
                title="Quick send to saved emergency contact"
              >
                🚀 Send Now
              </button>

              <button className="emergency-dismiss" onClick={dismissEmergency}>✕</button>
            </div>

            {/* Confirmation Modal (Inline) */}
            {showEmergencyEmailConfirm && (
              <div className="emergency-email-confirm" role="dialog" aria-modal="true">
                <div className="confirm-panel">
                  <h4>Send email to emergency contact?</h4>
                  <p style={{ margin: '6px 0' }}><strong>Detected alert:</strong> {emergencyType}</p>
                  <label style={{ display: 'block', marginBottom: 8 }}>
                    Emergency contact email:
                    <input
                      type="email"
                      value={emergencyEmailInput}
                      onChange={(e) => setEmergencyEmailInput(e.target.value)}
                      placeholder="Enter contact email"
                      style={{ width: '100%', padding: '8px', marginTop: '6px' }}
                    />
                  </label>

                  <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 6 }}>
                    <input type="checkbox" checked={autoSendNext} onChange={(e) => setAutoSendNext(e.target.checked)} />
                    <span style={{ fontSize: 13 }}>Auto-send on future emergencies</span>
                  </label>

                  <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', marginTop: 8 }}>
                    <button className="toolbar-btn" onClick={() => setShowEmergencyEmailConfirm(false)}>Cancel</button>
                    <button
                      className="toolbar-btn"
                      onClick={async () => {
                        try {
                          setSendingEmergencyEmail(true);
                          const convId = activeConversationId || (conversations[0] && conversations[0].conversation_id) || null;
                          // include recent messages (last 50)
                          const recent = messages.slice(-50).map(m => ({ type: m.type, text: m.text, timestamp: m.timestamp }));

                          // If user checked auto-send, persist preference to profile
                          if (autoSendNext) {
                            try {
                              await axios.patch(`${API_BASE_URL}/api/auth/me`, { emergencyEmail: emergencyEmailInput, emergencyAutoSend: true });
                              // Update local currentUser (best-effort)
                              if (typeof currentUser === 'object') {
                                const updated = { ...currentUser, emergencyEmail: emergencyEmailInput, emergencyAutoSend: true };
                                try {
                                  localStorage.setItem('user', JSON.stringify(updated));
                                  window.dispatchEvent(new CustomEvent('profileUpdated', { detail: updated }));
                                } catch (e) {
                                  console.warn('Failed to persist emergency email to localStorage', e);
                                }
                              }
                            } catch (e) {
                              console.warn('Failed to save auto-send preference', e?.response?.data || e.message);
                            }
                          }

                          const resp = await axios.post(`${API_BASE_URL}/api/notify/emergency`, {
                            conversation_id: convId,
                            contact_email: emergencyEmailInput,
                            alert_message: emergencyType,
                            messages: recent,
                          });

                          toast.success(resp.data?.message || 'Emergency email queued');
                          setShowEmergencyEmailConfirm(false);
                        } catch (err) {
                          console.error('❌ Failed to send emergency email', err);
                          toast.error(err.response?.data?.detail || err.message || 'Failed to send emergency email');
                        } finally {
                          setSendingEmergencyEmail(false);
                        }
                      }}
                      disabled={!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(emergencyEmailInput) || sendingEmergencyEmail}
                    >
                      {sendingEmergencyEmail ? 'Sending...' : 'Send Email'}
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ========== FEATURE TOOLBAR (below header) ========== */}
      {/* <div className="feature-toolbar">


       
        <div className="toolbar-section tips-section">
          {(() => {
            const activeTipsList = serverTips && serverTips.length ? serverTips.map(t => ({ icon: '💡', tip: t, category: '' })) : HEALTH_TIPS;
            const active = activeTipsList[currentTipIndex % activeTipsList.length];
            return null;
          })()}
        </div>

      
        <div className="toolbar-section toolbar-actions">

          <button className="toolbar-btn" onClick={() => setShowDashboard(!showDashboard)} title="Health Dashboard">
            📊 Dashboard
          </button>
          <button className="toolbar-btn" onClick={exportChatPdf} disabled={exportingPdf || messages.length < 2} title="Export Report">
            📄 {exportingPdf ? 'Exporting...' : 'Export'}
          </button>
        </div>
      </div> */}

     
      {showDashboard && (() => {
        const stats = getDashboardStats();
        return (
          <div className="dashboard-panel">
            <div className="dashboard-header">
              <h3>📊 Health Dashboard</h3>
              <button className="dashboard-close" onClick={() => setShowDashboard(false)}>✕</button>
            </div>
            <div className="dashboard-grid">
              <div className="stat-card">
                <div className="stat-number">{stats.totalMessages}</div>
                <div className="stat-label">Total Messages</div>
              </div>
              <div className="stat-card">
                <div className="stat-number">{stats.totalConversations}</div>
                <div className="stat-label">Conversations</div>
              </div>
              <div className="stat-card">
                <div className="stat-number">{stats.voiceMessages}</div>
                <div className="stat-label">Voice Queries</div>
              </div>
              <div className="stat-card">
                <div className="stat-number">{stats.userMessages}</div>
                <div className="stat-label">Questions Asked</div>
              </div>
            </div>
            {stats.topTopics.length > 0 && (
              <div className="dashboard-topics">
                <h4>🔥 Your Top Health Topics</h4>
                <div className="topic-bars">
                  {stats.topTopics.map(([topic, count]) => (
                    <div key={topic} className="topic-bar-item">
                      <span className="topic-name">{topic}</span>
                      <div className="topic-bar">
                        <div className="topic-bar-fill" style={{ width: `${Math.min(100, (count / Math.max(...stats.topTopics.map(t => t[1]))) * 100)}%` }} />
                      </div>
                      <span className="topic-count">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </div>
        );
      })()}



      {/* Toast notifications are handled by ToastContainer in App */}

      {/* Backdrop for mobile sidebar overlay (click to dismiss) */}
      {sidebarOpen && (
        <div
          className="sidebar-backdrop"
          role="button"
          tabIndex={0}
          aria-label="Close chats overlay"
          onClick={() => setSidebarOpen(false)}
          onKeyDown={(e) => { if (e.key === 'Escape' || e.key === 'Enter' || e.key === ' ') setSidebarOpen(false); }}
        />
      )}

      <div className={`chat-grid ${sidebarOpen ? '' : 'sidebar-closed'}`}>
        {/* Sidebar - Conversations (on mobile it overlays when `open`) */}
        <aside className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`} aria-hidden={!sidebarOpen}>
          <div className="sidebar-header">
            <h3>Chats</h3>
            <div className="sidebar-actions">
              <button className="btn new-chat" onClick={() => { startNewConversation(); setSidebarOpen(false); }}>New Chat</button>
              {conversations.length > 0 && (
                <button className="btn clear-all-btn" onClick={() => clearAllChats()} title="Delete all chats">
                  Clear All
                </button>
              )}

              {/* Mobile-only close button (also works as overlay close) */}
              <button className="btn sidebar-close" title="Close chats" aria-label="Close chats" onClick={() => setSidebarOpen(false)}>
                <FiX />
              </button>
            </div>
          </div>

          <div className="conversations-list">
            {conversationsLoading && <div className="conv-loading">Loading...</div>}
            {(!conversationsLoading && conversations.length === 0) && (
              <div className="empty-convos">No conversations yet — start a new chat</div>
            )}
            {conversations.map((c) => (
              <div key={c.conversation_id} className={`conversation-item ${activeConversationId === c.conversation_id ? 'active' : ''}`} onClick={() => { loadConversation(c.conversation_id); setSidebarOpen(false); }}>
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
            <div key={msg.id}>
              <div
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
              {/* Show personalized insights after bot message with context */}
              {msg.type === 'bot' && msg.context && (
                <PersonalizedInsights
                  personalizedTips={msg.context.personalizedTips}
                  geminiTips={msg.context.geminiTips}
                  wellnessReminder={msg.context.wellnessReminder}
                  encouragement={msg.context.encouragement}
                  followUpQuestions={msg.context.followUpQuestions}
                  quickActions={msg.context.quickActions}
                />
              )}
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


      {/* ========== VOICE ASSISTANT OVERLAY (Google Assistant-like) ========== */}
      {voiceAssistantActive && (
        <div className="voice-assistant-overlay">
          <div className="voice-assistant-backdrop" onClick={closeVoiceAssistant} />
          <div className="voice-assistant-panel">
            {/* Close button */}
            <button className="va-close-btn" onClick={closeVoiceAssistant} aria-label="Close voice assistant">
              <FiX />
            </button>

            {/* Settings row */}
            <div className="va-settings">
              <label className="va-toggle-label">
                <input type="checkbox" checked={continuousMode} onChange={(e) => setContinuousMode(e.target.checked)} />
                <span>Continuous</span>
              </label>
              <label className="va-toggle-label">
                <input type="checkbox" checked={autoSpeak} onChange={(e) => setAutoSpeak(e.target.checked)} />
                <span>Auto-speak</span>
              </label>
              <label className="va-toggle-label">
                <input type="checkbox" checked={wakeWordEnabled} onChange={(e) => setWakeWordEnabled(e.target.checked)} />
                <span>Wake word</span>
              </label>
            </div>

            {/* Response display */}
            {assistantResponse && (
              <div className="va-response-area">
                <p className="va-response-text">{assistantResponse}</p>
              </div>
            )}

            {/* Transcript display */}
            {assistantTranscript && (
              <div className="va-transcript-area">
                <p className="va-transcript-text">
                  {assistantTranscript}
                </p>
              </div>
            )}

            {/* State label */}
            <div className="va-status-label">
              {assistantState === 'idle' && 'Tap to speak'}
              {assistantState === 'listening' && 'Listening...'}
              {assistantState === 'processing' && 'Thinking...'}
              {assistantState === 'speaking' && 'Speaking...'}
            </div>

            {/* Animated Orb */}
            <div
              className={`va-orb-container va-state-${assistantState}`}
              onClick={handleAssistantTap}
              role="button"
              tabIndex={0}
              aria-label={
                assistantState === 'idle' ? 'Tap to start speaking' :
                assistantState === 'listening' ? 'Tap to stop listening' :
                assistantState === 'speaking' ? 'Tap to stop speaking' :
                'Processing your request'
              }
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') handleAssistantTap(); }}
            >
              {/* Ripple rings */}
              <div className="va-ripple va-ripple-1" style={{ transform: `scale(${1 + audioLevel * 0.5})` }} />
              <div className="va-ripple va-ripple-2" style={{ transform: `scale(${1 + audioLevel * 0.35})` }} />
              <div className="va-ripple va-ripple-3" style={{ transform: `scale(${1 + audioLevel * 0.2})` }} />

              {/* Central orb */}
              <div
                className="va-orb"
                style={{ transform: `scale(${1 + audioLevel * 0.15})` }}
              >
                {assistantState === 'listening' && <FiMic className="va-orb-icon" />}
                {assistantState === 'processing' && (
                  <div className="va-spinner">
                    <div /><div /><div />
                  </div>
                )}
                {assistantState === 'speaking' && <FiVolume2 className="va-orb-icon va-speaking-icon" />}
                {assistantState === 'idle' && <FiMic className="va-orb-icon" />}
              </div>
            </div>

            {/* Waveform bars (animated when listening) */}
            {assistantState === 'listening' && (
              <div className="va-waveform">
                {[...Array(5)].map((_, i) => (
                  <div
                    key={i}
                    className="va-waveform-bar"
                    style={{
                      height: `${12 + audioLevel * 30 + Math.sin(Date.now() / 200 + i * 1.2) * 8}px`,
                      animationDelay: `${i * 0.1}s`,
                    }}
                  />
                ))}
              </div>
            )}

            {/* Quick hint */}
            <p className="va-hint">
              {assistantState === 'idle' && '"What are the symptoms of diabetes?"'}
              {assistantState === 'listening' && 'I\'m listening...'}
              {assistantState === 'processing' && 'Getting your answer...'}
              {assistantState === 'speaking' && 'Tap the orb to stop'}
            </p>
            {wakeWordEnabled && (
              <p className="va-hint va-wake-hint">💡 Say <strong>&quot;Voice&quot;</strong> anytime to activate hands-free</p>
            )}
          </div>
        </div>
      )}
    </div>

    {/* ========== FLOATING VOICE ASSISTANT BUTTON (moved outside container) ========== */}
    {!voiceAssistantActive && (
      <div className="voice-assistant-fab-wrapper" aria-hidden={voiceAssistantActive}>
        {wakeWordEnabled && (
          <div className={`wake-word-indicator ${wakeWordListening ? 'active' : ''}`} title={wakeWordListening ? 'Say "Voice" to activate' : 'Wake word starting...'}>
            <div className="wake-word-dot" />
            <span className="wake-word-label">Say "Voice"</span>
          </div>
        )}
        <button className="voice-assistant-fab" onClick={openVoiceAssistant} title="Open Voice Assistant" aria-label="Open voice assistant">
          <FiMic />
          <span className="fab-pulse" />
        </button>
      </div>
    )}
  </>
  );
};

export default HealthcareChatbot;