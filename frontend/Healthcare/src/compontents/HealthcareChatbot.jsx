// ===================== HEALTHCARE CHATBOT - REACT FRONTEND (UPDATED) =====================
// Complete React component with PROPER voice input that sends to backend for AI processing
// Voice → Microphone → Record Audio → Send to Backend → AI Model → Response → Display

// ===================== 1. INSTALL DEPENDENCIES =====================
// npm install axios react-icons

// ===================== 2. HEALTHCARECHATBOT.JSX (COMPLETE WORKING VERSION) =====================
import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { FiMic, FiSend, FiRefreshCw, FiTrash2, FiVolume2, FiStopCircle, FiLogOut, FiUser, FiMenu, FiChevronLeft, FiX, FiMicOff } from 'react-icons/fi';
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

  // ========== WAKE WORD DETECTION ("Hey Voicebot") ==========
  const [wakeWordEnabled, setWakeWordEnabled] = useState(true); // background listening for wake word
  const [wakeWordListening, setWakeWordListening] = useState(false); // UI indicator

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

  // Keep refs in sync with state for use inside callbacks
  useEffect(() => { voiceAssistantActiveRef.current = voiceAssistantActive; }, [voiceAssistantActive]);
  useEffect(() => { continuousModeRef.current = continuousMode; }, [continuousMode]);
  useEffect(() => { wakeWordEnabledRef.current = wakeWordEnabled; }, [wakeWordEnabled]);

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
    console.log('📌 activeConversationId:', activeConversationId);
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
        console.log(`📊 Message count before user: ${prev.length}`);
        return [...prev, userMessage];
      });

      // Use the CURRENTLY SELECTED conversation (don't create new ones)
      let convId = activeConversationId;
      console.log('🔍 Checking convId:', convId);
      
      if (!convId) {
        // Only create new conversation if REALLY no conversation is selected
        console.log('⚠️ No active conversation ID found, creating new one');
        const createResp = await axios.post(`${API_BASE_URL}/api/chat/conversations`, {});
        
        if (createResp.data?.status === 'success' && createResp.data.conversation_id) {
          convId = createResp.data.conversation_id;
          setActiveConversationId(convId);
          localStorage.setItem('activeConversationId', convId);
          console.log('✅ New conversation created:', convId);
        } else {
          throw new Error('Failed to create conversation');
        }
      } else {
        console.log('✅ Using existing conversation:', convId);
      }

      // Send message to API
      console.log('📤 Sending to API with conversation_id:', convId);
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
        console.log('➕ Adding bot message:', botText);
        setMessages((prev) => {
          console.log(`📊 Message count before bot: ${prev.length}`);
          return [...prev, botMessage];
        });

        console.log('📊 Current active conversation after response:', activeConversationId);
        
        // Refresh conversation list to update message count
        await fetchConversations();

        // Speak the response (only once, no duplicate)
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
    if (voiceAssistantActiveRef.current) {
      console.log('Wake word: Assistant is active, skipping');
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
          console.log('🚀 Wake word DETECTED! Opening assistant...');
          // Show what was detected
          toast.success(`🎯 Detected: "${mainTranscript.trim()}"`, { autoClose: 1500 });
          toast.info('🎙️ "Voice" detected! Activating...', { autoClose: 2000 });
          setTimeout(() => openVoiceAssistant(), 300);
          return;
        }
      }
    };

    recognition.onend = () => {
      if (listeningTimeout) clearTimeout(listeningTimeout);
      wakeWordRunningRef.current = false;
      setWakeWordListening(false);
      if (wakeWordTriggered) return; // don't restart if we triggered the assistant
      console.log('⏹️ Wake word: Listening ended, will restart soon');
      // Auto-restart immediately for continuous availability
      if (wakeWordEnabledRef.current && !voiceAssistantActiveRef.current) {
        console.log('🔄 Wake word: Restarting listener...');
        setTimeout(() => {
          if (wakeWordEnabledRef.current && !voiceAssistantActiveRef.current && !wakeWordRunningRef.current) {
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

      // For no-speech, network, etc. — retry
      if (wakeWordEnabledRef.current && !voiceAssistantActiveRef.current) {
        const delay = event.error === 'no-speech' ? 500 : 1500;
        setTimeout(() => {
          if (wakeWordEnabledRef.current && !voiceAssistantActiveRef.current && !wakeWordRunningRef.current) {
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
        if (wakeWordEnabledRef.current && !voiceAssistantActiveRef.current && !wakeWordRunningRef.current) {
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
    if (wakeWordEnabled && !voiceAssistantActive) {
      timer = setTimeout(() => {
        if (!wakeWordRunningRef.current) {
          startWakeWordListener();
        }
      }, 600);
    } else {
      stopWakeWordListener();
    }
    return () => {
      clearTimeout(timer);
      stopWakeWordListener();
    };
  }, [wakeWordEnabled, voiceAssistantActive, startWakeWordListener, stopWakeWordListener]);

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

        // Auto-speak the response if enabled
        if (autoSpeak) {
          speakResponse(response.data.response);
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

  const handleResetInput = () => {
    setInputText('');
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
        <div className="chatbot-header-actions">
          <button className="menu-btn" title="Menu" onClick={() => setSidebarOpen(!sidebarOpen)}>
            <FiMenu />
          </button>
        </div>
        <div className="chatbot-title">
          <span className="chatbot-logo">🩺</span> Healthcare Chatbot
          <span className="chatbot-subtitle">Voice & Text Support with AI Model</span>
        </div>
        <div className="chatbot-header-actions">
          <span className="user-avatar">{currentUser?.username || 'User'}</span>
          <button className="logout-btn" onClick={onLogout}>Logout</button>
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
              {conversations.length > 0 && (
                <button className="btn clear-all-btn" onClick={() => clearAllChats()} title="Delete all chats">
                  🗑️ Clear All
                </button>
              )}
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

      {/* ========== FLOATING VOICE ASSISTANT BUTTON ========== */}
      {!voiceAssistantActive && (
        <div className="voice-assistant-fab-wrapper">
          {/* Wake word status indicator */}
          {wakeWordEnabled && (
            <div className={`wake-word-indicator ${wakeWordListening ? 'active' : ''}`}
                 title={wakeWordListening ? 'Say "Voice" to activate' : 'Wake word starting...'}
            >
              <div className="wake-word-dot" />
              <span className="wake-word-label">Say &quot;Voice&quot;</span>
            </div>
          )}
          <button
            className="voice-assistant-fab"
            onClick={openVoiceAssistant}
            title="Open Voice Assistant"
            aria-label="Open voice assistant"
          >
            <FiMic />
            <span className="fab-pulse" />
          </button>
        </div>
      )}

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
  );
};

export default HealthcareChatbot;