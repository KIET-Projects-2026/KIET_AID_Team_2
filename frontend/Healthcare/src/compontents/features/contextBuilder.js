// =====================================================================
// Context Builder - Assembles user data for personalized AI responses
// =====================================================================

/**
 * Build comprehensive user context from mood, symptoms, and chat history
 */
export const buildUserContext = (currentMood, selectedSymptoms, messages, moodHistory) => {
  // Get mood details
  const getMoodDetails = () => {
    const moodMap = {
      '😊': { name: 'happy', level: 'positive', stress: 'low' },
      '😐': { name: 'neutral', level: 'normal', stress: 'medium' },
      '😔': { name: 'sad', level: 'negative', stress: 'high' },
      '😰': { name: 'anxious', level: 'negative', stress: 'very_high' },
      '😴': { name: 'tired', level: 'negative', stress: 'medium' },
      '😍': { name: 'great', level: 'positive', stress: 'low' },
    };
    return moodMap[currentMood] || { name: 'unknown', level: 'neutral', stress: 'medium' };
  };

  // Analyze recent symptoms from chat history
  const analyzeSymptoms = () => {
    const symptomKeywords = {
      headache: ['headache', 'head pain', 'migraine', 'throbbing'],
      fever: ['fever', 'temperature', 'warm', 'hot'],
      fatigue: ['tired', 'fatigue', 'exhausted', 'sleepy'],
      anxiety: ['anxious', 'worried', 'nervous', 'stress'],
      pain: ['pain', 'ache', 'hurt', 'soreness'],
      cough: ['cough', 'coughing', 'throat'],
      nausea: ['nausea', 'sick', 'vomiting', 'stomach'],
      insomnia: ['sleep', 'insomnia', 'sleepless', 'awake'],
    };

    const detectedSymptoms = new Set();
    const recentMessages = messages.slice(-10); // Last 10 messages

    recentMessages.forEach(msg => {
      if (msg.type === 'user') {
        const text = msg.text.toLowerCase();
        Object.entries(symptomKeywords).forEach(([symptom, keywords]) => {
          if (keywords.some(kw => text.includes(kw))) {
            detectedSymptoms.add(symptom);
          }
        });
      }
    });

    return Array.from(detectedSymptoms);
  };

  // Analyze conversation topics
  const analyzeTopics = () => {
    const topicKeywords = {
      sleep: ['sleep', 'insomnia', 'rest', 'tired', 'bed'],
      nutrition: ['food', 'eat', 'diet', 'nutrition', 'hungry'],
      exercise: ['exercise', 'workout', 'fitness', 'gym', 'run'],
      stress: ['stress', 'anxiety', 'worry', 'pressure', 'tense'],
      immunity: ['immune', 'cold', 'flu', 'infection', 'virus'],
      heart: ['heart', 'cardiac', 'blood pressure', 'chest'],
      mental: ['depression', 'mental', 'anxiety', 'mood', 'emotional'],
    };

    const topics = {};
    messages.forEach(msg => {
      if (msg.type === 'user') {
        const text = msg.text.toLowerCase();
        Object.entries(topicKeywords).forEach(([topic, keywords]) => {
          if (keywords.some(kw => text.includes(kw))) {
            topics[topic] = (topics[topic] || 0) + 1;
          }
        });
      }
    });

    return topics;
  };

  // Calculate engagement metrics
  const getEngagementMetrics = () => {
    const totalMessages = messages.length;
    const userMessages = messages.filter(m => m.type === 'user').length;
    const voiceQueries = messages.filter(m => m.isVoice).length;
    const avgMessageLength = messages.reduce((sum, m) => sum + (m.text?.length || 0), 0) / userMessages || 0;

    return {
      totalMessages,
      userMessages,
      voiceQueries,
      avgMessageLength: Math.round(avgMessageLength),
      sessionDuration: messages.length > 0 ? 'ongoing' : 'new',
    };
  };

  // Get mood trend
  const getMoodTrend = () => {
    if (!moodHistory || moodHistory.length === 0) return 'stable';
    if (moodHistory.length < 2) return 'new_data';
    
    const recent = moodHistory.slice(-3);
    const positiveEmojis = ['😊', '😍'];
    const positiveCount = recent.filter(m => positiveEmojis.includes(m.emoji)).length;
    
    if (positiveCount >= 2) return 'improving';
    if (positiveCount === 0) return 'declining';
    return 'stable';
  };

  return {
    mood: getMoodDetails(),
    moodEmoji: currentMood,
    moodTrend: getMoodTrend(),
    symptoms: analyzeSymptoms(),
    selectedSymptoms: selectedSymptoms || [],
    topics: analyzeTopics(),
    metrics: getEngagementMetrics(),
    timestamp: new Date().toISOString(),
    messageCount: messages.length,
  };
};

/**
 * Generate system prompt with user context
 */
export const generateContextualSystemPrompt = (userContext) => {
  const moodInfo = userContext.mood.name;
  const stressLevel = userContext.mood.stress;
  const symptoms = userContext.symptoms.length > 0 
    ? `The user is experiencing: ${userContext.symptoms.join(', ')}`
    : 'No specific symptoms mentioned.';
  
  const basePrompt = `You are a compassionate healthcare assistant AI. Provide helpful medical guidance while being empathetic and supportive.

CURRENT USER CONTEXT:
- Mood: ${userContext.moodEmoji} (${moodInfo})
- Stress Level: ${stressLevel}
- Primary Concerns: ${Object.keys(userContext.topics).length > 0 ? Object.keys(userContext.topics).join(', ') : 'General health'}
- ${symptoms}
- Mood Trend: ${userContext.moodTrend}

GUIDELINES FOR THIS RESPONSE:
1. Acknowledge their current emotional state and validate their feelings
2. Provide practical, evidence-based health advice
3. If they're stressed/anxious: Recommend relaxation techniques and reassurance
4. If they're tired: Suggest rest, hydration, and sleep hygiene
5. If they're sad: Include mental health resources and positive affirmations
6. If they're great: Encourage maintaining healthy habits
7. Include 1-2 specific, actionable tips tailored to their current situation
8. Keep responses concise but caring (2-3 sentences for main advice + 1-2 tips)`;

  return basePrompt;
};

/**
 * Create personalized health tips based on mood and symptoms
 */
export const getPersonalizedTips = (userContext) => {
  // Tips feature disabled — return empty list so frontend receives no personalized tips.
  return [];
};

/**
 * Format user context for API request
 */
export const formatContextForAPI = (userContext) => {
  return {
    user_mood: userContext.moodEmoji,
    mood_state: userContext.mood.name,
    stress_level: userContext.mood.stress,
    symptoms: userContext.symptoms,
    primary_topics: Object.keys(userContext.topics),
    mood_trend: userContext.moodTrend,
    engagement: userContext.metrics,
  };
};
