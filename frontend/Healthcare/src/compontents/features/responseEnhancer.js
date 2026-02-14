// =====================================================================
// Response Enhancer - Adds personalized content to AI responses
// =====================================================================

/**
 * Enhance AI response with mood-based suggestions and tips
 */
export const enhanceResponseWithContext = (aiResponse, userContext, personalizedTips) => {
  let enhancedResponse = aiResponse;

  // Add empathetic prefix if user is in negative mood
  if (['sad', 'anxious', 'tired'].includes(userContext.mood.name)) {
    const empathyPrefixes = {
      sad: 'I understand you\'re feeling down right now. ',
      anxious: 'I sense you\'re feeling worried. ',
      tired: 'I can tell you\'re exhausted. ',
    };
    if (!enhancedResponse.toLowerCase().includes('understand') && 
        !enhancedResponse.toLowerCase().includes('hear')) {
      enhancedResponse = empathyPrefixes[userContext.mood.name] + enhancedResponse;
    }
  }

  return enhancedResponse;
};

/**
 * Format tips section to append to response
 */
export const formatTipsSection = (personalizedTips) => {
  if (!personalizedTips || personalizedTips.length === 0) {
    return '';
  }

  let tipsSection = '\n\n💡 **Personalized Tips for You:**\n';
  personalizedTips.forEach((tip, index) => {
    tipsSection += `${index + 1}. ${tip}\n`;
  });

  return tipsSection;
};

/**
 * Create a wellness reminder based on user context
 */
export const generateWellnessReminder = (userContext) => {
  const reminders = {
    happy: {
      title: '🌟 Keep Your Momentum!',
      message: 'You\'re in great spirits! This is an excellent time to reinforce your positive habits.',
    },
    neutral: {
      title: '⚖️ Staying Balanced',
      message: 'You\'re maintaining steady health. Keep up with regular check-ups and healthy routines.',
    },
    sad: {
      title: '💙 You\'re Not Alone',
      message: 'It\'s okay to feel down sometimes. Remember to reach out to friends, family, or professionals.',
    },
    anxious: {
      title: '🌿 Finding Calm',
      message: 'Take a deep breath. Short mindfulness exercises can help manage anxious feelings.',
    },
    tired: {
      title: '⚡ Rest is Important',
      message: 'Your body needs recovery. Prioritize sleep and be gentle with yourself today.',
    },
    great: {
      title: '🚀 On Top of the World!',
      message: 'You\'re feeling amazing! This is perfect timing for setting new health goals.',
    },
  };

  const moodName = userContext.mood.name;
  return reminders[moodName] || { title: '💚 Your Health Matters', message: 'Take care of yourself!' };
};

/**
 * Analyze message sentiment to detect urgency
 */
export const detectUrgency = (userMessage, userContext) => {
  const urgentKeywords = [
    'emergency', 'urgent', 'severe', 'critical', 'can\'t breathe',
    'chest pain', 'dying', 'help', 'hospital', 'ambulance'
  ];

  const isUrgent = urgentKeywords.some(keyword => 
    userMessage.toLowerCase().includes(keyword)
  );

  const isHighStress = userContext.mood.stress === 'very_high' || 
                       userContext.mood.stress === 'high';

  return {
    urgent: isUrgent,
    highStress: isHighStress,
    needsEscalation: isUrgent || (isHighStress && userMessage.length > 100),
  };
};

/**
 * Generate follow-up questions based on context
 */
export const generateFollowUpQuestions = (userContext) => {
  const questions = [];

  // Mood-based follow-ups
  if (userContext.mood.name === 'anxious') {
    questions.push('What specific situations trigger your anxiety?');
  }
  if (userContext.mood.name === 'sad') {
    questions.push('How long have you been feeling this way?');
  }
  if (userContext.mood.name === 'tired') {
    questions.push('How many hours of sleep are you getting?');
  }

  // Symptom-based follow-ups
  if (userContext.symptoms.includes('fever')) {
    questions.push('When did the fever start?');
  }
  if (userContext.symptoms.includes('headache')) {
    questions.push('Where exactly is the pain located?');
  }
  if (userContext.symptoms.includes('pain')) {
    questions.push('On a scale of 1-10, how severe is the pain?');
  }

  // Topic-based follow-ups
  const primaryTopic = Object.entries(userContext.topics)
    .sort((a, b) => b[1] - a[1])[0];
  
  if (primaryTopic) {
    const topicQuestions = {
      sleep: 'What time do you usually go to bed?',
      nutrition: 'What foods make you feel better?',
      exercise: 'What type of exercise do you enjoy?',
      stress: 'What causes you the most stress?',
      immunity: 'Have you been exposed to anyone sick?',
      heart: 'Has anyone in your family had heart problems?',
      mental: 'How long have you been experiencing these feelings?',
    };
    if (topicQuestions[primaryTopic[0]]) {
      questions.push(topicQuestions[primaryTopic[0]]);
    }
  }

  return questions.slice(0, 2); // Return top 2 follow-up questions
};

/**
 * Create quick action suggestions
 */
export const getQuickActions = (userContext) => {
  const actions = [];

  // Mood-based actions
  const moodActions = {
    happy: ['📊 View Health Dashboard', '💪 Log New Workout'],
    anxious: ['🧘 Try Breathing Exercise', '📞 Find Support Resources'],
    tired: ['😴 Check Sleep Tips', '💧 Hydration Reminder'],
    sad: ['💬 Chat with Counselor', '📚 Mental Health Resources'],
    neutral: ['📈 Track Health Metrics', '🎯 Set Wellness Goals'],
    great: ['🏆 Set New Goals', '👥 Share Success'],
  };

  if (moodActions[userContext.mood.name]) {
    actions.push(...moodActions[userContext.mood.name]);
  }

  // Symptom-based actions
  if (userContext.symptoms.length > 0) {
    actions.push('🩺 Consult Doctor');
  }

  // Context-based actions
  if (userContext.metrics.voiceQueries > 5) {
    actions.push('🎙️ Voice Guide');
  }

  return actions.slice(0, 3);
};

/**
 * Generate encouragement message based on progress
 */
export const getEncouragementMessage = (userContext) => {
  const messages = {
    improving: {
      emoji: '📈',
      text: 'Your mood has been improving! Keep up the positive momentum.',
    },
    declining: {
      emoji: '💪',
      text: 'I notice you\'ve been feeling lower. Let\'s focus on what can help you feel better.',
    },
    stable: {
      emoji: '⚖️',
      text: 'You\'re maintaining steady wellness. Consistency is key!',
    },
    new_data: {
      emoji: '👋',
      text: 'I\'m starting to learn about your health patterns. Keep sharing!',
    },
  };

  return messages[userContext.moodTrend] || messages.stable;
};
