// =====================================================================
// Feature Utilities and Constants
// =====================================================================

export const EMERGENCY_KEYWORDS = [
  'chest pain', 'heart attack', "can't breathe", 'cannot breathe', 'choking',
  'stroke', 'seizure', 'unable to move', 'severe bleeding', 'unconscious',
  'poisoning', 'allergic reaction', 'severe burn', 'broken bone', 'electric shock',
  'suicide', 'self harm', 'help', 'emergency', '911', '108', 'ambulance',
  'call emergency', 'in danger', 'attack', 'severe injury',
];

export const MOODS = [
  { emoji: '😊', name: 'Happy', color: '#4CAF50' },
  { emoji: '😐', name: 'Neutral', color: '#FFC107' },
  { emoji: '😔', name: 'Sad', color: '#2196F3' },
  { emoji: '😰', name: 'Anxious', color: '#FF6F00' },
  { emoji: '😴', name: 'Tired', color: '#9C27B0' },
  { emoji: '😍', name: 'Great', color: '#E91E63' },
];

export const HEALTH_TIPS = [
  { text: 'Stay hydrated! Drink at least 8 glasses of water daily.', emoji: '💧', category: 'Hydration' },
  { text: 'Get 7-9 hours of quality sleep every night.', emoji: '😴', category: 'Sleep' },
  { text: 'Exercise for 30 minutes daily to boost immunity.', emoji: '🏃', category: 'Exercise' },
  { text: 'Eat colorful vegetables for maximum nutrients.', emoji: '🥗', category: 'Nutrition' },
  { text: 'Practice deep breathing to reduce stress.', emoji: '🧘', category: 'Wellness' },
  { text: 'Limit sugar intake to prevent diabetes.', emoji: '🍬', category: 'Nutrition' },
  { text: 'Wash hands frequently to prevent infections.', emoji: '🧼', category: 'Hygiene' },
  { text: 'Take breaks during work to avoid eye strain.', emoji: '👁️', category: 'Wellness' },
  { text: 'Reduce salt consumption for heart health.', emoji: '❤️', category: 'Heart Health' },
  { text: 'Do regular check-ups with your doctor.', emoji: '⚕️', category: 'Prevention' },
  { text: 'Manage stress with meditation and yoga.', emoji: '🕉️', category: 'Mental Health' },
  { text: 'Avoid smoking and secondhand smoke exposure.', emoji: '🚭', category: 'Lifestyle' },
];

export const BODY_PARTS = {
  head: { label: 'Head', symptoms: ['Headache', 'Dizziness', 'Vision blur', 'Hearing loss'] },
  chest: { label: 'Chest', symptoms: ['Chest pain', 'Difficulty breathing', 'Heart palpitations'] },
  abdomen: { label: 'Abdomen', symptoms: ['Abdominal pain', 'Nausea', 'Indigestion', 'Diarrhea'] },
  leftArm: { label: 'Left Arm', symptoms: ['Arm pain', 'Numbness', 'Weakness', 'Tingling'] },
  rightArm: { label: 'Right Arm', symptoms: ['Arm pain', 'Numbness', 'Weakness', 'Tingling'] },
  back: { label: 'Back', symptoms: ['Back pain', 'Stiffness', 'Muscle tension'] },
  leftLeg: { label: 'Left Leg', symptoms: ['Leg pain', 'Swelling', 'Weakness', 'Cramping'] },
  rightLeg: { label: 'Right Leg', symptoms: ['Leg pain', 'Swelling', 'Weakness', 'Cramping'] },
};

/**
 * Check if text contains emergency keywords
 */
export const checkEmergency = (text) => {
  const lowerText = text.toLowerCase();
  for (let keyword of EMERGENCY_KEYWORDS) {
    if (lowerText.includes(keyword)) {
      return keyword;
    }
  }
  return null;
};

/**
 * Export chat as text file
 */
export const exportChatTxt = (messages, filename = null) => {
  if (messages.length === 0) {
    alert('No messages to export!');
    return;
  }

  let content = `Healthcare Chatbot Conversation Export\n`;
  content += `Date: ${new Date().toLocaleString()}\n`;
  content += `Total Messages: ${messages.length}\n`;
  content += `${'='.repeat(60)}\n\n`;

  messages.forEach((msg) => {
    const timestamp = msg.timestamp?.toLocaleString() || new Date().toLocaleString();
    const sender = msg.type === 'user' ? '👤 You' : '🤖 Assistant';
    content += `[${timestamp}] ${sender}:\n${msg.text}\n\n`;
  });

  const element = document.createElement('a');
  element.setAttribute('href', `data:text/plain;charset=utf-8,${encodeURIComponent(content)}`);
  element.setAttribute('download', filename || `healthcare_chat_${new Date().getTime()}.txt`);
  element.style.display = 'none';
  document.body.appendChild(element);
  element.click();
  document.body.removeChild(element);
};

/**
 * Get dashboard statistics from messages
 */
export const getDashboardStats = (messages, moodHistory) => {
  const totalMessages = messages.length;
  const userMessages = messages.filter(m => m.type === 'user').length;
  const botMessages = messages.filter(m => m.type === 'bot').length;
  
  const voiceQueries = messages.filter(m => m.isVoice).length;
  const questionsAsked = userMessages;

  const keywords = ['diabetes', 'heart', 'covid', 'allergy', 'headache', 'fever', 'pain', 'sleep', 'stress', 'weight'];
  const topicCounts = {};
  keywords.forEach(topic => {
    topicCounts[topic] = messages.filter(m => 
      m.text?.toLowerCase().includes(topic)
    ).length;
  });

  const topTopics = Object.entries(topicCounts)
    .filter(([_, count]) => count > 0)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  return {
    totalMessages,
    conversations: 1,
    voiceQueries,
    questionsAsked,
    topTopics: topTopics.length > 0 ? topTopics : [['general', userMessages]],
    moodHistory: moodHistory || [],
  };
};
