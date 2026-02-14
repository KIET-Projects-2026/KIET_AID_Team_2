// =====================================================================
// Mood-Symptom Based Tips Database
// =====================================================================

/**
 * Comprehensive tip database combining mood + symptoms + topics
 */
export const PERSONALIZED_TIPS_DB = {
  // Mood + Symptom combinations
  combinations: {
    'anxious_headache': [
      '🧘 Practice progressive muscle relaxation to ease tension headaches',
      '💧 Stay hydrated - dehydration worsens both anxiety and headaches',
      '🎵 Listen to calming music while resting in a dark room',
    ],
    'anxious_insomnia': [
      '🌙 Try the 4-7-8 breathing technique before bed',
      '📱 Put away all screens 1 hour before sleep',
      '🧘 Meditate for 10 minutes before bed',
    ],
    'anxious_fatigue': [
      '🚶 Light walking can help reduce anxiety and boost energy',
      '☕ Moderate caffeine intake (before 2 PM)',
      '🫂 Talk to someone you trust about your feelings',
    ],
    'sad_insomnia': [
      '☀️ Get morning sunlight exposure for mood regulation',
      '🛏️ Maintain consistent sleep/wake times',
      '🧠 Consider journaling before bed to process emotions',
    ],
    'sad_fatigue': [
      '💪 Gentle exercise releases mood-boosting endorphins',
      '🥗 Eat omega-3 rich foods (fish, walnuts, flax)',
      '🤝 Reach out to friends or family for support',
    ],
    'tired_headache': [
      '💧 Drink 2-3 liters of water daily',
      '⏸️ Take 20-minute power naps if needed',
      '👁️ Rest your eyes if you work on screens',
    ],
    'happy_pain': [
      '💪 Channel positive energy into healing',
      '🎯 Moderate activity - don\'t overdo it while feeling good',
      '🙏 Practice gratitude alongside pain management',
    ],
  },

  // Mood-specific daily tips
  moodTips: {
    anxious: {
      morning: '🌅 Start with 5 minutes of deep breathing',
      midday: '🚶 Take a short walk in fresh air to reset',
      evening: '🌿 Wind down with herbal tea',
      nutrition: 'Avoid excessive caffeine; eat magnesium-rich foods',
      exercise: 'Yoga or tai chi is ideal for anxiety',
    },
    sad: {
      morning: '🌞 Open curtains and let sunlight in',
      midday: '🤝 Connect with someone who makes you smile',
      evening: '📚 Engage in activities you enjoy',
      nutrition: 'Include mood-boosting foods: dark chocolate, berries',
      exercise: 'Even 10 minutes of walking helps elevate mood',
    },
    tired: {
      morning: '⏰ Go to bed and wake at consistent times',
      midday: '💧 Drink water and have a light snack',
      evening: '🛏️ Create a relaxing bedtime routine',
      nutrition: 'Iron-rich foods: spinach, lentils, chicken',
      exercise: 'Light movement enhances sleep quality',
    },
    happy: {
      morning: '🌟 Start day with gratitude',
      midday: '📈 Set a new health goal',
      evening: '🎉 Celebrate your progress',
      nutrition: 'Maintain balanced diet to sustain energy',
      exercise: 'Channel energy into strength training',
    },
    neutral: {
      morning: '⚖️ Establish a morning routine',
      midday: '📊 Track health metrics',
      evening: '📝 Plan tomorrow\'s healthy choices',
      nutrition: 'Follow balanced meal guidelines',
      exercise: 'Aim for 150 minutes weekly',
    },
  },

  // Symptom-specific management tips
  symptomCare: {
    headache: {
      immediate: '🧊 Apply cold compress to neck/temples',
      preventive: 'Stay hydrated, maintain posture, manage stress',
      foods: 'Magnesium-rich (almonds, spinach), stay hydrated',
      alert: 'If severe or sudden, seek medical help',
    },
    fever: {
      immediate: '❄️ Cool compress, lukewarm bath',
      preventive: 'Vaccines, hygiene, avoid sick people',
      foods: 'Warm fluids, vitamin C, broths',
      alert: 'Fever > 103°F or lasting > 3 days needs doctor',
    },
    cough: {
      immediate: '🍯 Honey, ginger tea, throat lozenges',
      preventive: 'Hand washing, avoid smoke/pollution',
      foods: 'Warm liquids, vitamin C, avoid dairy if phlegmy',
      alert: 'Cough lasting > 3 weeks needs evaluation',
    },
    fatigue: {
      immediate: '😴 Rest 20-30 minutes, drink water',
      preventive: '7-9 hrs sleep, exercise, balanced diet',
      foods: 'Iron, B-vitamins, complex carbs, protein',
      alert: 'Persistent fatigue may indicate underlying condition',
    },
    nausea: {
      immediate: '🍋 Ginger tea, lemon water, crackers',
      preventive: 'Eat small frequent meals, stay hydrated',
      foods: 'Bland: toast, rice, banana, apple',
      alert: 'If accompanied by fever, seek medical advice',
    },
    pain: {
      immediate: 'Rest affected area, apply heat/cold',
      preventive: 'Maintain proper posture, stretch regularly',
      foods: 'Anti-inflammatory: turmeric, ginger, berries',
      alert: 'Severe or persistent pain needs doctor visit',
    },
  },

  // Topic-focused wellness tips
  topicFocus: {
    sleep: [
      '🛏️ Keep bedroom cool (65-68°F), dark, quiet',
      '📱 No screens 1 hour before bed (blue light suppresses melatonin)',
      '⏰ Consistent bedtime/wake time even weekends',
      '🚫 Avoid caffeine after 2 PM',
      '🧘 Try meditation or ASMR before sleep',
      '💤 Aim for 7-9 hours nightly',
    ],
    nutrition: [
      '🥗 Fill half your plate with vegetables',
      '🥛 Include protein at every meal (lean meat, fish, beans)',
      '🍎 Eat fruits instead of candy for sweet cravings',
      '💧 Drink water before meals to aid digestion',
      '🚫 Limit processed foods and added sugars',
      '⏰ Eat every 3-4 hours to maintain steady energy',
    ],
    exercise: [
      '🏃 150 minutes moderate cardio per week',
      '💪 Strength training 2-3 times weekly',
      '🧘 Flexibility/yoga 2-3 times weekly',
      '🚶 Take the stairs, park farther away',
      '⏰ Exercise in morning for better sleep',
      '🎯 Set small achievable goals',
    ],
    stress: [
      '🧘 Meditation: Start with 5 minutes daily',
      '🌿 Deep breathing: 4 counts in, 6 out',
      '💬 Talk to someone - don\'t keep it bottled',
      '🎵 Music, art, nature for stress relief',
      '⏰ Take regular breaks from work',
      '📱 Limit news/social media consumption',
    ],
    immunity: [
      '💉 Keep vaccinations up to date',
      '🧼 Wash hands for 20 seconds regularly',
      '🥗 Eat vitamin C (citrus, berries, peppers)',
      '😴 Get 7-9 hours quality sleep',
      '🏃 Regular exercise boosts immunity',
      '🌡️ Avoid close contact when sick',
    ],
    heart: [
      '💓 Monitor blood pressure regularly',
      '🧂 Reduce salt intake',
      '💪 Cardiovascular exercise 150 min/week',
      '🥗 Eat omega-3: fish, flax, walnuts',
      '🚭 Don\'t smoke; avoid secondhand smoke',
      '⚖️ Maintain healthy weight (BMI 18.5-24.9)',
    ],
    mental: [
      '💬 Talk to a therapist or counselor',
      '📱 Reach out to support networks',
      '🧘 Mindfulness reduces depression/anxiety',
      '🎯 Set small achievable daily goals',
      '💪 Celebrate small wins',
      '📞 Crisis hotline: Available 24/7',
    ],
  },

  // Recovery protocols for common conditions
  recoveryProtocols: {
    cold: {
      duration: '3-7 days',
      steps: [
        '💧 Drink warm fluids (tea, broth, water)',
        '😴 Get adequate rest',
        '🍯 Honey and ginger for throat',
        '👃 Saline drops for congestion',
        '🌡️ Monitor for fever',
      ],
    },
    flu: {
      duration: '7-14 days',
      steps: [
        '🏥 Consult doctor - antivirals may help',
        '😴 Complete bed rest',
        '💧 Hydration is critical',
        '🧊 Manage fever with cool compress',
        '🚫 Avoid work/school for first 5 days',
      ],
    },
    stress: {
      duration: '2-4 weeks of practice',
      steps: [
        '🧘 Daily meditation (10-15 min)',
        '🏃 Exercise 30 min daily',
        '💬 Therapy/counseling sessions',
        '💤 Prioritize sleep',
        '🤝 Build support network',
      ],
    },
    insomnia: {
      duration: '2-8 weeks to reset',
      steps: [
        '⏰ Stick to consistent sleep schedule',
        '🛏️ Use bed only for sleep',
        '📱 No screens 1 hour before bed',
        '🧘 Relaxation techniques',
        '📞 See doctor if persists > 2 weeks',
      ],
    },
  },
};

/**
 * Get tips based on mood and symptoms
 */
export const getTipsByMoodAndSymptoms = (mood, symptoms) => {
  const tips = [];

  // Check combinations
  symptoms.forEach(symptom => {
    const key = `${mood}_${symptom}`;
    if (PERSONALIZED_TIPS_DB.combinations[key]) {
      tips.push(...PERSONALIZED_TIPS_DB.combinations[key]);
    }
  });

  // If no combinations found, use mood-specific tips
  if (tips.length === 0 && PERSONALIZED_TIPS_DB.moodTips[mood]) {
    tips.push(PERSONALIZED_TIPS_DB.moodTips[mood].nutrition);
    tips.push(PERSONALIZED_TIPS_DB.moodTips[mood].exercise);
  }

  return tips.slice(0, 3);
};

/**
 * Get recovery protocol for a condition
 */
export const getRecoveryProtocol = (condition) => {
  return PERSONALIZED_TIPS_DB.recoveryProtocols[condition] || null;
};

/**
 * Get topic-specific tips
 */
export const getTopicTips = (topic) => {
  return PERSONALIZED_TIPS_DB.topicFocus[topic] || [];
};
