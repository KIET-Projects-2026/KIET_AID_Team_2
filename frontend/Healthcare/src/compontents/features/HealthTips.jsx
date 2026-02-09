import React, { useState, useEffect } from 'react';
import './HealthTips.css';

const HEALTH_TIPS = [
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

const HealthTips = ({ currentTipIndex, onTipChange }) => {
  const tip = HEALTH_TIPS[currentTipIndex];

  useEffect(() => {
    const interval = setInterval(() => {
      onTipChange((prev) => (prev + 1) % HEALTH_TIPS.length);
    }, 8000); // Change tip every 8 seconds

    return () => clearInterval(interval);
  }, [onTipChange]);

  const handlePrevTip = () => {
    onTipChange((prev) => (prev - 1 + HEALTH_TIPS.length) % HEALTH_TIPS.length);
  };

  const handleNextTip = () => {
    onTipChange((prev) => (prev + 1) % HEALTH_TIPS.length);
  };

  return (
    <div className="tips-section">
      <div className="health-tip-card">
        <div className="tip-icon">{tip.emoji}</div>
        <div className="tip-text">{tip.text}</div>
        <div className="tip-category">{tip.category}</div>
      </div>
      <div style={{ display: 'flex', gap: '4px' }}>
        <button 
          onClick={handlePrevTip} 
          style={{ padding: '4px 8px', fontSize: '12px', border: 'none', background: '#f0f0f0', borderRadius: '4px', cursor: 'pointer' }}
        >
          ←
        </button>
        <button 
          onClick={handleNextTip} 
          style={{ padding: '4px 8px', fontSize: '12px', border: 'none', background: '#f0f0f0', borderRadius: '4px', cursor: 'pointer' }}
        >
          →
        </button>
      </div>
    </div>
  );
};

export default HealthTips;
export { HEALTH_TIPS };
