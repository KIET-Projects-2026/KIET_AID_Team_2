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
      <div className="health-tip-card" role="group" aria-label="Health tip">
        <div className="tip-icon" aria-hidden>{tip.emoji}</div>
        <div className="tip-text">{tip.text}</div>
        <div className="tip-category" aria-hidden>{tip.category}</div>
      </div>
      <div className="tips-controls" role="toolbar" aria-label="Tip controls">
        <button className="tip-btn" onClick={handlePrevTip} aria-label="Previous tip">←</button>
        <button className="tip-btn" onClick={handleNextTip} aria-label="Next tip">→</button>
      </div>
    </div>
  );
};

export default HealthTips;
export { HEALTH_TIPS };
