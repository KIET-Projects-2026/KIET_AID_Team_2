import React from 'react';
import './EmergencySOS.css';

const EMERGENCY_KEYWORDS = [
  'chest pain', 'heart attack', "can't breathe", 'cannot breathe', 'choking',
  'stroke', 'seizure', 'unable to move', 'severe bleeding', 'unconscious',
  'poisoning', 'allergic reaction', 'severe burn', 'broken bone', 'electric shock',
  'suicide', 'self harm', 'help', 'emergency', '911', '108', 'ambulance',
  'call emergency', 'in danger', 'attack', 'severe injury',
];

const EmergencySOS = ({ messages, onDismiss }) => {
  const checkEmergency = (text) => {
    const lowerText = text.toLowerCase();
    for (let keyword of EMERGENCY_KEYWORDS) {
      if (lowerText.includes(keyword)) {
        return keyword;
      }
    }
    return null;
  };

  // Check last user message for emergency keywords
  const emergencyType = (() => {
    const lastUserMsg = [...messages].reverse().find(m => m.type === 'user');
    return lastUserMsg ? checkEmergency(lastUserMsg.text) : null;
  })();

  if (!emergencyType) return null;

  return (
    <div className="emergency-banner">
      <div className="emergency-content">
        <div className="emergency-icon">🚨</div>
        <div className="emergency-text">
          <strong>EMERGENCY DETECTED</strong>
          <p>Please call emergency services immediately</p>
        </div>
        <div className="emergency-numbers">
          <a href="tel:911" className="emergency-call">📞 911 (US)</a>
          <a href="tel:108" className="emergency-call">📞 108 (India)</a>
          <a href="tel:112" className="emergency-call">📞 112 (EU)</a>
        </div>
        <button className="emergency-dismiss" onClick={onDismiss}>✕</button>
      </div>
    </div>
  );
};

export default EmergencySOS;
