import React, { useState } from 'react';
import './MoodTracker.css';

const MOODS = [
  { emoji: '😊', name: 'Happy', color: '#4CAF50' },
  { emoji: '😐', name: 'Neutral', color: '#FFC107' },
  { emoji: '😔', name: 'Sad', color: '#2196F3' },
  { emoji: '😰', name: 'Anxious', color: '#FF6F00' },
  { emoji: '😴', name: 'Tired', color: '#9C27B0' },
  { emoji: '😍', name: 'Great', color: '#E91E63' },
];

const MoodTracker = ({ currentMood, onMoodChange }) => {
  const [expanded, setExpanded] = useState(false);

  const handleMoodSelect = (mood) => {
    onMoodChange(mood);
    setExpanded(false);
    
    // Add to mood history with timestamp
    const today = new Date().toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
    const history = JSON.parse(localStorage.getItem('moodHistory') || '[]');
    
    if (history.length === 0 || history[history.length - 1].date !== today) {
      history.push({ date: today, emoji: mood.emoji });
      localStorage.setItem('moodHistory', JSON.stringify(history.slice(-7))); // Keep last 7 days
    }
  };

  const selectedMood = MOODS.find(m => m.emoji === currentMood);

  return (
    <div className="mood-section">
      <div className="mood-picker">
        <span className="mood-label">Mood:</span>
        {!expanded ? (
          <button 
            className="mood-display"
            onClick={() => setExpanded(true)}
            title="How are you feeling?"
          >
            {selectedMood ? selectedMood.emoji : '😐'} {selectedMood ? selectedMood.name : 'Select'}
          </button>
        ) : (
          <div className="mood-options">
            {MOODS.map(mood => (
              <button
                key={mood.emoji}
                className="mood-btn"
                onClick={() => handleMoodSelect(mood)}
                style={{ '--mood-color': mood.color }}
                title={mood.name}
              >
                <span className="mood-emoji">{mood.emoji}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default MoodTracker;
export { MOODS };
