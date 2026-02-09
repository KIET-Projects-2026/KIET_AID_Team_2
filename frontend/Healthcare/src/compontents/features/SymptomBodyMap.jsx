import React, { useState } from 'react';
import './SymptomBodyMap.css';

const BODY_PARTS = {
  head: { label: 'Head', symptoms: ['Headache', 'Dizziness', 'Vision blur', 'Hearing loss'] },
  chest: { label: 'Chest', symptoms: ['Chest pain', 'Difficulty breathing', 'Heart palpitations'] },
  abdomen: { label: 'Abdomen', symptoms: ['Abdominal pain', 'Nausea', 'Indigestion', 'Diarrhea'] },
  leftArm: { label: 'Left Arm', symptoms: ['Arm pain', 'Numbness', 'Weakness', 'Tingling'] },
  rightArm: { label: 'Right Arm', symptoms: ['Arm pain', 'Numbness', 'Weakness', 'Tingling'] },
  back: { label: 'Back', symptoms: ['Back pain', 'Stiffness', 'Muscle tension'] },
  leftLeg: { label: 'Left Leg', symptoms: ['Leg pain', 'Swelling', 'Weakness', 'Cramping'] },
  rightLeg: { label: 'Right Leg', symptoms: ['Leg pain', 'Swelling', 'Weakness', 'Cramping'] },
};

const SymptomBodyMap = ({ onSymptomSelect, onClose }) => {
  const [selectedPart, setSelectedPart] = useState(null);

  const handleSymptomSelect = (symptom) => {
    if (onSymptomSelect) {
      onSymptomSelect(symptom);
    }
    onClose();
  };

  return (
    <div className="body-map-overlay">
      <div className="body-map-backdrop" onClick={onClose} />
      <div className="body-map-panel">
        <div className="body-map-header">
          <h3>😷 Symptom Checker</h3>
          <p>Select the area where you feel discomfort</p>
          <button className="body-map-close" onClick={onClose}>✕</button>
        </div>

        <div className="body-map-content">
          {/* Simple SVG Body Figure */}
          <div className="body-figure">
            <svg className="body-svg" viewBox="0 0 100 200" xmlns="http://www.w3.org/2000/svg">
              {/* Head */}
              <circle cx="50" cy="25" r="15" className={`body-part ${selectedPart === 'head' ? 'selected' : ''}`} onClick={() => setSelectedPart('head')} />
              
              {/* Torso/Chest */}
              <rect x="40" y="45" width="20" height="30" rx="5" className={`body-part ${selectedPart === 'chest' ? 'selected' : ''}`} onClick={() => setSelectedPart('chest')} />
              
              {/* Abdomen */}
              <rect x="40" y="75" width="20" height="25" rx="5" className={`body-part ${selectedPart === 'abdomen' ? 'selected' : ''}`} onClick={() => setSelectedPart('abdomen')} />
              
              {/* Left Arm */}
              <rect x="15" y="50" width="22" height="15" rx="5" className={`body-part ${selectedPart === 'leftArm' ? 'selected' : ''}`} onClick={() => setSelectedPart('leftArm')} />
              
              {/* Right Arm */}
              <rect x="63" y="50" width="22" height="15" rx="5" className={`body-part ${selectedPart === 'rightArm' ? 'selected' : ''}`} onClick={() => setSelectedPart('rightArm')} />
              
              {/* Back (shown as rectangle behind) */}
              <rect x="35" y="45" width="30" height="55" rx="5" opacity="0.3" className={`body-part ${selectedPart === 'back' ? 'selected' : ''}`} onClick={() => setSelectedPart('back')} />
              
              {/* Left Leg */}
              <rect x="35" y="100" width="12" height="35" rx="5" className={`body-part ${selectedPart === 'leftLeg' ? 'selected' : ''}`} onClick={() => setSelectedPart('leftLeg')} />
              
              {/* Right Leg */}
              <rect x="53" y="100" width="12" height="35" rx="5" className={`body-part ${selectedPart === 'rightLeg' ? 'selected' : ''}`} onClick={() => setSelectedPart('rightLeg')} />
            </svg>

            {/* Body Part Labels */}
            {Object.entries(BODY_PARTS).map(([key, part]) => (
              <button
                key={key}
                className={`body-label ${selectedPart === key ? 'selected' : ''}`}
                onClick={() => setSelectedPart(key)}
                style={{
                  left: key === 'head' ? '50%' : key === 'leftArm' ? '20%' : key === 'rightArm' ? '80%' : key === 'leftLeg' ? '40%' : key === 'rightLeg' ? '60%' : '50%',
                  top: key === 'head' ? '15%' : key === 'chest' ? '35%' : key === 'abdomen' ? '60%' : key === 'back' ? '55%' : key === 'leftLeg' ? '85%' : key === 'rightLeg' ? '85%' : '50%',
                }}
              >
                {part.label}
              </button>
            ))}
          </div>

          {/* Symptoms Selector */}
          {selectedPart && (
            <div className="symptom-selector">
              <h4>Common symptoms in {BODY_PARTS[selectedPart].label}:</h4>
              <div className="symptom-grid">
                {BODY_PARTS[selectedPart].symptoms.map((symptom, idx) => (
                  <button
                    key={idx}
                    className="symptom-btn"
                    onClick={() => handleSymptomSelect(symptom)}
                  >
                    {symptom}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SymptomBodyMap;
export { BODY_PARTS };
