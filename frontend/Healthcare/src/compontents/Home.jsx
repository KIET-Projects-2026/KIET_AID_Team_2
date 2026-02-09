import React, { useEffect, useRef, useState } from 'react';
import { FiUserPlus, FiLogIn, FiMic, FiMessageSquare, FiShield, FiBarChart2 } from 'react-icons/fi';
import './Home.css';

const Home = ({ onSwitchToLogin, onSwitchToSignup }) => {
  const slides = [
    {
      id: 'how-to',
      content: (
        <section className="how-to card">
          <h2>How to use this bot</h2>
          <ol>
            <li>Create an account or sign in.</li>
            <li>Start a new conversation or continue an existing one from the Chats sidebar.</li>
            <li>Type or speak your question and get an AI-powered response.</li>
            <li>Tap a conversation to revisit all messages — the first message is always preserved.</li>
          </ol>
        </section>
      )
    },
    {
      id: 'features',
      content: (
        <section className="features card">
          <h2>Features</h2>
          <ul className="features-list">
            <li><FiMic className="feature-icon" /> Text and voice input</li>
            <li><FiMessageSquare className="feature-icon" /> Persistent chat history grouped by conversation</li>
            <li><FiShield className="feature-icon" /> Secure accounts and token-based authentication</li>
            <li><FiBarChart2 className="feature-icon" /> Exportable logs and health statistics</li>
          </ul>
        </section>
      )
    },
    {
      id: 'contact',
      content: (
        <section className="contact card">
          <h2>Contact Us</h2>
          <p>If you have feedback, bugs, or collaboration requests, email us at <a href="mailto:hello@healthcare.example">hello@healthcare.example</a> or use the support link in the footer.</p>
        </section>
      )
    }
  ];

  const [currentSlide, setCurrentSlide] = useState(0);
  const [disableTransition, setDisableTransition] = useState(false);
  const trackRef = useRef(null);

  useEffect(() => {
    const intervalId = setInterval(() => {
      setCurrentSlide(prev => prev + 1);
    }, 4500);
    return () => clearInterval(intervalId);
  }, []);

  const handleTrackTransitionEnd = () => {
    if (currentSlide !== slides.length) return;
    setDisableTransition(true);
    setCurrentSlide(0);

    // Re-enable transition on the next paint so the reset is not visible
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        setDisableTransition(false);
      });
    });
  };

  return (
    <div className="home-container">
      <header className="home-hero">
        <div className="hero-inner">
          <h1 className="home-title">Welcome to Healthcare Chatbot</h1>
          <p className="lead">Your virtual health assistant — ask questions, get guidance, and save conversations for later.</p>
          <div className="hero-actions">
            <button className="btn primary" onClick={onSwitchToSignup}><FiUserPlus style={{marginRight: '0.5rem'}} />Create Account</button>
            <button className="btn outline" onClick={onSwitchToLogin}><FiLogIn style={{marginRight: '0.5rem'}} />Sign In</button>
          </div>
        </div>
      </header>

      <main className="home-main home-slider-main">
        <div className="home-slider">
          <div
            ref={trackRef}
            className={`home-slider-track ${disableTransition ? 'no-transition' : ''}`}
            style={{ transform: `translateX(-${currentSlide * 100}%)` }}
            onTransitionEnd={handleTrackTransitionEnd}
          >
            {[...slides, slides[0]].map((slide, index) => (
              <div className="home-slide" key={slide.id}>
                {slide.content}
              </div>
            ))}
          </div>
          <div className="home-slider-dots">
            {slides.map((slide, index) => (
              <button
                key={slide.id}
                className={`home-slider-dot ${index === (currentSlide % slides.length) ? 'active' : ''}`}
                onClick={() => setCurrentSlide(index)}
                aria-label={`Go to slide ${index + 1}`}
              />
            ))}
          </div>
        </div>
      </main>
    </div>
  );
};

export default Home;