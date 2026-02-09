import React from 'react';
import './Home.css';

const Home = ({ onSwitchToLogin, onSwitchToSignup }) => {
  return (
    <div className="home-container">
      <header className="home-hero">
        <div className="hero-inner">
          <h1>🏥 Healthcare Chatbot</h1>
          <p className="lead">Your virtual health assistant — ask questions, get guidance, and save conversations for later.</p>
          <div className="hero-actions">
            <button className="btn primary" onClick={onSwitchToSignup}>Create Account</button>
            <button className="btn outline" onClick={onSwitchToLogin}>Sign In</button>
          </div>
        </div>
      </header>

      <main className="home-main">
        <section className="how-to">
          <h2>How to use this bot</h2>
          <ol>
            <li>Create an account or sign in.</li>
            <li>Start a new conversation or continue an existing one from the Chats sidebar.</li>
            <li>Type or speak your question and get an AI-powered response.</li>
            <li>Tap a conversation to revisit all messages — the first message is always preserved.</li>
          </ol>
        </section>

        <section className="features">
          <h2>Features</h2>
          <ul>
            <li>Text and voice input</li>
            <li>Persistent chat history grouped by conversation</li>
            <li>Secure accounts and token-based authentication</li>
            <li>Exportable logs and health statistics</li>
          </ul>
        </section>

        <section className="contact">
          <h2>Contact Us</h2>
          <p>If you have feedback, bugs, or collaboration requests, email us at <a href="mailto:hello@healthcare.example">hello@healthcare.example</a> or use the support link in the footer.</p>
        </section>
      </main>
    </div>
  );
};

export default Home;