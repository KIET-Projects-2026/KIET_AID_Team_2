import React from 'react';
import './Footer.css';

const Footer = () => {
  return (
    <footer className="site-footer">
      <div className="footer-inner">
        <div className="brand">Healthcare Chatbot</div>
        <nav className="footer-links">
          <a href="/privacy">Privacy</a>
          <a href="mailto:aravindswamymajjuri143gmail.com">Contact</a>
          <a href="/privacy">How to use</a>
        </nav>
        <div className="copyright">© {new Date().getFullYear()} Healthcare. All rights reserved.</div>
      </div>
    </footer>
  );
};

export default Footer;