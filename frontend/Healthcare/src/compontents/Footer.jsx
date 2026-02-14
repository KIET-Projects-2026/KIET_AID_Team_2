import React, { useEffect, useState } from 'react';
import './Footer.css';

const PRIVACY_TEXT = `We respect your privacy. This demo stores session data (chat history and profile) locally
in the browser and in simple JSON files on the server (for demo/testing). Authentication uses
JWT tokens stored in localStorage for convenience. Do not store highly sensitive personal data here.
Third-party integrations (e.g. Gemini) are optional and only used if configured. For questions or
data removal requests contact hello@healthcare.example.`;

const HOWTO_TEXT = `How to use the Healthcare Chatbot:\n\n1. Create an account or sign in.\n2. Start a new conversation or continue an existing one from the sidebar.\n3. Type or use the microphone (voice input) — you can also use the wake-word to open the assistant.\n4. Click any conversation to view full history; export logs if needed.\n\nNotes: audio features require microphone permission; chat history is saved locally/ server-side for your account.`;

const Footer = () => {
  const [modalOpen, setModalOpen] = useState(false);
  const [modalType, setModalType] = useState(''); // 'privacy' | 'howto'

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') setModalOpen(false); };
    if (modalOpen) {
      document.addEventListener('keydown', onKey);
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => { document.removeEventListener('keydown', onKey); document.body.style.overflow = ''; };
  }, [modalOpen]);

  const openModal = (type) => { setModalType(type); setModalOpen(true); };
  const closeModal = () => setModalOpen(false);

  const renderModalContent = () => {
    if (modalType === 'privacy') return (
      <>
        <h2>Privacy</h2>
        <p style={{whiteSpace: 'pre-line'}}>{PRIVACY_TEXT}</p>
      </>
    );
    return (
      <>
        <h2>How to use</h2>
        <p style={{whiteSpace: 'pre-line'}}>{HOWTO_TEXT}</p>
      </>
    );
  };

  return (
    <footer className="site-footer">
      <div className="footer-inner">
        <div className="brand">Healthcare Chatbot</div>
        <nav className="footer-links">
          <button type="button" className="footer-link" onClick={() => openModal('privacy')}>Privacy</button>
          <a href="mailto:hello@healthcare.example">Contact</a>
          <button type="button" className="footer-link" onClick={() => openModal('howto')}>How to use</button>
        </nav>
        <div className="copyright">© {new Date().getFullYear()} Healthcare. All rights reserved.</div>
      </div>

      {modalOpen && (
        <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="footer-modal-title" onClick={(e) => { if (e.target.classList.contains('modal-overlay')) closeModal(); }}>
          <div className="footer-modal">
            <button className="modal-close" onClick={closeModal} aria-label="Close">✕</button>
            <div id="footer-modal-title">
              {renderModalContent()}
            </div>
          </div>
        </div>
      )}
    </footer>
  );
};

export default Footer;