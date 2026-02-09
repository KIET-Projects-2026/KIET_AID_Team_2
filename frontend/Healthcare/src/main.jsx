import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// Force light theme (white background) across the app for a clean, consistent UI
if (typeof document !== 'undefined') {
  document.documentElement.setAttribute('data-color-scheme', 'light');
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
