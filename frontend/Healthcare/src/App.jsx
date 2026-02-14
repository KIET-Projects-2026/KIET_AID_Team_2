import { useState, useEffect } from 'react'
import axios from 'axios'
import { ToastContainer } from 'react-toastify'
import 'react-toastify/dist/ReactToastify.css'
import './App.css'
import HealthcareChatbot from './compontents/HealthcareChatbot'
import Profile from './compontents/Profile'
import { FiUser } from 'react-icons/fi'
import Login from './compontents/Login'
import Signup from './compontents/Signup'
import Home from './compontents/Home'
import Footer from './compontents/Footer'

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [currentUser, setCurrentUser] = useState(null)
  const [showLogin, setShowLogin] = useState(true)
  const [showHome, setShowHome] = useState(true)
  const [isLoading, setIsLoading] = useState(true)
  const [showProfile, setShowProfile] = useState(false)

  useEffect(() => {
    // Check if user is already logged in
    const token = localStorage.getItem('token')
    const user = localStorage.getItem('user')
    
    if (token && user) {
      try {
        const parsedUser = JSON.parse(user)
        setCurrentUser(parsedUser)
        setIsAuthenticated(true)
        // Set axios default header
        axios.defaults.headers.common['Authorization'] = `Bearer ${token}`
      } catch (error) {
        console.error('Error parsing user data:', error)
        localStorage.removeItem('token')
        localStorage.removeItem('user')
      }
    }
    setIsLoading(false)
  }, [])
  // Keep `currentUser` in sync when other components update the profile
  useEffect(() => {
    const handler = (e) => {
      if (e && e.detail) setCurrentUser(e.detail);
    };
    window.addEventListener('profileUpdated', handler);

    // listen for requests from child components to open/close the Profile view
    const openProfileHandler = () => setShowProfile(true);
    const closeProfileHandler = () => setShowProfile(false);
    window.addEventListener('openProfile', openProfileHandler);
    window.addEventListener('closeProfile', closeProfileHandler);

    return () => {
      window.removeEventListener('profileUpdated', handler);
      window.removeEventListener('openProfile', openProfileHandler);
      window.removeEventListener('closeProfile', closeProfileHandler);
    };
  }, []);
  const handleLoginSuccess = (data) => {
    setCurrentUser(data.user)
    setIsAuthenticated(true)
  }

  const handleSignupSuccess = (data) => {
    setCurrentUser(data.user)
    setIsAuthenticated(true)
  }

  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    delete axios.defaults.headers.common['Authorization']
    setCurrentUser(null)
    setIsAuthenticated(false)
  }

  const switchToSignup = () => {
    setShowLogin(false)
    setShowHome(false)
  }

  const switchToLogin = () => {
    setShowLogin(true)
    setShowHome(false)
  }

  const goHome = () => {
    setShowHome(true)
  }

  if (isLoading) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
      }}>
        <div style={{
          color: 'white',
          fontSize: '1.5rem',
          fontWeight: '600'
        }}>
          Loading...
        </div>
      </div>
    )
  }

  return (
    <>
      <ToastContainer
        position="top-right"
        autoClose={3000}
        hideProgressBar={false}
        newestOnTop={true}
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="light"
        limit={1} /* ensure only one toast is visible at a time */
      />
      {!isAuthenticated ? (
        showHome ? (
          <>
            <Home onSwitchToLogin={switchToLogin} onSwitchToSignup={switchToSignup} />
            <Footer />
          </>
        ) : (
          showLogin ? (
            <Login 
              onLoginSuccess={handleLoginSuccess} 
              onSwitchToSignup={switchToSignup}
            />
          ) : (
            <Signup 
              onSignupSuccess={handleSignupSuccess} 
              onSwitchToLogin={switchToLogin}
            />
          )
        )
      ) : (
        <>
         
          {showProfile ? (
            <>
              <Profile onLogout={handleLogout} />
            </>
          ) : (
            <HealthcareChatbot 
              currentUser={currentUser}
              onLogout={handleLogout}
            />
          )}
        </>
      )}
    </>
  )
}

export default App
