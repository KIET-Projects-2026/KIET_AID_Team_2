import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'
import HealthcareChatbot from './compontents/HealthcareChatbot'
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
          <HealthcareChatbot 
            currentUser={currentUser}
            onLogout={handleLogout}
          />
          <Footer />
        </>
      )}
    </>
  )
}

export default App
