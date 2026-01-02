import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import HealthcareChatbot from './compontents/HealthcareChatbot'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
     <HealthcareChatbot />
     
    </>
  )
}

export default App
