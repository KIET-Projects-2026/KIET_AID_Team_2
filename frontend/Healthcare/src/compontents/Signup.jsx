import React, { useState, useEffect } from 'react';
import { FiLock, FiUser, FiEye, FiEyeOff, FiUserPlus } from 'react-icons/fi';
import axios from 'axios';
import './Auth.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const Signup = ({ onSignupSuccess, onSwitchToLogin }) => {
  const [formData, setFormData] = useState({ username: '', full_name: '', password: '', confirmPassword: '' });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [isAnimating, setIsAnimating] = useState(false);
  const [passwordStrength, setPasswordStrength] = useState(0);

  useEffect(() => { setTimeout(() => setIsAnimating(true), 80); }, []);

  useEffect(() => {
    if (formData.password) {
      let strength = 0;
      if (formData.password.length >= 8) strength += 25;
      if (formData.password.length >= 12) strength += 25;
      if (/[a-z]/.test(formData.password) && /[A-Z]/.test(formData.password)) strength += 25;
      if (/\d/.test(formData.password)) strength += 15;
      if (/[@$!%*?]/.test(formData.password)) strength += 10;
      setPasswordStrength(Math.min(strength, 100));
    } else { setPasswordStrength(0); }
  }, [formData.password]);

  const [usernameAvailable, setUsernameAvailable] = useState(null);
  const [usernameChecking, setUsernameChecking] = useState(false);
  const usernameCheckRef = React.useRef(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    if (error) setError('');

    if (name === 'username') {
      setUsernameAvailable(null);
      if (usernameCheckRef.current) clearTimeout(usernameCheckRef.current);
      const v = value;
      usernameCheckRef.current = setTimeout(async () => {
        if (!v || v.length < 3) { setUsernameAvailable(null); setUsernameChecking(false); return; }
        setUsernameChecking(true);
        try {
          const res = await axios.get(`${API_BASE_URL}/api/auth/exists`, { params: { username: v }, timeout: 5000 });
          setUsernameAvailable(!res.data.exists ? true : false);
        } catch (e) { setUsernameAvailable(null); } finally { setUsernameChecking(false); }
      }, 500);
    }
  };

  const validateForm = () => {
    if (formData.username.length < 3) { setError('Username must be at least 3 characters long'); return false; }
    if (usernameAvailable === false) { setError('Username is already taken'); return false; }
    if (formData.password.length < 6) { setError('Password must be at least 6 characters long'); return false; }
    if (formData.password !== formData.confirmPassword) { setError('Passwords do not match'); return false; }
    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    setIsLoading(true);
    setError('');

    try {
      const signupData = { username: formData.username, password: formData.password, full_name: formData.full_name || undefined };
      const response = await axios.post(`${API_BASE_URL}/api/auth/signup`, signupData, { headers: { 'Content-Type': 'application/json' }, timeout: 10000 });

      if (response.data && response.data.token) {
        localStorage.setItem('token', response.data.token);
        localStorage.setItem('user', JSON.stringify(response.data.user));
        axios.defaults.headers.common['Authorization'] = `Bearer ${response.data.token}`;
        onSignupSuccess(response.data);
      } else { setError('Unexpected response. Please try again.'); }
    } catch (err) {
      let detail = err.response?.data?.detail ?? err.response?.data?.message ?? null;
      if (Array.isArray(detail)) detail = detail.map(d => d.msg || (typeof d === 'string' ? d : JSON.stringify(d))).join('; ');
      if (!detail) { detail = err.response?.status === 400 ? 'Invalid signup data or username already exists' : err.message === 'Network Error' ? `Network error. Is backend running on ${API_BASE_URL}?` : err.message || 'Signup failed.'; }
      setError(detail);
    } finally { setIsLoading(false); }
  };

  const getPasswordStrengthColor = () => { if (passwordStrength < 40) return '#ef4444'; if (passwordStrength < 70) return '#f59e0b'; return '#10b981'; };
  const getPasswordStrengthText = () => { if (passwordStrength < 40) return 'Weak'; if (passwordStrength < 70) return 'Medium'; return 'Strong'; };

  return (
    <div className={`auth-container ${isAnimating ? 'active' : ''}`}>
      <div className="auth-card">
        <div className="auth-header">
          <div className="auth-icon-wrapper signup-icon">
            <FiUserPlus className="auth-icon" />
          </div>
          <h1 className="auth-title">Create Account</h1>
          <p className="auth-subtitle">Join us for personalized health assistance</p>
        </div>

        {error && (<div className="error-message" role="alert">{error}</div>)}

        <form onSubmit={handleSubmit} className="auth-form" aria-label="Signup form">
          <div className="form-group">
            <label className="form-label" htmlFor="signup-username"><FiUser className="label-icon" />Username</label>
            <input id="signup-username" type="text" name="username" placeholder="Enter your username" value={formData.username} onChange={handleChange} className="form-input" disabled={isLoading} minLength="3" required />
            <div className="helper-text">
              {usernameChecking && <small>Checking availability...</small>}
              {!usernameChecking && usernameAvailable === false && <small style={{color: '#ef4444'}}>Username is already taken</small>}
              {!usernameChecking && usernameAvailable === true && <small style={{color: '#10b981'}}>Username is available</small>}
            </div>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="signup-fullname"><FiUser className="label-icon" />Full Name (Optional)</label>
            <input id="signup-fullname" type="text" name="full_name" placeholder="Your full name" value={formData.full_name} onChange={handleChange} className="form-input" disabled={isLoading} />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="signup-password"><FiLock className="label-icon" />Password</label>
            <div className="password-input-wrapper">
              <input id="signup-password" type={showPassword ? 'text' : 'password'} name="password" placeholder="Create a strong password" value={formData.password} onChange={handleChange} className="form-input" disabled={isLoading} minLength="6" required />
              <button type="button" className="password-toggle" onClick={() => setShowPassword(!showPassword)} aria-pressed={showPassword} aria-label="Toggle password visibility">{showPassword ? <FiEyeOff /> : <FiEye />}</button>
            </div>
            {formData.password && (
              <div className="password-strength">
                <div className="password-strength-bar">
                  <div className="password-strength-fill" style={{ width: `${passwordStrength}%`, backgroundColor: getPasswordStrengthColor() }}></div>
                </div>
                <span className="password-strength-text" style={{ color: getPasswordStrengthColor() }}>{getPasswordStrengthText()}</span>
              </div>
            )}
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="signup-confirm"><FiLock className="label-icon" />Confirm Password</label>
            <div className="password-input-wrapper">
              <input id="signup-confirm" type={showConfirmPassword ? 'text' : 'password'} name="confirmPassword" placeholder="Confirm your password" value={formData.confirmPassword} onChange={handleChange} className="form-input" disabled={isLoading} minLength="6" required />
              <button type="button" className="password-toggle" onClick={() => setShowConfirmPassword(!showConfirmPassword)} aria-pressed={showConfirmPassword} aria-label="Toggle password visibility">{showConfirmPassword ? <FiEyeOff /> : <FiEye />}</button>
            </div>
          </div>

          <button type="submit" className={`auth-button ${isLoading ? 'loading' : ''}`} disabled={isLoading} aria-busy={isLoading}>{isLoading ? (<><div className="spinner" />Creating account...</>) : (<><FiUserPlus />Create account</>)}</button>
        </form>

        <div className="auth-footer">
          <p className="auth-switch-text">Already have an account? <button type="button" className="link-button" onClick={onSwitchToLogin} disabled={isLoading}>Sign in</button></p>
        </div>
      </div>
    </div>
  );
};

export default Signup;
