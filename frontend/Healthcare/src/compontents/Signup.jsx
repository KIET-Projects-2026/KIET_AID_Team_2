import React, { useState, useEffect } from 'react';
import { FiLock, FiUser, FiEye, FiEyeOff, FiUserPlus } from 'react-icons/fi';
import axios from 'axios';
import { toast } from 'react-toastify';
import './Auth.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const Signup = ({ onSignupSuccess, onSwitchToLogin }) => {
  const [formData, setFormData] = useState({
    username: '',
    full_name: '',
    password: '',
    confirmPassword: '',
    phone: '',
    email: '',
    age: '',
    gender: '',
    allergies: '',
    emergencyContact: '',
    emergencyEmail: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
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
    if (!formData.username || formData.username.length < 3) { toast.error('Username is required and must be at least 3 characters long'); return false; }
    if (usernameAvailable === false) { toast.error('Username is already taken'); return false; }
    if (formData.password.length < 6) { toast.error('Password must be at least 6 characters long'); return false; }
    if (formData.password !== formData.confirmPassword) { toast.error('Passwords do not match'); return false; }
    if (!formData.email || !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(formData.email)) { toast.error('Valid email is required'); return false; }
    if (!formData.emergencyEmail || !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(formData.emergencyEmail)) { toast.error('Valid emergency email is required'); return false; }
    if (!formData.phone || !/^\d{10,15}$/.test(formData.phone)) { toast.error('Valid phone number is required'); return false; }
    if (!formData.age || isNaN(formData.age) || formData.age < 0 || formData.age > 120) { toast.error('Valid age is required'); return false; }
    if (!formData.gender) { toast.error('Gender is required'); return false; }
    if (!formData.emergencyContact) { toast.error('Emergency contact is required'); return false; }
    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    setIsLoading(true);

    try {
      const signupData = {
        username: formData.username,
        password: formData.password,
        full_name: formData.full_name || undefined,
        phone: formData.phone,
        email: formData.email,
        age: formData.age,
        gender: formData.gender,
        allergies: formData.allergies,
        emergencyContact: formData.emergencyContact,
        emergencyEmail: formData.emergencyEmail
      };
      const response = await axios.post(`${API_BASE_URL}/api/auth/signup`, signupData, { headers: { 'Content-Type': 'application/json' }, timeout: 10000 });

      if (response.data && response.data.token) {
        localStorage.setItem('token', response.data.token);
        localStorage.setItem('user', JSON.stringify(response.data.user));
        axios.defaults.headers.common['Authorization'] = `Bearer ${response.data.token}`;
        toast.success('✨ Account created successfully!');
        onSignupSuccess(response.data);
      } else { toast.error('Unexpected response. Please try again.'); }
    } catch (err) {
      let detail = err.response?.data?.detail ?? err.response?.data?.message ?? null;
      if (Array.isArray(detail)) detail = detail.map(d => d.msg || (typeof d === 'string' ? d : JSON.stringify(d))).join('; ');
      if (!detail) { detail = err.response?.status === 400 ? 'Invalid signup data or username already exists' : err.message === 'Network Error' ? `Network error. Is backend running on ${API_BASE_URL}?` : err.message || 'Signup failed.'; }
      toast.error(detail);
    } finally { setIsLoading(false); }
  };

  const getPasswordStrengthColor = () => { if (passwordStrength < 40) return '#ef4444'; if (passwordStrength < 70) return '#f59e0b'; return '#10b981'; };
  const getPasswordStrengthText = () => { if (passwordStrength < 40) return 'Weak'; if (passwordStrength < 70) return 'Medium'; return 'Strong'; };

  return (
    <div className={`auth-container ${isAnimating ? 'active' : ''}`}>
      <div className="auth-card">
        <div className="auth-header">
          <h1 className="auth-title">Create Account</h1>
          <p className="auth-subtitle">Join us for personalized health assistance</p>
        </div>


        <form onSubmit={handleSubmit} className="auth-form" aria-label="Signup form">
          <div className="form-group">
            <label className="form-label" htmlFor="signup-username"><FiUser className="label-icon" />Username <span style={{color: 'red'}}>*</span></label>
            <input id="signup-username" type="text" name="username" placeholder="Enter your username" value={formData.username} onChange={handleChange} className="form-input" disabled={isLoading} minLength="3" required />
            <div className="helper-text">
              {usernameChecking && <small>Checking availability...</small>}
              {!usernameChecking && usernameAvailable === false && <small style={{color: '#ef4444'}}>Username is already taken</small>}
              {!usernameChecking && usernameAvailable === true && <small style={{color: '#10b981'}}>Username is available</small>}
            </div>
          </div>
          <div className="form-group">
            <label className="form-label" htmlFor="signup-emergency-email">Emergency Email <span style={{color: 'red'}}>*</span></label>
            <input id="signup-emergency-email" type="email" name="emergencyEmail" placeholder="Enter emergency email" value={formData.emergencyEmail} onChange={handleChange} className="form-input" disabled={isLoading} required />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="signup-fullname"><FiUser className="label-icon" />Full Name (Optional)</label>
            <input id="signup-fullname" type="text" name="full_name" placeholder="Your full name" value={formData.full_name} onChange={handleChange} className="form-input" disabled={isLoading} />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="signup-email">Email</label>
            <input id="signup-email" type="email" name="email" placeholder="Enter your email" value={formData.email} onChange={handleChange} className="form-input" disabled={isLoading} required />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="signup-phone">Phone Number</label>
            <input id="signup-phone" type="tel" name="phone" placeholder="Enter your phone number" value={formData.phone} onChange={handleChange} className="form-input" disabled={isLoading} required pattern="\d{10,15}" />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="signup-age">Age</label>
            <input id="signup-age" type="number" name="age" placeholder="Enter your age" value={formData.age} onChange={handleChange} className="form-input" disabled={isLoading} min="0" max="120" required />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="signup-gender">Gender</label>
            <select id="signup-gender" name="gender" value={formData.gender} onChange={handleChange} className="form-input" disabled={isLoading} required>
              <option value="">Select gender</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="other">Other</option>
              <option value="prefer_not_to_say">Prefer not to say</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="signup-allergies">Allergies (Medical Conditions)</label>
            <input id="signup-allergies" type="text" name="allergies" placeholder="List allergies or medical conditions" value={formData.allergies} onChange={handleChange} className="form-input" disabled={isLoading} />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="signup-emergency">Emergency Contact</label>
            <input id="signup-emergency" type="text" name="emergencyContact" placeholder="Emergency contact details" value={formData.emergencyContact} onChange={handleChange} className="form-input" disabled={isLoading} required />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="signup-password"><FiLock className="label-icon" />Password</label>
            <div className="password-input-wrapper">
              <input
                id="signup-password"
                type="password"
                name="password"
                placeholder="Create a strong password"
                value={formData.password}
                onChange={handleChange}
                className="form-input"
                disabled={isLoading}
                minLength="6"
                required
                style={{ paddingRight: '0.5rem' }}
              />
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
              <input
                id="signup-confirm"
                type="password"
                name="confirmPassword"
                placeholder="Confirm your password"
                value={formData.confirmPassword}
                onChange={handleChange}
                className="form-input"
                disabled={isLoading}
                minLength="6"
                required
                style={{ paddingRight: '0.5rem' }}
              />
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
