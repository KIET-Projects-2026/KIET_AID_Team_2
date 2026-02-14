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
  // new: wizard step state (0..3)
  const [step, setStep] = useState(0);

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

  const handleNext = () => {
    const err = validateStep(step);
    if (err) { toast.dismiss(); toast.error(err); return; }
    setStep(s => Math.min(3, s + 1));
  };
  const handleBack = () => setStep(s => Math.max(0, s - 1));

  // allow clicking the stepper; forward moves are guarded by per-step validation
  const handleStepClick = (index) => {
    if (index === step) return;
    if (index < step) { setStep(index); return; }
    // moving forward: validate each intermediate step first
    for (let s = step; s < index; s++) {
      const err = validateStep(s);
      if (err) { toast.dismiss(); toast.error(err); return; }
    }
    setStep(index);
  };

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

  // Return a single error message string when invalid, otherwise null
  const validateForm = () => {
    if (!formData.username || formData.username.length < 3) return 'Username is required and must be at least 3 characters long';
    if (usernameAvailable === false) return 'Username is already taken';
    if (formData.password.length < 6) return 'Password must be at least 6 characters long';
    if (formData.password !== formData.confirmPassword) return 'Passwords do not match';
    if (!formData.email || !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(formData.email)) return 'Valid email is required';
    if (!formData.emergencyEmail || !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(formData.emergencyEmail)) return 'Valid emergency email is required';
    if (!formData.phone || !/^\d{10,15}$/.test(formData.phone)) return 'Valid phone number is required';
    if (!formData.age || isNaN(formData.age) || formData.age < 0 || formData.age > 120) return 'Valid age is required';
    if (!formData.gender) return 'Gender is required';
    if (!formData.emergencyContact) return 'Emergency contact is required';
    return null;
  };

  // per-step validation for wizard — return an error message or null
  const validateStep = (s) => {
    if (s === 0) {
      if (!formData.username || formData.username.length < 3) return 'Please choose a username (min 3 chars)';
      if (usernameAvailable === false) return 'Username already taken';
      if (!formData.password || formData.password.length < 6) return 'Enter a password (min 6 chars)';
      if (formData.password !== formData.confirmPassword) return 'Passwords must match';
      return null;
    }
    if (s === 1) {
      if (!formData.email || !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(formData.email)) return 'Please enter a valid email';
      if (!formData.phone || !/^\d{10,15}$/.test(formData.phone)) return 'Please enter a valid phone number';
      if (!formData.age || isNaN(formData.age) || formData.age < 0 || formData.age > 120) return 'Please enter a valid age';
      return null;
    }
    if (s === 2) {
      if (!formData.emergencyContact) return 'Please provide an emergency contact';
      if (!formData.emergencyEmail || !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(formData.emergencyEmail)) return 'Please provide a valid emergency email';
      return null;
    }
    return null;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const err = validateForm();
    if (err) { toast.dismiss(); toast.error(err); return; }

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
        toast.dismiss();
        toast.success('✨ Account created successfully!');
        onSignupSuccess(response.data);
      } else { toast.dismiss(); toast.error('Unexpected response. Please try again.'); }
    } catch (err) {
      let detail = err.response?.data?.detail ?? err.response?.data?.message ?? null;
      if (Array.isArray(detail)) detail = detail.map(d => d.msg || (typeof d === 'string' ? d : JSON.stringify(d))).join('; ');
      if (!detail) { detail = err.response?.status === 400 ? 'Invalid signup data or username already exists' : err.message === 'Network Error' ? `Network error. Is backend running on ${API_BASE_URL}?` : err.message || 'Signup failed.'; }
      toast.dismiss();
      toast.error(detail);
    } finally { setIsLoading(false); }
  };

  const getPasswordStrengthColor = () => { if (passwordStrength < 40) return '#ef4444'; if (passwordStrength < 70) return '#f59e0b'; return '#10b981'; };
  const getPasswordStrengthText = () => { if (passwordStrength < 40) return 'Weak'; if (passwordStrength < 70) return 'Medium'; return 'Strong'; };

  return (
    <div className={`auth-container ${isAnimating ? 'active' : ''}`}>
      <div className="auth-card">
        <div className="auth-hero">
          <div className="hero-icon">🩺</div>
          <h2 className="hero-title">Create Account</h2>
          <p className="hero-sub">Join us for personalized health assistance</p>
        </div>


        {/* Stepper */}
        <div className="signup-stepper" role="tablist" aria-label="Signup steps">
          <div role="tab" tabIndex={0} onClick={() => handleStepClick(0)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') handleStepClick(0); }} className={`step ${step === 0 ? 'active' : step > 0 ? 'completed' : ''}`}>Account</div>
          <div role="tab" tabIndex={0} onClick={() => handleStepClick(1)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') handleStepClick(1); }} className={`step ${step === 1 ? 'active' : step > 1 ? 'completed' : ''}`}>Personal</div>
          <div role="tab" tabIndex={0} onClick={() => handleStepClick(2)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') handleStepClick(2); }} className={`step ${step === 2 ? 'active' : step > 2 ? 'completed' : ''}`}>Medical</div>
          <div role="tab" tabIndex={0} onClick={() => handleStepClick(3)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') handleStepClick(3); }} className={`step ${step === 3 ? 'active' : ''}`}>Review</div>
        </div>

        <form onSubmit={handleSubmit} className="auth-form" aria-label="Signup wizard form">
          {step === 0 && (
            <>
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
                <label className="form-label" htmlFor="signup-password"><FiLock className="label-icon" />Password</label>
                <div className="password-input-wrapper">
                  <input
                    id="signup-password"
                    type={showPassword ? 'text' : 'password'}
                    name="password"
                    placeholder="Create a strong password"
                    value={formData.password}
                    onChange={handleChange}
                    className="form-input"
                    disabled={isLoading}
                    minLength="6"
                    required
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
                <label className="form-label" htmlFor="signup-confirm-password">Confirm Password</label>
                <div className="password-input-wrapper">
                  <input
                    id="signup-confirm-password"
                    type={showConfirmPassword ? 'text' : 'password'}
                    name="confirmPassword"
                    placeholder="Confirm your password"
                    value={formData.confirmPassword}
                    onChange={handleChange}
                    className="form-input"
                    disabled={isLoading}
                    minLength="6"
                    required
                  />
                </div>
              </div>

              <div className="form-actions step-actions">
                <div />
                <button type="button" className="btn" onClick={handleNext}>Next →</button>
              </div>
            </>
          )}

          {step === 1 && (
            <>
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

              <div className="form-actions step-actions">
                <button type="button" className="btn btn-outline" onClick={handleBack}>← Back</button>
                <button type="button" className="btn" onClick={handleNext}>Next →</button>
              </div>
            </>
          )}

          {step === 2 && (
            <>
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
                <label className="form-label" htmlFor="signup-emergency-email">Emergency Email <span style={{color: 'red'}}>*</span></label>
                <input id="signup-emergency-email" type="email" name="emergencyEmail" placeholder="Enter emergency email" value={formData.emergencyEmail} onChange={handleChange} className="form-input" disabled={isLoading} required />
              </div>

              <div className="form-actions step-actions">
                <button type="button" className="btn btn-outline" onClick={handleBack}>← Back</button>
                <button type="button" className="btn" onClick={handleNext}>Next →</button>
              </div>
            </>
          )}

          {step === 3 && (
            <>
              <div className="form-group">
                <h3>Review your details</h3>
                <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12}}>
                  <div><strong>Username</strong><div>{formData.username}</div></div>
                  <div><strong>Full name</strong><div>{formData.full_name || '-'}</div></div>
                  <div><strong>Email</strong><div>{formData.email}</div></div>
                  <div><strong>Phone</strong><div>{formData.phone}</div></div>
                  <div><strong>Age</strong><div>{formData.age}</div></div>
                  <div><strong>Gender</strong><div>{formData.gender}</div></div>
                  <div style={{gridColumn: '1 / -1'}}><strong>Allergies</strong><div>{formData.allergies || '-'}</div></div>
                </div>
              </div>

              <div className="form-actions step-actions">
                <button type="button" className="btn btn-outline" onClick={handleBack}>← Back</button>
                <button type="submit" className={`auth-button ${isLoading ? 'loading' : ''}`} disabled={isLoading} aria-busy={isLoading}>{isLoading ? (<><div className="spinner" />Creating account...</>) : (<><FiUserPlus />Create account</>)}</button>
              </div>
            </>
          )}
          <div className="auth-footer">
            <p className="auth-switch-text">Already have an account? <button type="button" className="auth-switch-button" onClick={onSwitchToLogin} disabled={isLoading}>Sign in</button></p>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Signup;
