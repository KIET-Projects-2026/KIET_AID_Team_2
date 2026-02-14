import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './Profile.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Get initials from name for avatar
const getInitials = (name) => {
  if (!name) return '?';
  const parts = name.trim().split(' ');
  if (parts.length >= 2) {
    return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
  }
  return name.substring(0, 2).toUpperCase();
};

const Profile = ({ onLogout }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [editMode, setEditMode] = useState(false);
  const [editData, setEditData] = useState({});
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(null);

  // track whether any changes were made compared to the loaded user
  const isDirty = JSON.stringify(editData || {}) !== JSON.stringify(user || {});

  useEffect(() => {
    const fetchProfile = async () => {
      setLoading(true);
      setError(null);
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          setError('Not authenticated');
          setLoading(false);
          return;
        }
        const res = await axios.get(`${API_BASE_URL}/api/auth/me`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (res.data && res.data.user) {
          setUser(res.data.user);
          setEditData(res.data.user);
          try {
            localStorage.setItem('user', JSON.stringify(res.data.user));
            window.dispatchEvent(new CustomEvent('profileUpdated', { detail: res.data.user }));
          } catch (e) {
            console.warn('Failed to persist fetched user to localStorage', e);
          }
        } else {
          setError('Failed to fetch user details');
        }
      } catch (err) {
        setError('Failed to fetch user details');
      } finally {
        setLoading(false);
      }
    };
    fetchProfile();
  }, []);

  const handleEdit = () => {
    setEditData(user);
    setEditMode(true);
    setSaveError(null);
  };

  const handleBack = () => {
    if (editMode) {
      // In edit mode, cancel editing and return to profile view
      setEditMode(false);
      setEditData(user);
      setSaveError(null);
    } else {
      // Not in edit mode, close the profile
      window.dispatchEvent(new CustomEvent('closeProfile'));
    }
  };

  const handleCancel = () => {
    setEditMode(false);
    setEditData(user);
    setSaveError(null);
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setEditData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSave = async () => {
    setSaving(true);
    setSaveError(null);
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        setSaveError('Not authenticated');
        setSaving(false);
        return;
      }
      const updateFields = {
        full_name: editData.full_name,
        age: editData.age,
        gender: editData.gender,
        allergies: editData.allergies,
        emergencyEmail: editData.emergencyEmail,
      };
      const res = await axios.patch(`${API_BASE_URL}/api/auth/me`, updateFields, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.data && res.data.user) {
        setUser(res.data.user);
        setEditMode(false);
        try {
          localStorage.setItem('user', JSON.stringify(res.data.user));
          window.dispatchEvent(new CustomEvent('profileUpdated', { detail: res.data.user }));
        } catch (e) {
          console.warn('Failed to persist updated user to localStorage', e);
        }
      } else {
        setSaveError('Failed to update profile');
      }
    } catch (err) {
      setSaveError('Failed to update profile');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="profile-page">
        <div className="profile-loading">
          <div className="loading-spinner"></div>
          <p>Loading profile...</p>
        </div>
      </div>
    );
  }
  if (error) {
    return (
      <div className="profile-page">
        <div className="profile-error">
          <span className="error-icon">⚠</span>
          <p>{error}</p>
        </div>
      </div>
    );
  }
  if (!user) {
    return (
      <div className="profile-page">
        <div className="profile-error">
          <span className="error-icon">📋</span>
          <p>No user data found.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="profile-page">
      <div className="profile-header-bar">
        <button className="back-btn" onClick={handleBack}>
          <span className="back-arrow">←</span> Back
        </button>
        <h2>My Profile</h2>
        <div className="header-spacer"></div>
      </div>

      <div className="profile-content">
        {/* Profile Avatar Section */}
        <div className="profile-avatar-section">
          <div className="profile-avatar">
            <span className="avatar-initials">{getInitials(user.full_name || user.username)}</span>
            <div className="avatar-status"></div>
          </div>
          <h3 className="profile-name">{user.full_name || user.username}</h3>
          <p className="profile-email">{user.email}</p>
          {!editMode && (
            <button onClick={handleEdit} className="btn-edit-profile">
              <span>✏️</span> Edit Profile
            </button>
          )}

          {editMode && (
            <div className="avatar-edit-actions">
              <button className="btn-save avatar-save" onClick={handleSave} disabled={!isDirty || saving}>
                {saving ? (<><span className="btn-spinner" /> Saving...</>) : (<>✓ Save</>)}
              </button>
              <button className="btn-cancel avatar-cancel" onClick={handleCancel} disabled={saving}>✕ Cancel</button>
            </div>
          )}
        </div>

        {/* Profile Details Cards */}
        <div className="profile-cards-container">
          {/* Personal Information Card */}
          <div className="profile-card-modern">
            <div className="card-header">
              <span className="card-icon">👤</span>
              <h4>Personal Information</h4>
            </div>
            <div className="card-content">
              {editMode ? (
                <div className="edit-form-grid">
                  <div className="form-group">
                    <label>Full Name</label>
                    <input 
                      type="text" 
                      name="full_name" 
                      value={editData.full_name || ''} 
                      onChange={handleChange} 
                      placeholder="Enter your full name"
                    />
                  </div>
                  <div className="form-group">
                    <label>Age</label>
                    <input 
                      type="number" 
                      name="age" 
                      value={editData.age || ''} 
                      onChange={handleChange} 
                      min="0" 
                      max="120" 
                      placeholder="Enter age"
                    />
                  </div>
                  <div className="form-group">
                    <label>Gender</label>
                    <input 
                      type="text" 
                      name="gender" 
                      value={editData.gender || ''} 
                      onChange={handleChange} 
                      placeholder="Enter gender"
                    />
                  </div>
                </div>
              ) : (
                <div className="info-grid">
                  <div className="info-item">
                    <span className="info-label">Username</span>
                    <span className="info-value">{user.username}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Full Name</span>
                    <span className="info-value">{user.full_name || '-'}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Age</span>
                    <span className="info-value">{user.age || '-'}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Gender</span>
                    <span className="info-value">{user.gender || '-'}</span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Contact Information Card */}
          <div className="profile-card-modern">
            <div className="card-header">
              <span className="card-icon">📞</span>
              <h4>Contact Information</h4>
            </div>
            <div className="card-content">
              <div className="info-grid">
                <div className="info-item">
                  <span className="info-label">Email</span>
                  <span className="info-value">{user.email}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Phone</span>
                  <span className="info-value">{user.phone || '-'}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Emergency Contact</span>
                  <span className="info-value">{user.emergencyContact || '-'}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Medical Information Card */}
          <div className="profile-card-modern">
            <div className="card-header">
              <span className="card-icon">🏥</span>
              <h4>Medical Information</h4>
            </div>
            <div className="card-content">
              {editMode ? (
                <div className="edit-form-grid">
                  <div className="form-group full-width">
                    <label>Allergies</label>
                    <input 
                      type="text" 
                      name="allergies" 
                      value={editData.allergies || ''} 
                      onChange={handleChange} 
                      placeholder="Enter any allergies"
                    />
                  </div>
                  <div className="form-group full-width">
                    <label>Emergency Email</label>
                    <input 
                      type="email" 
                      name="emergencyEmail" 
                      value={editData.emergencyEmail || ''} 
                      onChange={handleChange} 
                      placeholder="Enter emergency email"
                    />
                  </div>
                </div>
              ) : (
                <div className="info-grid">
                  <div className="info-item">
                    <span className="info-label">Allergies</span>
                    <span className="info-value medical">{user.allergies || '-'}</span>
                  </div>
                  <div className="info-item full-width">
                    <span className="info-label">Emergency Email</span>
                    <span className="info-value">{user.emergencyEmail || '-'}</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        {editMode && (
          <div className="profile-actions">
            <button 
              className="btn-save" 
              onClick={handleSave} 
              disabled={saving}
            >
              {saving ? (
                <><span className="btn-spinner"></span> Saving...</>
              ) : (
                <><span>✓</span> Save Changes</>
              )}
            </button>
            <button 
              className="btn-cancel" 
              onClick={handleCancel} 
              disabled={saving}
            >
              <span>✕</span> Cancel
            </button>
          </div>
        )}

        {saveError && (
          <div className="profile-message error">
            <span>⚠</span> {saveError}
          </div>
        )}
      </div>
    </div>
  );
};

export default Profile;
