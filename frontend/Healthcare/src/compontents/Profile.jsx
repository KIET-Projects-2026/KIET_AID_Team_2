import React, { useEffect, useState } from 'react';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const Profile = ({ onLogout }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [editMode, setEditMode] = useState(false);
  const [editData, setEditData] = useState({});
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(null);

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
          // Ensure other components (App, Chat) see the latest profile immediately
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
      // Only send editable fields
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
        // Persist updated user to localStorage so the rest of the app sees the change
        try {
          localStorage.setItem('user', JSON.stringify(res.data.user));
          // Notify other components the profile changed
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
    return <div className="profile-container">Loading profile...</div>;
  }
  if (error) {
    return <div className="profile-container">{error}</div>;
  }
  if (!user) {
    return <div className="profile-container">No user data found.</div>;
  }

  return (
    <div className="profile-container">
      <div className="profile-header">
        <button className="back-btn" onClick={() => window.dispatchEvent(new CustomEvent('closeProfile'))}>◀ Back</button>
        <h2>User Profile</h2>

        {/* action buttons moved to header for better alignment */}
        <div className="profile-actions header-actions">
          {!editMode ? (
            <button onClick={handleEdit} className="btn primary edit-btn" aria-label="Edit profile">Edit Profile</button>
          ) : (
            <>
              <button className="btn primary" onClick={handleSave} disabled={saving}>{saving ? 'Saving...' : 'Save changes'}</button>
              <button className="btn btn-outline" onClick={handleCancel} disabled={saving}>Cancel</button>
            </>
          )}
        </div>
      </div>

      <div className="profile-card">
        <div className="profile-details">
          <div className="profile-grid">
            <div>
              <label>Username</label>
              <p>{user.username}</p>
            </div>

            <div>
              <label>Email</label>
              <p>{user.email}</p>
            </div>

            <div>
              <label>Phone</label>
              <p>{user.phone || '-'}</p>
            </div>

            <div>
              <label>Emergency Contact</label>
              <p>{user.emergencyContact || '-'}</p>
            </div>

            {editMode ? (
              <>
                <div>
                  <label>Full name</label>
                  <input className="form-input" name="full_name" value={editData.full_name || ''} onChange={handleChange} />
                </div>
                <div>
                  <label>Age</label>
                  <input className="form-input" name="age" value={editData.age || ''} onChange={handleChange} type="number" min="0" max="120" />
                </div>
                <div>
                  <label>Gender</label>
                  <input className="form-input" name="gender" value={editData.gender || ''} onChange={handleChange} />
                </div>
                <div>
                  <label>Allergies</label>
                  <input className="form-input" name="allergies" value={editData.allergies || ''} onChange={handleChange} />
                </div>
                <div style={{ gridColumn: '1 / -1' }}>
                  <label>Emergency Email</label>
                  <input className="form-input" name="emergencyEmail" value={editData.emergencyEmail || ''} onChange={handleChange} />
                </div>
              </>
            ) : (
              <>
                <div>
                  <label>Full name</label>
                  <p>{user.full_name || '-'}</p>
                </div>
                <div>
                  <label>Age</label>
                  <p>{user.age || '-'}</p>
                </div>
                <div>
                  <label>Gender</label>
                  <p>{user.gender || '-'}</p>
                </div>
                <div>
                  <label>Allergies</label>
                  <p>{user.allergies || '-'}</p>
                </div>
                <div style={{ gridColumn: '1 / -1' }}>
                  <label>Emergency Email</label>
                  <p>{user.emergencyEmail || '-'}</p>
                </div>
              </>
            )}
          </div>

          {saveError && <div className="error" style={{ marginTop: 12 }}>{saveError}</div>}
        </div>
      </div>
    </div>
  );
};

export default Profile;
