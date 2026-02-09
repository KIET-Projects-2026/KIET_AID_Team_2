import React, { useEffect, useState } from 'react';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const Profile = () => {
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
        emergencyContact: editData.emergencyContact,
        emergencyEmail: editData.emergencyEmail,
      };
      const res = await axios.patch(`${API_BASE_URL}/api/auth/me`, updateFields, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.data && res.data.user) {
        setUser(res.data.user);
        setEditMode(false);
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
      <h2>User Profile</h2>
      <div className="profile-details">
        <p><strong>Username:</strong> {user.username}</p>
        <p><strong>Email:</strong> {user.email}</p>
        <p><strong>Phone:</strong> {user.phone}</p>
        {!editMode && (
          <p><strong>Emergency Contact:</strong> {user.emergencyContact}</p>
        )}
        {editMode ? (
          <>
            <label><strong>Full Name:</strong> <input name="full_name" value={editData.full_name || ''} onChange={handleChange} /></label><br />
            <label><strong>Age:</strong> <input name="age" value={editData.age || ''} onChange={handleChange} type="number" min="0" max="120" /></label><br />
            <label><strong>Gender:</strong> <input name="gender" value={editData.gender || ''} onChange={handleChange} /></label><br />
            <label><strong>Allergies:</strong> <input name="allergies" value={editData.allergies || ''} onChange={handleChange} /></label><br />
            <label><strong>Emergency Contact:</strong> <input name="emergencyContact" value={editData.emergencyContact || ''} onChange={handleChange} /></label><br />
            <label><strong>Emergency Email:</strong> <input name="emergencyEmail" value={editData.emergencyEmail || ''} onChange={handleChange} /></label><br />
            {saveError && <div style={{ color: 'red' }}>{saveError}</div>}
            <button onClick={handleSave} disabled={saving}>{saving ? 'Saving...' : 'Save'}</button>
            <button onClick={handleCancel} disabled={saving}>Cancel</button>
          </>
        ) : (
          <>
            <p><strong>Full Name:</strong> {user.full_name || '-'}</p>
            <p><strong>Age:</strong> {user.age}</p>
            <p><strong>Gender:</strong> {user.gender}</p>
            <p><strong>Allergies:</strong> {user.allergies}</p>
            <p><strong>Emergency Contact:</strong> {user.emergencyContact}</p>
            <p><strong>Emergency Email:</strong> {user.emergencyEmail}</p>
            <button onClick={handleEdit}>Edit Profile</button>
          </>
        )}
      </div>
    </div>
  );
};

export default Profile;
