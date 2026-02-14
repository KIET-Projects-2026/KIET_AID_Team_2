import React from 'react';
import './HealthDashboard.css';

const HealthDashboard = ({ messages, onClose }) => {
  const getDashboardStats = () => {
    const totalMessages = messages.length;
    const userMessages = messages.filter(m => m.type === 'user').length;
    const botMessages = messages.filter(m => m.type === 'bot').length;
    
    // Count voice queries (messages with voice indicators)
    const voiceQueries = messages.filter(m => m.isVoice).length;
    
    // Count questions (messages ending with ?)
    const questionsAsked = userMessages;

    // Topic frequency (analyze message keywords)
    const keywords = ['diabetes', 'heart', 'covid', 'allergy', 'headache', 'fever', 'pain', 'sleep', 'stress', 'weight'];
    const topicCounts = {};
    keywords.forEach(topic => {
      topicCounts[topic] = messages.filter(m => 
        m.text?.toLowerCase().includes(topic)
      ).length;
    });

    const topTopics = Object.entries(topicCounts)
      .filter(([_, count]) => count > 0)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);

    return {
      totalMessages,
      conversations: 1, // Placeholder - would come from backend
      voiceQueries,
      questionsAsked,
      topTopics: topTopics.length > 0 ? topTopics : [['general', userMessages]],
    };
  };

  const stats = getDashboardStats();

  return (
    <div className="dashboard-panel">
      <div className="dashboard-header">
        <h3>📊 Health Dashboard</h3>
        <button className="dashboard-close" onClick={onClose}>✕</button>
      </div>

      <div className="dashboard-grid">
        <div className="stat-card">
          <div className="stat-number">{stats.totalMessages}</div>
          <div className="stat-label">Total Messages</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">{stats.voiceQueries}</div>
          <div className="stat-label">Voice Queries</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">{stats.questionsAsked}</div>
          <div className="stat-label">Questions Asked</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">{stats.conversations}</div>
          <div className="stat-label">Conversations</div>
        </div>
      </div>

      {/* Topic Frequency */}
      <div className="dashboard-topics">
        <h4>Health Topics Discussed</h4>
        <div className="topic-bars">
          {stats.topTopics.map(([topic, count], idx) => {
            const maxCount = Math.max(...stats.topTopics.map(t => t[1]));
            const percentage = (count / maxCount) * 100;
            return (
              <div key={idx} className="topic-bar-item">
                <div className="topic-name">{topic}</div>
                <div className="topic-bar">
                  <div className="topic-bar-fill" style={{ width: `${percentage}%` }} />
                </div>
                <div className="topic-count">{count}</div>
              </div>
            );
          })}
        </div>
      </div>


    </div>
  );
};

export default HealthDashboard;
