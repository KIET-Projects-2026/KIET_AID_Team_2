// =====================================================================
// Feature Components Export Index
// =====================================================================

export { default as EmergencySOS } from './EmergencySOS';
export { default as HealthTips } from './HealthTips';
export { default as HealthDashboard } from './HealthDashboard';
export { default as ChatExport } from './ChatExport';
export { default as PersonalizedInsights } from './PersonalizedInsights';

// Utilities
export {
  EMERGENCY_KEYWORDS,
  MOODS,
  HEALTH_TIPS,
  checkEmergency,
  exportChatTxt,
  getDashboardStats,
} from './featureUtils';

// Context Builder - User data analysis for personalized responses
export {
  buildUserContext,
  generateContextualSystemPrompt,
  getPersonalizedTips,
  formatContextForAPI,
} from './contextBuilder';

// Response Enhancer - Mood-aware response enhancements
export {
  enhanceResponseWithContext,
  formatTipsSection,
  generateWellnessReminder,
  detectUrgency,
  generateFollowUpQuestions,
  getQuickActions,
  getEncouragementMessage,
} from './responseEnhancer';

// Tips Database - Comprehensive mood + symptom based tips
export {
  PERSONALIZED_TIPS_DB,
  getTipsByMoodAndSymptoms,
  getRecoveryProtocol,
  getTopicTips,
} from './tipsDatabase';
