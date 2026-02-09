import React from 'react';

const ChatExport = ({ messages }) => {
  const exportChatPdf = (format = 'txt') => {
    if (messages.length === 0) {
      alert('No messages to export!');
      return;
    }

    let content = `Healthcare Chatbot Conversation Export\n`;
    content += `Date: ${new Date().toLocaleString()}\n`;
    content += `Total Messages: ${messages.length}\n`;
    content += `${'='.repeat(60)}\n\n`;

    messages.forEach((msg) => {
      const timestamp = msg.timestamp?.toLocaleString() || new Date().toLocaleString();
      const sender = msg.type === 'user' ? '👤 You' : '🤖 Assistant';
      content += `[${timestamp}] ${sender}:\n${msg.text}\n\n`;
    });

    const element = document.createElement('a');
    element.setAttribute('href', `data:text/plain;charset=utf-8,${encodeURIComponent(content)}`);
    element.setAttribute('download', `healthcare_chat_${new Date().getTime()}.txt`);
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  return { exportChatPdf };
};

export default ChatExport;
