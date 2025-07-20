// File: static/script.js
// This file contains the updated JavaScript logic.

const chatForm = document.getElementById('chat-form');
const messageInput = document.getElementById('message-input');
const chatMessages = document.getElementById('chat-messages');
const sendButton = document.getElementById('send-button');

function appendMessage(sender, message) {
    const isUser = sender === 'user';
    const messageContainer = document.createElement('div');
    messageContainer.className = `flex items-start gap-4 ${isUser ? 'justify-end' : ''}`;

    // --- HOW TO CHANGE THE BOT AVATAR ---
    // To change the bot's avatar, replace the SVG code inside the 'botIcon'
    // variable with an <img> tag pointing to your image URL.
    // Example: const botIcon = `<img src="https://your-image-url.com/avatar.png" class="w-full h-full rounded-full" />`;
    const botIcon = `<img src="/static/ai_icon.jpg" alt="AI Avatar" class="w-full h-full rounded-full object-cover">`;
    const userIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`;

    const iconContainer = `
        <div class="w-10 h-10 rounded-full flex-shrink-0 flex items-center justify-center ${isUser ? 'bg-blue-600' : 'bg-gray-600'}">
            ${isUser ? userIcon : botIcon}
        </div>`;

    // CHANGED: The bot's message bubble is now 'bg-slate-700' to be more prominent.
    const messageBubble = `
        <div class="p-4 rounded-2xl max-w-md ${isUser ? 'bg-blue-600 rounded-br-none' : 'bg-slate-700 rounded-bl-none'}">
            <p>${message}</p>
        </div>`;
    
    messageContainer.innerHTML = isUser ? messageBubble + iconContainer : iconContainer + messageBubble;
    chatMessages.appendChild(messageContainer);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typing-indicator';
    typingDiv.className = 'flex items-start gap-4';
    typingDiv.innerHTML = `
        <div class="w-10 h-10 rounded-full flex-shrink-0 flex items-center justify-center bg-gray-600">
            <svg xmlns="[http://www.w3.org/2000/svg](http://www.w3.org/2000/svg)" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"/><rect x="4" y="12" width="16" height="8" rx="2"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="M17 12v-2a2 2 0 0 0-2-2h-2a2 2 0 0 0-2 2v2"/></svg>
        </div>
        <div class="bg-slate-700 p-4 rounded-2xl rounded-bl-none max-w-md">
            <p class="animate-pulse">در حال نوشتن...</p>
        </div>`;
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const userMessage = messageInput.value.trim();
    if (!userMessage) return;

    appendMessage('user', userMessage);
    messageInput.value = '';
    showTypingIndicator();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: userMessage }),
        });

        document.getElementById('typing-indicator').remove();

        if (!response.ok) {
            appendMessage('bot', 'متاسفانه خطایی در سرور رخ داد. لطفا دوباره تلاش کنید.');
            return;
        }

        const data = await response.json();
        appendMessage('bot', data.response);

    } catch (error) {
        document.getElementById('typing-indicator')?.remove();
        appendMessage('bot', 'خطا در برقراری ارتباط با سرور. آیا سرور در حال اجراست؟');
    }
});

window.addEventListener('load', () => {
    appendMessage('bot', 'سلام! من دستیار هوشمند شما برای محصولات الکترونیکی هستم. سوال خود را بپرسید.');
});