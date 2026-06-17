// static/script.js
document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    const clearBtn = document.getElementById('clear-btn');
    const newChatBtn = document.getElementById('new-chat-btn');
    const exportBtn = document.getElementById('export-chat-btn');
    const chips = document.querySelectorAll('.chip');
    
    let chatHistory = []; // To store for export

    // Handle Quick Action Chips
    chips.forEach(chip => {
        chip.addEventListener('click', () => {
            userInput.value = chip.textContent;
            chatForm.dispatchEvent(new Event('submit'));
        });
    });

    // Handle Clear Chat
    const clearChat = () => {
        // Keep only the initial greeting
        const initialGreeting = chatBox.firstElementChild.outerHTML;
        chatBox.innerHTML = initialGreeting;
        chatHistory = [];
    };
    clearBtn.addEventListener('click', clearChat);
    newChatBtn.addEventListener('click', clearChat);

    // Handle Export Chat
    exportBtn.addEventListener('click', () => {
        if (chatHistory.length === 0) {
            alert("No chat history to export yet!");
            return;
        }
        
        let textData = "BrightTech Solutions SLM - Chat Export\n";
        textData += "========================================\n\n";
        
        chatHistory.forEach(msg => {
            textData += `${msg.role.toUpperCase()}:\n${msg.content}\n\n`;
        });
        
        const blob = new Blob([textData], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `slm_chat_export_${new Date().getTime()}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    });

    function scrollToBottom() {
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Add User Message
    function addUserMessage(text) {
        chatHistory.push({ role: 'user', content: text });
        
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message user-message';
        msgDiv.innerHTML = `
            <div class="message-body">
                <div class="message-content">${text}</div>
            </div>
            <div class="avatar user-avatar">U</div>
        `;
        chatBox.appendChild(msgDiv);
        scrollToBottom();
    }

    // Add Bot Message container
    function createBotMessageContainer() {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message bot-message';
        
        msgDiv.innerHTML = `
            <div class="avatar bot-avatar">AI</div>
            <div class="message-body">
                <div class="message-content"></div>
            </div>
        `;
        chatBox.appendChild(msgDiv);
        scrollToBottom();
        return msgDiv;
    }

    // Stream text token by token
    async function streamText(container, text, source, genTime) {
        chatHistory.push({ role: 'bot', content: text });
        const contentDiv = container.querySelector('.message-content');
        
        // Tokenize text roughly by words to simulate LLM streaming
        const tokens = text.match(/\S+|\s+/g) || [text];
        
        for (let i = 0; i < tokens.length; i++) {
            contentDiv.textContent += tokens[i];
            scrollToBottom();
            // Wait 20-40ms between tokens for realistic streaming
            await new Promise(r => setTimeout(r, Math.random() * 20 + 20));
        }
        
        // Add metadata after streaming finishes
        const bodyDiv = container.querySelector('.message-body');
        const metaBar = document.createElement('div');
        metaBar.className = 'metadata-bar';
        
        let sourceClass = source === 'Database' ? 'db' : 'kb';
        metaBar.innerHTML = `
            <span class="meta-tag">⚡ ${source}</span>
            <span class="meta-tag">⏱️ ${genTime}s</span>
        `;
        bodyDiv.appendChild(metaBar);
        scrollToBottom();
    }

    // Handle form submission
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const message = userInput.value.trim();
        if (!message) return;

        // 1. Show user message
        addUserMessage(message);
        userInput.value = '';
        
        // 2. Create empty bot container
        const botContainer = createBotMessageContainer();
        const contentDiv = botContainer.querySelector('.message-content');
        
        // Blinking cursor simulation
        contentDiv.innerHTML = '<span style="animation: blink 1s step-end infinite;">█</span>';

        try {
            // 3. Send request to Flask backend
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message })
            });

            if (!response.ok) throw new Error('Network error');

            const data = await response.json();
            
            // 4. Remove cursor and stream response
            contentDiv.innerHTML = '';
            await streamText(botContainer, data.answer, data.source, data.time || '0.12');

        } catch (error) {
            console.error('Error:', error);
            contentDiv.innerHTML = '';
            contentDiv.textContent = 'Sorry, I encountered a network error. Ensure the server is running.';
        }
    });

    userInput.focus();
});
