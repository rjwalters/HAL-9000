/**
 * HAL-9000 Conversation Pane
 *
 * Collapsible monitoring panel showing status, stats,
 * and scrolling conversation history.
 */

class ConversationPane {
    constructor() {
        this.pane = document.getElementById('conversation-pane');
        this.toggle = document.getElementById('pane-toggle');
        this.closeBtn = document.getElementById('pane-close');
        this.messagesEl = document.getElementById('pane-messages');

        // Stats elements
        this.stateEl = document.getElementById('pane-state');
        this.msgCountEl = document.getElementById('pane-msg-count');
        this.visionAgeEl = document.getElementById('pane-vision-age');
        this.uptimeEl = document.getElementById('pane-uptime');
        this.clientsEl = document.getElementById('pane-clients');

        this.isOpen = false;
        this._autoScroll = true;

        // Bind events
        this.toggle.addEventListener('click', () => this.togglePane());
        this.closeBtn.addEventListener('click', () => this.collapse());

        // Backtick keyboard shortcut
        document.addEventListener('keydown', (e) => {
            if (e.key === '`' && !e.ctrlKey && !e.metaKey && !e.altKey) {
                const tag = e.target.tagName;
                if (tag === 'INPUT' || tag === 'TEXTAREA') return;
                e.preventDefault();
                this.togglePane();
            }
        });

        // Track scroll position for auto-scroll
        this.messagesEl.addEventListener('scroll', () => {
            const el = this.messagesEl;
            this._autoScroll = (el.scrollTop + el.clientHeight >= el.scrollHeight - 20);
        });
    }

    togglePane() {
        if (this.isOpen) {
            this.collapse();
        } else {
            this.expand();
        }
    }

    expand() {
        this.isOpen = true;
        this.pane.classList.add('open');
        this.toggle.style.display = 'none';
    }

    collapse() {
        this.isOpen = false;
        this.pane.classList.remove('open');
        this.toggle.style.display = '';
    }

    setConnected(connected) {
        // Connection status is already shown by the eye's indicator dot
    }

    handleMessage(data) {
        if (data.type === 'stats') {
            this.updateStats(data);
        } else if (data.type === 'conversation') {
            this.renderConversation(data.messages);
        } else if (data.state !== undefined) {
            // State-only message (eye state/amplitude)
            this.updateState(data.state);
        }
    }

    updateState(state) {
        this.stateEl.textContent = state;
        this.stateEl.className = 'pane-stat-value pane-stat-state ' + state;
    }

    updateStats(data) {
        this.updateState(data.state);
        this.msgCountEl.textContent = data.message_count;
        this.clientsEl.textContent = data.connected_clients;

        if (data.vision_cache_age === null) {
            this.visionAgeEl.textContent = '--';
        } else {
            this.visionAgeEl.textContent = data.vision_cache_age.toFixed(0) + 's';
        }

        this.uptimeEl.textContent = this.formatUptime(data.uptime);
    }

    formatUptime(seconds) {
        if (seconds < 60) return Math.floor(seconds) + 's';
        if (seconds < 3600) return Math.floor(seconds / 60) + 'm ' + Math.floor(seconds % 60) + 's';
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        return h + 'h ' + m + 'm';
    }

    renderConversation(messages) {
        if (!messages || messages.length === 0) {
            this.messagesEl.innerHTML = '<div class="pane-empty">No messages yet</div>';
            return;
        }

        this.msgCountEl.textContent = messages.length;

        let html = '';
        for (const msg of messages) {
            let roleLabel;
            let extraClass = '';
            if (msg.role === 'vision') {
                roleLabel = 'EYE';
                extraClass = ' vision-msg';
            } else {
                roleLabel = msg.role === 'user' ? 'YOU' : 'HAL';
            }
            html += '<div class="pane-message' + extraClass + '">'
                + '<span class="pane-message-ts">' + this.escapeHtml(msg.timestamp) + '</span>'
                + '<span class="pane-message-role ' + msg.role + '">' + roleLabel + '</span>'
                + '<span class="pane-message-content">' + this.escapeHtml(msg.content) + '</span>'
                + '</div>';
        }

        this.messagesEl.innerHTML = html;

        // Auto-scroll to bottom
        if (this._autoScroll) {
            this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
        }
    }

    escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
}
