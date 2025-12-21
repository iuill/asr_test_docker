/**
 * Viewer Client for Real-time Transcription Sharing
 *
 * This script handles:
 * - Session authentication (viewer password)
 * - WebSocket connection for receiving transcriptions
 * - Real-time transcription display
 */

class ViewerClient {
    constructor() {
        // Session info
        this.sessionId = this.getSessionIdFromUrl();
        this.viewerToken = null;
        this.ws = null;

        // Transcription data
        this.transcriptions = new Map(); // model_id -> { segments: [], partialText: '', partialSpeaker: 0 }
        this.models = [];

        // Speaker colors for diarization
        this.speakerColors = [
            '#3b82f6', // blue
            '#ef4444', // red
            '#22c55e', // green
            '#f59e0b', // amber
            '#8b5cf6', // violet
            '#ec4899', // pink
            '#14b8a6', // teal
            '#f97316', // orange
        ];

        // DOM elements
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');
        this.resultsContainer = document.getElementById('resultsContainer');
        this.loginModal = document.getElementById('loginModal');
        this.loginForm = document.getElementById('loginForm');
        this.loginError = document.getElementById('loginError');
        this.loginBtn = document.getElementById('loginBtn');
        this.sessionEndedOverlay = document.getElementById('sessionEndedOverlay');

        // Bind event handlers
        this.loginForm.addEventListener('submit', (e) => this.handleLogin(e));
    }

    /**
     * Get session ID from URL path
     */
    getSessionIdFromUrl() {
        const path = window.location.pathname;
        const match = path.match(/\/view\/([^\/]+)/);
        return match ? match[1] : null;
    }

    /**
     * Initialize the viewer
     */
    async init() {
        if (!this.sessionId) {
            this.showError('無効なセッションURLです');
            return;
        }

        // Check if session exists
        try {
            const response = await fetch(`/api/sessions/${this.sessionId}`);
            if (!response.ok) {
                if (response.status === 404) {
                    this.showError('セッションが見つかりません');
                } else {
                    this.showError('セッションの確認に失敗しました');
                }
                return;
            }

            const sessionInfo = await response.json();
            this.models = sessionInfo.models;

            // Show login modal
            this.loginModal.classList.remove('hidden');
        } catch (error) {
            console.error('Init error:', error);
            this.showError('ネットワークエラーが発生しました');
        }
    }

    /**
     * Handle login form submission
     */
    async handleLogin(e) {
        e.preventDefault();

        const password = document.getElementById('password').value;
        this.loginBtn.disabled = true;
        this.loginError.textContent = '';

        try {
            const response = await fetch(`/api/sessions/${this.sessionId}/auth`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ password }),
            });

            if (response.ok) {
                const data = await response.json();
                this.viewerToken = data.token;
                this.models = data.models;

                // Hide login modal
                this.loginModal.classList.add('hidden');

                // Initialize transcription data for each model
                for (const modelId of this.models) {
                    this.transcriptions.set(modelId, {
                        segments: [],
                        partialText: '',
                        partialSpeaker: 0,
                    });
                }

                // Render results area
                this.renderResultsArea();

                // Connect to WebSocket
                this.connect();
            } else {
                const errorData = await response.json();
                this.loginError.textContent = errorData.detail || 'パスワードが正しくありません';
            }
        } catch (error) {
            console.error('Login error:', error);
            this.loginError.textContent = 'ネットワークエラーが発生しました';
        } finally {
            this.loginBtn.disabled = false;
        }
    }

    /**
     * Connect to WebSocket
     */
    connect() {
        this.updateStatus('接続中...', 'connecting');

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/view/${this.sessionId}?token=${encodeURIComponent(this.viewerToken)}`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.updateStatus('接続中', 'connected');
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason);
            if (event.code !== 1000) {
                this.updateStatus('切断されました', 'error');
                // Try to reconnect after 3 seconds
                setTimeout(() => this.connect(), 3000);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('エラー', 'error');
        };

        this.ws.onmessage = (event) => {
            this.handleMessage(event.data);
        };
    }

    /**
     * Handle incoming WebSocket message
     */
    handleMessage(data) {
        // Handle ping/pong
        if (data === 'ping') {
            this.ws.send('pong');
            return;
        }
        if (data === 'pong') {
            return;
        }

        try {
            const message = JSON.parse(data);
            console.log('Received message:', message);

            switch (message.type) {
                case 'init':
                    this.handleInit(message);
                    break;
                case 'transcription':
                    this.handleTranscription(message);
                    break;
                case 'session_end':
                    this.handleSessionEnd(message);
                    break;
                default:
                    console.log('Unknown message type:', message.type);
            }
        } catch (error) {
            console.error('Error parsing message:', error);
        }
    }

    /**
     * Handle init message (initial data)
     */
    handleInit(message) {
        this.models = message.models;

        // Load existing transcriptions
        for (const [modelId, segments] of Object.entries(message.transcriptions)) {
            const conn = this.transcriptions.get(modelId) || {
                segments: [],
                partialText: '',
                partialSpeaker: 0,
            };

            conn.segments = segments.map(seg => ({
                text: seg.text,
                speakerTag: seg.speaker_tag,
            }));

            this.transcriptions.set(modelId, conn);
        }

        this.renderResultsArea();
        this.updateAllTranscriptions();
    }

    /**
     * Handle transcription message
     */
    handleTranscription(message) {
        const modelId = message.model_id;
        let conn = this.transcriptions.get(modelId);

        // Initialize if not exists
        if (!conn) {
            conn = {
                segments: [],
                partialText: '',
                partialSpeaker: 0,
            };
            this.transcriptions.set(modelId, conn);
        }

        // Add model to list if not exists and re-render
        if (!this.models.includes(modelId)) {
            this.models.push(modelId);
            this.renderResultsArea();
            this.updateAllTranscriptions();
        }

        const speakerTag = message.speaker_tag || 0;
        const text = message.text || '';

        if (message.is_final) {
            conn.segments.push({
                text: text,
                speakerTag: speakerTag,
            });
            conn.partialText = '';
            conn.partialSpeaker = 0;
        } else {
            conn.partialText = text;
            conn.partialSpeaker = speakerTag;
        }

        this.updateTranscriptionForModel(modelId);
    }

    /**
     * Handle session end message
     */
    handleSessionEnd(message) {
        console.log('Session ended:', message);
        this.updateStatus('セッション終了', 'error');
        this.sessionEndedOverlay.classList.add('active');

        if (this.ws) {
            this.ws.close(1000);
            this.ws = null;
        }
    }

    /**
     * Update status display
     */
    updateStatus(text, state = '') {
        this.statusText.textContent = text;
        this.statusDot.className = 'status-dot';
        if (state) {
            this.statusDot.classList.add(state);
        }
    }

    /**
     * Show error message
     */
    showError(message) {
        this.resultsContainer.innerHTML = `
            <div class="no-content" style="color: #dc2626;">
                ${message}
            </div>
        `;
        this.loginModal.classList.add('hidden');
    }

    /**
     * Render results area
     */
    renderResultsArea() {
        if (this.models.length === 0) {
            this.resultsContainer.innerHTML = `
                <div class="no-content">
                    モデル情報を取得中...
                </div>
            `;
            return;
        }

        this.resultsContainer.innerHTML = `
            <div class="results-grid">
                ${this.models.map(modelId => `
                    <div class="result-card" data-model="${modelId}">
                        <div class="result-header">
                            <span class="result-model-name">${modelId}</span>
                        </div>
                        <div class="transcription-area">
                            <div class="transcription-text" id="transcription-${modelId}"></div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    /**
     * Update all transcription displays
     */
    updateAllTranscriptions() {
        for (const modelId of this.models) {
            this.updateTranscriptionForModel(modelId);
        }
    }

    /**
     * Update transcription display for a specific model
     */
    updateTranscriptionForModel(modelId) {
        const conn = this.transcriptions.get(modelId);
        if (!conn) return;

        const el = document.getElementById(`transcription-${modelId}`);
        if (!el) return;

        el.innerHTML = this.buildTranscriptionHtml(conn);

        // Auto-scroll to bottom
        const container = el.closest('.transcription-area');
        if (container) {
            container.scrollTop = container.scrollHeight;
        }
    }

    /**
     * Build HTML for transcription with speaker colors
     */
    buildTranscriptionHtml(conn) {
        let html = '';
        let lastSpeaker = 0;

        for (const segment of conn.segments) {
            if (segment.speakerTag > 0 && segment.speakerTag !== lastSpeaker) {
                const color = this.getSpeakerColor(segment.speakerTag);
                html += `<span class="speaker-label" style="color: ${color};">[話者${segment.speakerTag}]</span> `;
                lastSpeaker = segment.speakerTag;
            }

            const displayText = segment.text.replace(/\n/g, '<br>');
            if (segment.speakerTag > 0) {
                const color = this.getSpeakerColor(segment.speakerTag);
                html += `<span class="speaker-text" style="color: ${color};">${displayText}</span>`;
            } else {
                html += displayText;
            }
        }

        if (conn.partialText) {
            if (conn.partialSpeaker > 0 && conn.partialSpeaker !== lastSpeaker) {
                const color = this.getSpeakerColor(conn.partialSpeaker);
                html += `<span class="speaker-label" style="color: ${color};">[話者${conn.partialSpeaker}]</span> `;
            }
            html += `<span class="partial">${conn.partialText}</span>`;
        }

        return html || '<span style="color: #6b7280;">テキストを待機中...</span>';
    }

    /**
     * Get color for a speaker tag
     */
    getSpeakerColor(speakerTag) {
        if (speakerTag <= 0) return 'inherit';
        return this.speakerColors[(speakerTag - 1) % this.speakerColors.length];
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.viewerClient = new ViewerClient();
    window.viewerClient.init();
});
