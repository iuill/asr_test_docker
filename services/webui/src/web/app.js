/**
 * Real-time ASR Client with Multi-Model Support
 *
 * This script handles:
 * - Multiple model selection (checkboxes)
 * - Microphone access and audio capture
 * - Multiple WebSocket connections (one per selected model)
 * - Audio visualization
 * - Separate transcription display per model
 */

class ASRClient {
    constructor() {
        // DOM elements
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');
        this.visualizerCanvas = document.getElementById('visualizer');
        this.modelSelector = document.getElementById('modelSelector');
        this.resultsContainer = document.getElementById('resultsContainer');

        // Audio context and nodes
        this.audioContext = null;
        this.mediaStream = null;
        this.processor = null;
        this.analyser = null;

        // Multi-model state
        this.models = [];
        this.selectedModels = new Set();  // Set of selected model IDs
        this.connections = new Map();     // Map of modelId -> { ws, segments, status }

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

        // Recording state
        this.isRecording = false;

        // Audio settings
        this.sampleRate = 16000;
        this.bufferSize = 4096;

        // Bind event handlers
        this.startBtn.addEventListener('click', () => this.start());
        this.stopBtn.addEventListener('click', () => this.stop());
        this.clearBtn.addEventListener('click', () => this.clear());

        // Initialize
        this.setupVisualizer();
        this.loadModels();

        // Refresh model status periodically
        setInterval(() => this.refreshModelStatus(), 30000);
    }

    /**
     * Load available models from the server
     */
    async loadModels() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();

            this.models = data.models;

            // Select default model initially
            const defaultModel = this.models.find(m => m.id === data.default);
            if (defaultModel && defaultModel.status === 'healthy' && defaultModel.model_loaded) {
                this.selectedModels.add(data.default);
            }

            this.renderModelSelector();
            this.renderResultsArea();
            this.updateStartButton();
        } catch (error) {
            console.error('Failed to load models:', error);
            this.modelSelector.innerHTML = `
                <div class="model-card" style="color: #dc2626;">
                    モデル情報の読み込みに失敗しました。ページを再読み込みしてください。
                </div>
            `;
        }
    }

    /**
     * Refresh model status without full reload
     */
    async refreshModelStatus() {
        if (this.isRecording) return; // Don't refresh while recording

        try {
            const response = await fetch('/api/models');
            const data = await response.json();

            this.models = data.models;

            // Remove any selected models that are now offline
            for (const modelId of this.selectedModels) {
                const model = this.models.find(m => m.id === modelId);
                if (!model || model.status !== 'healthy' || !model.model_loaded) {
                    this.selectedModels.delete(modelId);
                }
            }

            this.renderModelSelector();
            this.renderResultsArea();
            this.updateStartButton();
        } catch (error) {
            console.error('Failed to refresh model status:', error);
        }
    }

    /**
     * Render the model selector UI with checkboxes
     */
    renderModelSelector() {
        this.modelSelector.innerHTML = this.models.map(model => {
            const isSelected = this.selectedModels.has(model.id);
            const isHealthy = model.status === 'healthy' && model.model_loaded;
            const statusClass = isHealthy ? 'healthy' : (model.status === 'offline' ? 'offline' : 'loading');
            const statusText = isHealthy ? '利用可能' : (model.status === 'offline' ? 'オフライン' : '準備中');

            return `
                <div class="model-card ${isSelected ? 'selected' : ''} ${!isHealthy ? 'disabled' : ''}"
                     data-model-id="${model.id}"
                     onclick="window.asrClient.toggleModel('${model.id}')">
                    <input type="checkbox"
                           class="model-checkbox"
                           ${isSelected ? 'checked' : ''}
                           ${!isHealthy ? 'disabled' : ''}
                           onclick="event.stopPropagation(); window.asrClient.toggleModel('${model.id}')">
                    <div class="model-info">
                        <div class="model-name">${model.name}</div>
                        <div class="model-description">${model.description}</div>
                        <div class="model-status">
                            <span class="status-indicator ${statusClass}"></span>
                            <span>${statusText}</span>
                            <span class="speed-badge ${model.speed === 'fast' ? 'fast' : ''}">${model.speed === 'fast' ? '高速' : '標準'}</span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }

    /**
     * Toggle model selection
     */
    toggleModel(modelId) {
        const model = this.models.find(m => m.id === modelId);
        if (!model) return;

        // Check if model is available
        const isHealthy = model.status === 'healthy' && model.model_loaded;
        if (!isHealthy) {
            return; // Don't allow selecting unavailable models
        }

        if (this.selectedModels.has(modelId)) {
            this.selectedModels.delete(modelId);
        } else {
            this.selectedModels.add(modelId);
        }

        this.renderModelSelector();
        this.renderResultsArea();
        this.updateStartButton();

        // If recording, update connections
        if (this.isRecording) {
            this.updateConnections();
        }
    }

    /**
     * Render the results area based on selected models
     */
    renderResultsArea() {
        if (this.selectedModels.size === 0) {
            this.resultsContainer.innerHTML = `
                <div class="no-model-selected">
                    モデルを選択してください
                </div>
            `;
            return;
        }

        const selectedModelsList = Array.from(this.selectedModels);
        this.resultsContainer.innerHTML = `
            <div class="results-grid">
                ${selectedModelsList.map(modelId => {
                    const model = this.models.find(m => m.id === modelId);
                    const connection = this.connections.get(modelId);
                    const statusText = connection ?
                        (connection.status === 'connected' ? '接続中' :
                         connection.status === 'error' ? 'エラー' : '待機中')
                        : '待機中';
                    const statusClass = connection?.status || '';

                    // Build initial transcription HTML
                    let transcriptionHtml = '';
                    if (connection) {
                        transcriptionHtml = this.buildTranscriptionHtml(connection);
                    }

                    return `
                        <div class="result-card" data-model="${modelId}">
                            <div class="result-header">
                                <span class="result-model-name">${model?.name || modelId}</span>
                                <span class="result-status ${statusClass}">${statusText}</span>
                            </div>
                            <div class="transcription-area">
                                <div class="transcription-text" id="transcription-${modelId}">
                                    ${transcriptionHtml}
                                </div>
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
            <p class="info">
                マイクに向かって話すと、各モデルでリアルタイムに文字起こしされます。
            </p>
        `;
    }

    /**
     * Update start button state based on model selection
     */
    updateStartButton() {
        const hasSelectedModels = this.selectedModels.size > 0;
        this.startBtn.disabled = !hasSelectedModels || this.isRecording;
    }

    /**
     * Set up the audio visualizer
     */
    setupVisualizer() {
        this.canvasCtx = this.visualizerCanvas.getContext('2d');
        this.visualizerCanvas.width = this.visualizerCanvas.offsetWidth * 2;
        this.visualizerCanvas.height = this.visualizerCanvas.offsetHeight * 2;
    }

    /**
     * Update status display
     */
    updateStatus(status, isRecording = false, isConnected = false) {
        this.statusText.textContent = status;
        this.statusDot.classList.remove('connected', 'recording');

        if (isRecording) {
            this.statusDot.classList.add('recording');
        } else if (isConnected) {
            this.statusDot.classList.add('connected');
        }
    }

    /**
     * Start recording and transcription for all selected models
     */
    async start() {
        try {
            // Request microphone access
            this.updateStatus('マイクにアクセス中...');
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: this.sampleRate,
                    echoCancellation: true,
                    noiseSuppression: true,
                }
            });

            // Set up audio context
            this.audioContext = new AudioContext({ sampleRate: this.sampleRate });
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);

            // Set up analyser for visualization
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            source.connect(this.analyser);

            // Set up script processor for audio capture
            this.processor = this.audioContext.createScriptProcessor(this.bufferSize, 1, 1);
            source.connect(this.processor);
            this.processor.connect(this.audioContext.destination);

            // Connect to all selected models
            await this.connectAllModels();

            // Handle audio data
            this.processor.onaudioprocess = (e) => {
                if (!this.isRecording) return;

                const inputData = e.inputBuffer.getChannelData(0);
                this.sendAudioToAll(inputData);
            };

            // Update UI
            this.isRecording = true;
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.updateStatus(`録音中... (${this.selectedModels.size}モデル)`, true);

            // Start visualization
            this.drawVisualizer();

        } catch (error) {
            console.error('Failed to start recording:', error);
            this.updateStatus('エラー: ' + error.message);
            this.cleanup();
        }
    }

    /**
     * Connect to all selected models
     */
    async connectAllModels() {
        const connectPromises = Array.from(this.selectedModels).map(modelId =>
            this.connectToModel(modelId)
        );

        await Promise.allSettled(connectPromises);
        this.renderResultsArea();
    }

    /**
     * Connect to a specific model
     */
    async connectToModel(modelId) {
        return new Promise((resolve, reject) => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/asr?model=${modelId}`;

            const ws = new WebSocket(wsUrl);

            // Initialize connection state
            this.connections.set(modelId, {
                ws,
                segments: [],      // Array of { text, speakerTag }
                partialText: '',
                partialSpeaker: 0,
                status: 'connecting'
            });

            ws.onopen = () => {
                console.log(`WebSocket connected to ${modelId}`);
                const conn = this.connections.get(modelId);
                if (conn) {
                    conn.status = 'connected';
                    this.renderResultsArea();
                }
                resolve();
            };

            ws.onclose = () => {
                console.log(`WebSocket disconnected from ${modelId}`);
                const conn = this.connections.get(modelId);
                if (conn) {
                    conn.status = 'disconnected';
                    this.renderResultsArea();
                }
            };

            ws.onerror = (error) => {
                console.error(`WebSocket error for ${modelId}:`, error);
                const conn = this.connections.get(modelId);
                if (conn) {
                    conn.status = 'error';
                    this.renderResultsArea();
                }
                reject(error);
            };

            ws.onmessage = (event) => {
                this.handleMessage(modelId, JSON.parse(event.data));
            };

            // Timeout
            setTimeout(() => {
                if (ws.readyState !== WebSocket.OPEN) {
                    const conn = this.connections.get(modelId);
                    if (conn) {
                        conn.status = 'error';
                        this.renderResultsArea();
                    }
                    reject(new Error(`WebSocket connection timeout for ${modelId}`));
                }
            }, 5000);
        });
    }

    /**
     * Update connections when model selection changes during recording
     */
    async updateConnections() {
        // Close connections for deselected models
        for (const [modelId, conn] of this.connections) {
            if (!this.selectedModels.has(modelId)) {
                if (conn.ws && conn.ws.readyState === WebSocket.OPEN) {
                    conn.ws.send(JSON.stringify({ type: 'end' }));
                    conn.ws.close();
                }
                this.connections.delete(modelId);
            }
        }

        // Connect to newly selected models
        for (const modelId of this.selectedModels) {
            if (!this.connections.has(modelId)) {
                try {
                    await this.connectToModel(modelId);
                } catch (error) {
                    console.error(`Failed to connect to ${modelId}:`, error);
                }
            }
        }

        this.renderResultsArea();
    }

    /**
     * Handle incoming WebSocket message for a specific model
     */
    handleMessage(modelId, data) {
        const conn = this.connections.get(modelId);
        if (!conn) return;

        console.log(`Received from ${modelId}:`, data);

        if (data.type === 'transcription') {
            const speakerTag = data.speaker_tag || 0;

            if (data.is_final) {
                // Add to segments with speaker info
                conn.segments.push({
                    text: data.text,
                    speakerTag: speakerTag
                });
                conn.partialText = '';
                conn.partialSpeaker = 0;
            } else {
                // Filter interim results by result_index (Google STT only)
                if (modelId === 'google-stt') {
                    const resultIndex = data.provider_info?.result_index ?? 0;
                    // Only show result_index=0 (first/current utterance)
                    // result_index=0 always has high stability (~0.9)
                    if (resultIndex === 0) {
                        conn.partialText = data.text;
                        conn.partialSpeaker = speakerTag;
                    }
                    // result_index > 0: unstable next utterance, don't update
                } else {
                    conn.partialText = data.text;
                    conn.partialSpeaker = speakerTag;
                }
            }
            this.updateTranscriptionForModel(modelId);
        } else if (data.type === 'error') {
            console.error(`Server error for ${modelId}:`, data.message);
            conn.status = 'error';
            this.renderResultsArea();
        } else if (data.type === 'end') {
            console.log(`Transcription ended for ${modelId}`);
        }
    }

    /**
     * Get color for a speaker tag
     */
    getSpeakerColor(speakerTag) {
        if (speakerTag <= 0) return 'inherit';
        return this.speakerColors[(speakerTag - 1) % this.speakerColors.length];
    }

    /**
     * Get speaker label
     */
    getSpeakerLabel(speakerTag) {
        if (speakerTag <= 0) return '';
        return `話者${speakerTag}`;
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
                html += `<span class="speaker-label" style="color: ${color};">[${this.getSpeakerLabel(segment.speakerTag)}]</span> `;
                lastSpeaker = segment.speakerTag;
            }

            if (segment.speakerTag > 0) {
                const color = this.getSpeakerColor(segment.speakerTag);
                html += `<span class="speaker-text" style="color: ${color};">${segment.text}</span>`;
            } else {
                html += segment.text;
            }
        }

        if (conn.partialText) {
            if (conn.partialSpeaker > 0 && conn.partialSpeaker !== lastSpeaker) {
                const color = this.getSpeakerColor(conn.partialSpeaker);
                html += `<span class="speaker-label" style="color: ${color};">[${this.getSpeakerLabel(conn.partialSpeaker)}]</span> `;
            }
            html += `<span class="partial">${conn.partialText}</span>`;
        }

        return html;
    }

    /**
     * Update transcription display for a specific model
     */
    updateTranscriptionForModel(modelId) {
        const conn = this.connections.get(modelId);
        if (!conn) return;

        const el = document.getElementById(`transcription-${modelId}`);
        if (el) {
            el.innerHTML = this.buildTranscriptionHtml(conn);
            el.scrollTop = el.scrollHeight;
        }
    }

    /**
     * Send audio data to all connected models
     */
    sendAudioToAll(audioData) {
        // Convert float32 to int16 once
        const int16Data = new Int16Array(audioData.length);
        for (let i = 0; i < audioData.length; i++) {
            const s = Math.max(-1, Math.min(1, audioData[i]));
            int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        // Create message with sample rate header
        const header = new ArrayBuffer(4);
        new DataView(header).setUint32(0, this.sampleRate, true);

        const message = new Uint8Array(4 + int16Data.byteLength);
        message.set(new Uint8Array(header), 0);
        message.set(new Uint8Array(int16Data.buffer), 4);

        // Send to all connected models
        for (const [modelId, conn] of this.connections) {
            if (conn.ws && conn.ws.readyState === WebSocket.OPEN) {
                conn.ws.send(message.buffer);
            }
        }
    }

    /**
     * Stop recording
     */
    stop() {
        this.isRecording = false;

        // Send end signal to all connections
        for (const [modelId, conn] of this.connections) {
            if (conn.ws && conn.ws.readyState === WebSocket.OPEN) {
                conn.ws.send(JSON.stringify({ type: 'end' }));
            }
        }

        // Cleanup
        this.cleanup();

        // Update UI
        this.updateStartButton();
        this.stopBtn.disabled = true;
        this.updateStatus('停止', false, false);
        this.renderResultsArea();
    }

    /**
     * Clean up audio resources
     */
    cleanup() {
        if (this.processor) {
            this.processor.disconnect();
            this.processor = null;
        }

        if (this.analyser) {
            this.analyser.disconnect();
            this.analyser = null;
        }

        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        // Close all WebSocket connections
        for (const [modelId, conn] of this.connections) {
            if (conn.ws) {
                conn.ws.close();
            }
        }
        this.connections.clear();
    }

    /**
     * Clear all transcriptions
     */
    clear() {
        for (const [modelId, conn] of this.connections) {
            conn.segments = [];
            conn.partialText = '';
            conn.partialSpeaker = 0;
        }

        // Also clear for non-connected but selected models
        for (const modelId of this.selectedModels) {
            const el = document.getElementById(`transcription-${modelId}`);
            if (el) {
                el.innerHTML = '';
            }
        }

        this.renderResultsArea();
    }

    /**
     * Draw audio visualizer
     */
    drawVisualizer() {
        if (!this.analyser || !this.isRecording) {
            // Clear canvas when not recording
            this.canvasCtx.fillStyle = '#f9fafb';
            this.canvasCtx.fillRect(0, 0, this.visualizerCanvas.width, this.visualizerCanvas.height);
            return;
        }

        requestAnimationFrame(() => this.drawVisualizer());

        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        this.analyser.getByteFrequencyData(dataArray);

        const width = this.visualizerCanvas.width;
        const height = this.visualizerCanvas.height;

        this.canvasCtx.fillStyle = '#f9fafb';
        this.canvasCtx.fillRect(0, 0, width, height);

        const barWidth = (width / bufferLength) * 2.5;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const barHeight = (dataArray[i] / 255) * height;

            // Gradient from blue to purple
            const hue = 220 + (dataArray[i] / 255) * 40;
            this.canvasCtx.fillStyle = `hsl(${hue}, 70%, 50%)`;

            this.canvasCtx.fillRect(
                x,
                height - barHeight,
                barWidth,
                barHeight
            );

            x += barWidth + 1;
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.asrClient = new ASRClient();
});
