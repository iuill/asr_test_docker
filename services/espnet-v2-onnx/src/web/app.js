/**
 * ReazonSpeech Real-time ASR Client (ESPnet ONNX)
 *
 * This script handles:
 * - Microphone access and audio capture
 * - WebSocket communication with the server
 * - Audio visualization
 * - Transcription display
 */

class ASRClient {
    constructor() {
        // DOM elements
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');
        this.transcriptionEl = document.getElementById('transcription');
        this.visualizerCanvas = document.getElementById('visualizer');

        // Audio context and nodes
        this.audioContext = null;
        this.mediaStream = null;
        this.processor = null;
        this.analyser = null;

        // WebSocket
        this.ws = null;
        this.isRecording = false;

        // Transcription state
        this.fullText = '';

        // Audio settings
        this.sampleRate = 16000;
        this.bufferSize = 4096;

        // Bind event handlers
        this.startBtn.addEventListener('click', () => this.start());
        this.stopBtn.addEventListener('click', () => this.stop());
        this.clearBtn.addEventListener('click', () => this.clear());

        // Initialize
        this.setupVisualizer();
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
     * Start recording and transcription
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

            // Connect to WebSocket
            await this.connectWebSocket();

            // Handle audio data
            this.processor.onaudioprocess = (e) => {
                if (!this.isRecording) return;

                const inputData = e.inputBuffer.getChannelData(0);
                this.sendAudio(inputData);
            };

            // Update UI
            this.isRecording = true;
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.updateStatus('録音中...', true);

            // Start visualization
            this.drawVisualizer();

        } catch (error) {
            console.error('Failed to start recording:', error);
            this.updateStatus('エラー: ' + error.message);
            this.cleanup();
        }
    }

    /**
     * Connect to WebSocket server
     */
    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/asr`;

            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                resolve();
            };

            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                if (this.isRecording) {
                    this.stop();
                }
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                reject(error);
            };

            this.ws.onmessage = (event) => {
                this.handleMessage(JSON.parse(event.data));
            };

            // Timeout
            setTimeout(() => {
                if (this.ws.readyState !== WebSocket.OPEN) {
                    reject(new Error('WebSocket connection timeout'));
                }
            }, 5000);
        });
    }

    /**
     * Handle incoming WebSocket message
     */
    handleMessage(data) {
        console.log('Received:', data);

        if (data.type === 'transcription') {
            if (data.is_final) {
                this.fullText += data.text;
                this.updateTranscription(this.fullText);
            } else {
                // Show partial result
                this.updateTranscription(this.fullText + `<span class="partial">${data.text}</span>`);
            }
        } else if (data.type === 'error') {
            console.error('Server error:', data.message);
        } else if (data.type === 'end') {
            console.log('Transcription ended');
        }
    }

    /**
     * Send audio data to server
     */
    sendAudio(audioData) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            return;
        }

        // Convert float32 to int16
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

        this.ws.send(message.buffer);
    }

    /**
     * Stop recording
     */
    stop() {
        this.isRecording = false;

        // Send end signal
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'end' }));
        }

        // Cleanup
        this.cleanup();

        // Update UI
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.updateStatus('停止', false, false);
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

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    /**
     * Clear transcription
     */
    clear() {
        this.fullText = '';
        this.transcriptionEl.innerHTML = '';
    }

    /**
     * Update transcription display
     */
    updateTranscription(html) {
        this.transcriptionEl.innerHTML = html;
        this.transcriptionEl.scrollTop = this.transcriptionEl.scrollHeight;
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

            // Gradient from purple to violet (ONNX theme)
            const hue = 270 + (dataArray[i] / 255) * 30;
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
