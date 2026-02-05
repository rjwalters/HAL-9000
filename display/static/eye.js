/**
 * HAL-9000 Eye Display Controller
 *
 * Connects to the WebSocket server and animates the eye
 * based on state and audio amplitude.
 */

class HALEye {
    constructor() {
        this.eyeGlow = document.getElementById('eye-glow');
        this.stateIndicator = document.getElementById('state-indicator');
        this.connectionStatus = document.getElementById('connection-status');

        this.ws = null;
        this.currentState = 'idle';
        this.currentAmplitude = 0;
        this.targetAmplitude = 0;

        // Animation frame ID for cleanup
        this.animationFrame = null;

        // Reconnection settings
        this.reconnectDelay = 1000;
        this.maxReconnectDelay = 30000;

        this.connect();
        this.startAnimationLoop();
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        console.log(`Connecting to ${wsUrl}...`);

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('Connected to HAL server');
            this.connectionStatus.classList.add('connected');
            this.reconnectDelay = 1000; // Reset delay on successful connection
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            } catch (e) {
                // Handle non-JSON messages (like pong)
                if (event.data !== 'pong') {
                    console.error('Failed to parse message:', e);
                }
            }
        };

        this.ws.onclose = () => {
            console.log('Disconnected from HAL server');
            this.connectionStatus.classList.remove('connected');
            this.scheduleReconnect();
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        // Send periodic pings to keep connection alive
        this.pingInterval = setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send('ping');
            }
        }, 25000);
    }

    scheduleReconnect() {
        console.log(`Reconnecting in ${this.reconnectDelay}ms...`);
        setTimeout(() => {
            this.connect();
        }, this.reconnectDelay);

        // Exponential backoff
        this.reconnectDelay = Math.min(
            this.reconnectDelay * 2,
            this.maxReconnectDelay
        );
    }

    handleMessage(data) {
        if (data.state !== undefined) {
            this.setState(data.state);
        }
        if (data.amplitude !== undefined) {
            this.targetAmplitude = data.amplitude;
        }
    }

    setState(state) {
        if (state === this.currentState) return;

        console.log(`State changed: ${this.currentState} -> ${state}`);
        this.currentState = state;

        // Update CSS classes
        this.eyeGlow.className = 'eye-glow';
        this.eyeGlow.classList.add(state);

        // Update state indicator
        this.stateIndicator.textContent = state.toUpperCase();
    }

    startAnimationLoop() {
        const animate = () => {
            // Smooth amplitude interpolation
            const smoothing = 0.15;
            this.currentAmplitude += (this.targetAmplitude - this.currentAmplitude) * smoothing;

            // Apply amplitude-based effects when speaking
            if (this.currentState === 'speaking') {
                this.applyAmplitudeEffect(this.currentAmplitude);
            }

            this.animationFrame = requestAnimationFrame(animate);
        };

        animate();
    }

    applyAmplitudeEffect(amplitude) {
        // Scale the glow based on amplitude (0 to 1)
        const baseScale = 1;
        const maxScale = 1.3;
        const scale = baseScale + (amplitude * (maxScale - baseScale));

        // Increase glow intensity based on amplitude
        const baseGlow = 20;
        const maxGlow = 100;
        const glowSize = baseGlow + (amplitude * (maxGlow - baseGlow));

        // Apply the effect
        this.eyeGlow.style.transform = `translate(-50%, -50%) scale(${scale})`;
        this.eyeGlow.style.boxShadow = `
            0 0 ${glowSize * 0.5}px #ff0000,
            0 0 ${glowSize}px #ff0000,
            0 0 ${glowSize * 1.5}px #cc0000,
            0 0 ${glowSize * 2}px #990000
        `;

        // Adjust brightness based on amplitude
        const brightness = 1 + (amplitude * 0.5);
        this.eyeGlow.style.filter = `brightness(${brightness})`;
    }

    destroy() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
        }
        if (this.ws) {
            this.ws.close();
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.halEye = new HALEye();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.halEye) {
        window.halEye.destroy();
    }
});
