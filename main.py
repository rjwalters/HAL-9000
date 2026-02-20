#!/usr/bin/env python3
"""
HAL-9000 Voice Assistant
Main orchestrator that coordinates all components for a low-latency
voice assistant with HAL-9000 personality.
"""

import asyncio
import json
import logging
import os
import random
import re
import signal
import struct
import sys
import time
from contextlib import asynccontextmanager

import uvicorn

import config
from audio import AudioInput, AudioOutput, VoiceActivityDetector
from video import VideoCapture
from services import DeepgramSTT, GroqLLM, ElevenLabsTTS, ClaudeVision
from hal import ConversationManager
from hal.personality import get_system_prompt, get_greeting, get_error_response, get_proactive_prompt
from hal.transcript_logger import TranscriptLogger
from display.server import app, display_state, set_state, set_amplitude, broadcast_conversation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


_PERSON_WORDS = re.compile(
    r"\b(person|man|woman|people|someone|individual|figure|guest|he|she|they)\b",
    re.IGNORECASE,
)


def _description_has_person(description: str) -> bool:
    """Check if a vision description mentions a person."""
    return bool(_PERSON_WORDS.search(description))


def _parse_questions(response: str) -> list[str]:
    """Parse a list of questions from an LLM response.

    Tries JSON first, falls back to line-by-line parsing.
    """
    # Try JSON array
    try:
        parsed = json.loads(response)
        if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
            return [q.strip() for q in parsed if q.strip()]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: split on newlines and strip numbering
    questions = []
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering like "1.", "2)", "- ", "* "
        line = re.sub(r"^[\d]+[.)]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        line = line.strip().strip('"').strip("'")
        if line:
            questions.append(line)
    return questions


class HALOrchestrator:
    """
    Main orchestrator for HAL-9000 voice assistant.

    Manages the state machine and coordinates all components:
    - Audio input/output
    - Voice activity detection
    - Speech-to-text
    - LLM generation
    - Text-to-speech
    - Vision processing
    - Display updates
    """

    def __init__(self):
        # State
        self._state = config.State.IDLE
        self._running = False
        self._stop_event = asyncio.Event()

        # Audio components
        self.audio_input = AudioInput(device_index=config.AUDIO_INPUT_DEVICE)
        self.audio_output = AudioOutput(device_index=config.PLAYBACK_OUTPUT_DEVICE)
        self.vad = VoiceActivityDetector()

        # Video component (optional)
        self.video = VideoCapture()
        self._video_enabled = False

        # API services
        self.stt = DeepgramSTT()
        self.llm = GroqLLM()
        self.tts = ElevenLabsTTS()
        self.vision = ClaudeVision()

        # Conversation management
        self.conversation = ConversationManager()

        # Transcript logging
        self.transcript = TranscriptLogger()

        # Audio buffer for STT
        self._audio_buffer: list[bytes] = []
        self._buffer_lock = asyncio.Lock()

        # Speech event
        self._speech_end_event = asyncio.Event()

        # Event loop reference for thread-safe callbacks
        self._loop: asyncio.AbstractEventLoop | None = None

        # Timestamp when speaking ended (for echo cooldown)
        self._speaking_ended_at: float = 0.0

        # Debug STT audio counter
        self._stt_attempt: int = 0

        # Proactive conversation tracking
        self._person_frames: int = 0
        self._last_proactive_at: float = 0.0

    @property
    def state(self) -> str:
        return self._state

    async def set_state(self, new_state: str) -> None:
        """Update state and notify display."""
        if new_state != self._state:
            logger.info(f"State: {self._state} -> {new_state}")
            self._state = new_state
            await set_state(new_state)

    async def start(self) -> None:
        """Start all components and begin processing."""
        logger.info("Starting HAL-9000...")

        # Store event loop reference for thread callbacks
        self._loop = asyncio.get_running_loop()

        # Wire orchestrator to display for stats/conversation
        display_state.set_orchestrator(self)

        # Start hardware components
        self.audio_input.start()
        self.audio_output.start()
        self._video_enabled = self.video.start()
        if not self._video_enabled:
            logger.warning("Vision disabled - camera not available")

        # Register audio callback for VAD and buffering
        self.audio_input.add_callback(self._on_audio_chunk)

        # Set up VAD callbacks
        self.vad.on_speech_start(self._on_speech_start)
        self.vad.on_speech_end(self._on_speech_end)

        self._running = True
        await self.set_state(config.State.IDLE)

        logger.info("HAL-9000 started. Listening...")

        # Play greeting
        await self._speak(get_greeting())

        # Start main processing tasks
        await asyncio.gather(
            self._vision_loop(),
            self._amplitude_loop(),
            display_state.start_stats_loop(),
            self._wait_for_stop(),
        )

    async def stop(self) -> None:
        """Stop all components gracefully."""
        logger.info("Stopping HAL-9000...")
        self._running = False
        self._stop_event.set()

        # Stop hardware components
        self.audio_input.stop()
        self.audio_output.stop()
        self.video.stop()

        logger.info("HAL-9000 stopped")

    def _on_audio_chunk(self, audio_bytes: bytes) -> None:
        """Handle incoming audio chunks (runs in audio thread)."""
        if not self._running or not self._loop:
            return

        # Skip VAD while SPEAKING and during echo cooldown after speaking
        if self._state == config.State.SPEAKING:
            pass  # Never run VAD during speech output
        elif time.time() - self._speaking_ended_at < 0.6:
            pass  # Echo cooldown — suppress VAD for 600ms after speaking ends
        else:
            self.vad.process_audio(audio_bytes)

        # Buffer audio when listening
        if self._state in (config.State.IDLE, config.State.LISTENING):
            # Use thread-safe approach
            self._loop.call_soon_threadsafe(
                lambda b=audio_bytes: self._audio_buffer.append(b)
            )

    def _on_speech_start(self) -> None:
        """Called when VAD detects speech start."""
        if self._state == config.State.IDLE and self._loop:
            logger.info("Speech detected")
            self._loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self._handle_speech_start())
            )

    def _on_speech_end(self) -> None:
        """Called when VAD detects end of speech."""
        if self._state == config.State.LISTENING and self._loop:
            logger.info("End of speech detected")
            self._loop.call_soon_threadsafe(self._speech_end_event.set)

    async def _handle_speech_start(self) -> None:
        """Handle the beginning of user speech."""
        await self.set_state(config.State.LISTENING)
        self._speech_end_event.clear()
        self._audio_buffer.clear()

        # Start processing pipeline
        asyncio.create_task(self._process_speech())

    async def _process_speech(self) -> None:
        """Process user speech through the full pipeline."""
        try:
            # Wait for end of speech and transcribe
            transcript = await self._transcribe_speech()

            if not transcript or len(transcript.strip()) < 2:
                logger.info("No speech detected, returning to idle")
                await self.set_state(config.State.IDLE)
                return

            logger.info(f"Transcript: {transcript}")

            # Log user speech
            self.transcript.log_user_speech(transcript)

            # Update state and generate response
            await self.set_state(config.State.PROCESSING)

            # Add user message to conversation
            self.conversation.add_user_message(transcript)
            await broadcast_conversation()

            # Generate and speak response
            await self._generate_and_speak(transcript)

        except Exception as e:
            logger.error(f"Error processing speech: {e}", exc_info=True)
            await self._speak(get_error_response())

        finally:
            # Mark when speaking ended so echo cooldown kicks in
            if self._state == config.State.SPEAKING:
                self._speaking_ended_at = time.time()
            # Reset VAD and return to idle
            self.vad.reset()
            await self.set_state(config.State.IDLE)

    async def _transcribe_speech(self) -> str:
        """Transcribe buffered speech using Deepgram."""
        self._stt_attempt += 1
        attempt = self._stt_attempt
        debug_chunks: list[bytes] = []

        await self.stt.connect()
        self.stt.clear_transcript()

        try:
            # Send buffered audio (drain to avoid double-send on next loop)
            chunks_sent = 0
            while self._audio_buffer:
                chunk = self._audio_buffer.pop(0)
                await self.stt.send_audio(chunk)
                debug_chunks.append(chunk)
                chunks_sent += 1

            logger.info(f"STT attempt #{attempt}: sent {chunks_sent} buffered chunks")

            # Continue sending audio until Deepgram detects utterance end,
            # VAD detects silence, or we hit the max recording duration
            utterance_end = self.stt.utterance_end_event
            deadline = asyncio.get_event_loop().time() + config.VAD_MAX_RECORDING_SECS

            while not utterance_end.is_set() and not self._speech_end_event.is_set():
                if asyncio.get_event_loop().time() >= deadline:
                    logger.warning("Max recording duration reached, cutting off")
                    break

                await asyncio.sleep(0.01)

                # Send any new audio that arrived
                while self._audio_buffer:
                    chunk = self._audio_buffer.pop(0)
                    await self.stt.send_audio(chunk)
                    debug_chunks.append(chunk)
                    chunks_sent += 1

            if utterance_end.is_set():
                logger.info("Speech ended (Deepgram utterance end)")
            elif self._speech_end_event.is_set():
                logger.info("Speech ended (VAD silence)")

            # Flush any remaining buffered audio after speech end
            while self._audio_buffer:
                chunk = self._audio_buffer.pop(0)
                await self.stt.send_audio(chunk)
                debug_chunks.append(chunk)
                chunks_sent += 1

            logger.info(f"STT attempt #{attempt}: total {chunks_sent} chunks sent")

            # Save debug audio
            if config.DEBUG_SAVE_AUDIO and debug_chunks:
                self._save_stt_debug_audio(attempt, debug_chunks)

            # Signal Deepgram to finalize and flush any pending results
            await self.stt.finalize()

            # Wait for final transcript (up to 1.5s, or until we get one)
            wait_start = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - wait_start < 1.5:
                transcript = self.stt.get_final_transcript()
                if transcript:
                    break
                await asyncio.sleep(0.05)
            else:
                transcript = self.stt.get_final_transcript()

            logger.info(f"STT attempt #{attempt}: transcript='{transcript}'")
            return transcript

        finally:
            await self.stt.disconnect()

    def _save_stt_debug_audio(self, attempt: int, chunks: list[bytes]) -> None:
        """Save STT audio to WAV for debugging."""
        os.makedirs(config.DEBUG_AUDIO_DIR, exist_ok=True)
        pcm_data = b"".join(chunks)
        path = os.path.join(config.DEBUG_AUDIO_DIR, f"stt_{attempt:03d}.wav")

        num_frames = len(pcm_data) // (config.AUDIO_CHANNELS * config.AUDIO_FORMAT_WIDTH)
        byte_rate = config.AUDIO_SAMPLE_RATE * config.AUDIO_CHANNELS * config.AUDIO_FORMAT_WIDTH
        block_align = config.AUDIO_CHANNELS * config.AUDIO_FORMAT_WIDTH
        data_size = len(pcm_data)

        with open(path, "wb") as f:
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + data_size))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))
            f.write(struct.pack("<H", 1))  # PCM
            f.write(struct.pack("<H", config.AUDIO_CHANNELS))
            f.write(struct.pack("<I", config.AUDIO_SAMPLE_RATE))
            f.write(struct.pack("<I", byte_rate))
            f.write(struct.pack("<H", block_align))
            f.write(struct.pack("<H", 16))  # bits per sample
            f.write(b"data")
            f.write(struct.pack("<I", data_size))
            f.write(pcm_data)

        duration = num_frames / config.AUDIO_SAMPLE_RATE
        logger.info(f"STT debug audio saved: {path} ({duration:.1f}s, {len(chunks)} chunks)")

    async def _generate_and_speak(self, user_input: str) -> None:
        """Generate LLM response and synthesize speech with streaming."""
        await self.set_state(config.State.SPEAKING)

        # Get conversation history and system prompt
        messages = self.conversation.get_messages()
        system_prompt = get_system_prompt()

        # Append vision context to system prompt (keeps conversation
        # messages as clean user/assistant pairs)
        scene_description = self.vision.get_cached_description()
        if scene_description and scene_description != "No visual information available.":
            system_prompt += (
                f"\n\n[Current visual context — do not describe this aloud,"
                f" use it to inform your responses] {scene_description}"
            )

        # Stream response sentence by sentence
        full_response = ""

        async for sentence in self.llm.generate_sentences(messages, system_prompt):
            full_response += sentence + " "

            # Synthesize and play this sentence
            await self._synthesize_and_play(sentence)

        # Add assistant response to conversation
        self.conversation.add_assistant_message(full_response.strip())
        await broadcast_conversation()

        # Log HAL's response
        self.transcript.log_hal_response(full_response.strip())

    async def _synthesize_and_play(self, text: str) -> None:
        """Synthesize text to speech and play it."""
        if not text.strip():
            return

        try:
            async for audio_chunk in self.tts.synthesize_stream(text):
                self.audio_output.write(audio_chunk)

            # Wait for audio to finish playing
            while self.audio_output.is_playing():
                await asyncio.sleep(0.05)

        except Exception as e:
            logger.error(f"TTS error: {e}")

    async def _speak(self, text: str) -> None:
        """Speak a complete message."""
        await self.set_state(config.State.SPEAKING)
        await self._synthesize_and_play(text)
        self._speaking_ended_at = time.time()
        await self.set_state(config.State.IDLE)

    async def _initiate_proactive_conversation(self) -> None:
        """Generate and speak a proactive conversation starter."""
        if self._state != config.State.IDLE:
            return

        scene = self.vision.get_cached_description()
        prompt = get_proactive_prompt(scene)

        # Generate 10 questions, pick one at random
        response = await self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
        )
        questions = _parse_questions(response)
        if not questions:
            logger.warning("Proactive conversation: failed to parse questions from LLM response")
            return

        question = random.choice(questions)
        logger.info(f"Proactive conversation: \"{question}\"")

        # Add to conversation and speak
        self.conversation.add_assistant_message(question)
        await broadcast_conversation()
        self.transcript.log_hal_response(question)
        await self._speak(question)

    async def _vision_loop(self) -> None:
        """Background task to update vision cache."""
        if not self._video_enabled:
            logger.info("Vision loop disabled - no camera")
            return

        logger.info("Starting vision loop")

        while self._running:
            try:
                frame_base64 = await self.video.get_frame_base64_async()
                if frame_base64:
                    await self.vision.update_cache(frame_base64)

                # Check for person presence in vision description
                description = self.vision.get_cached_description()
                if description and _description_has_person(description):
                    self._person_frames += 1
                    logger.debug(f"Person detected, consecutive frames: {self._person_frames}")
                else:
                    self._person_frames = 0

                # Trigger proactive conversation after 2+ consecutive frames (~30s)
                if (
                    self._person_frames >= 2
                    and self._state == config.State.IDLE
                    and time.time() - self._last_proactive_at >= 120
                ):
                    logger.info("Person sustained presence detected — initiating proactive conversation")
                    self._person_frames = 0
                    self._last_proactive_at = time.time()
                    await self._initiate_proactive_conversation()

            except Exception as e:
                logger.error(f"Vision loop error: {e}")

            # Wait for next interval or stop
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=config.VISION_CAPTURE_INTERVAL,
                )
                break
            except asyncio.TimeoutError:
                continue

        logger.info("Vision loop stopped")

    async def _amplitude_loop(self) -> None:
        """Background task to update display amplitude."""
        while self._running:
            if self._state == config.State.SPEAKING:
                amplitude = self.audio_output.get_amplitude()
                await set_amplitude(amplitude)
            else:
                await set_amplitude(0.0)

            await asyncio.sleep(0.03)  # ~30fps update rate

    async def _wait_for_stop(self) -> None:
        """Wait for stop signal."""
        await self._stop_event.wait()


# Global orchestrator instance
orchestrator: HALOrchestrator | None = None


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan context manager."""
    global orchestrator

    # Start orchestrator
    orchestrator = HALOrchestrator()
    orchestrator_task = asyncio.create_task(orchestrator.start())

    yield

    # Stop orchestrator
    if orchestrator:
        await orchestrator.stop()
        orchestrator_task.cancel()
        try:
            await orchestrator_task
        except asyncio.CancelledError:
            pass


# Apply lifespan to FastAPI app
app.router.lifespan_context = lifespan


def signal_handler(sig, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {sig}, shutting down...")
    if orchestrator:
        asyncio.create_task(orchestrator.stop())
    sys.exit(0)


def main():
    """Main entry point."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Validate configuration
    missing_keys = []
    if not config.DEEPGRAM_API_KEY:
        missing_keys.append("DEEPGRAM_API_KEY")
    if not config.GROQ_API_KEY:
        missing_keys.append("GROQ_API_KEY")
    if not config.ELEVENLABS_API_KEY:
        missing_keys.append("ELEVENLABS_API_KEY")
    if not config.ELEVENLABS_VOICE_ID:
        missing_keys.append("ELEVENLABS_VOICE_ID")
    if not config.ANTHROPIC_API_KEY:
        missing_keys.append("ANTHROPIC_API_KEY")

    if missing_keys:
        logger.error(f"Missing required API keys: {', '.join(missing_keys)}")
        logger.error("Please set them in .env file (see .env.example)")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("HAL-9000 Voice Assistant")
    logger.info("=" * 60)
    logger.info(f"Display: http://localhost:{config.DISPLAY_PORT}")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    # Run the server
    uvicorn.run(
        app,
        host=config.DISPLAY_HOST,
        port=config.DISPLAY_PORT,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
