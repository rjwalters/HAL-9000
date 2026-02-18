#!/usr/bin/env python3
"""
HAL-9000 Voice Assistant
Main orchestrator that coordinates all components for a low-latency
voice assistant with HAL-9000 personality.
"""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager

import uvicorn

import config
from audio import AudioInput, AudioOutput, VoiceActivityDetector
from video import VideoCapture
from services import DeepgramSTT, GroqLLM, ElevenLabsTTS, ClaudeVision
from hal import ConversationManager
from hal.personality import get_system_prompt, get_greeting, get_error_response
from hal.transcript_logger import TranscriptLogger
from display.server import app, display_state, set_state, set_amplitude, broadcast_conversation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
        self.audio_input = AudioInput()
        self.audio_output = AudioOutput()
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

        # Barge-in event — set when user speaks during HAL's response
        self._barge_in_event = asyncio.Event()

        # Event loop reference for thread-safe callbacks
        self._loop: asyncio.AbstractEventLoop | None = None

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

        # Process through VAD
        self.vad.process_audio(audio_bytes)

        # Buffer audio when listening, or during SPEAKING after barge-in
        if self._state in (config.State.IDLE, config.State.LISTENING) or (
            self._state == config.State.SPEAKING and self._barge_in_event.is_set()
        ):
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
        elif self._state == config.State.SPEAKING and self._loop:
            logger.info("Barge-in detected during speech")
            self._loop.call_soon_threadsafe(self._barge_in_event.set)
            self._loop.call_soon_threadsafe(self._audio_buffer.clear)

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
        """Process user speech through the full pipeline.

        Uses a loop to support barge-in: when the user interrupts HAL's
        response, we re-enter transcription with the buffered barge-in
        audio instead of returning to idle.
        """
        try:
            while True:
                # Wait for end of speech and transcribe
                transcript = await self._transcribe_speech()

                if not transcript or len(transcript.strip()) < 2:
                    logger.info("No speech detected, returning to idle")
                    break

                logger.info(f"Transcript: {transcript}")

                # Log user speech
                self.transcript.log_user_speech(transcript)

                # Update state and generate response
                await self.set_state(config.State.PROCESSING)

                # Add user message to conversation
                self.conversation.add_user_message(transcript)
                await broadcast_conversation()

                # Generate and speak response
                self._barge_in_event.clear()
                await self._generate_and_speak(transcript)

                # If barge-in occurred, loop back to transcribe the interruption
                if self._barge_in_event.is_set():
                    logger.info("Processing barge-in speech")
                    self._barge_in_event.clear()
                    self.vad.reset()
                    self._speech_end_event.clear()
                    await self.set_state(config.State.LISTENING)
                    continue

                break

        except Exception as e:
            logger.error(f"Error processing speech: {e}", exc_info=True)
            await self._speak(get_error_response())

        finally:
            # Reset VAD and return to idle
            self.vad.reset()
            await self.set_state(config.State.IDLE)

    async def _transcribe_speech(self) -> str:
        """Transcribe buffered speech using Deepgram."""
        await self.stt.connect()
        self.stt.clear_transcript()

        try:
            # Send buffered audio
            for chunk in self._audio_buffer:
                await self.stt.send_audio(chunk)

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

            if utterance_end.is_set():
                logger.info("Speech ended (Deepgram utterance end)")
            elif self._speech_end_event.is_set():
                logger.info("Speech ended (VAD silence)")

            # Wait for final transcript
            await asyncio.sleep(0.1)
            return self.stt.get_final_transcript()

        finally:
            await self.stt.disconnect()

    async def _generate_and_speak(self, user_input: str) -> None:
        """Generate LLM response and synthesize speech with pipelined TTS.

        Uses a producer/consumer pattern: the LLM streams sentences into an
        asyncio.Queue while the consumer streams TTS audio chunks to the
        output buffer.  This eliminates inter-sentence silence gaps.

        Supports barge-in: both producer and consumer check _barge_in_event
        each iteration and bail out early when set.
        """
        await self.set_state(config.State.SPEAKING)

        # Add current visual context as a conversation message
        scene_description = self.vision.get_cached_description()
        if scene_description and scene_description != "No visual information available.":
            self.conversation.add_vision_message(scene_description)
            await broadcast_conversation()

        # Get conversation history and system prompt
        messages = self.conversation.get_messages()
        system_prompt = get_system_prompt()

        # Pipeline TTS — producer/consumer via asyncio.Queue
        sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
        spoken_sentences: list[str] = []

        async def producer():
            try:
                async for sentence in self.llm.generate_sentences(messages, system_prompt):
                    if self._barge_in_event.is_set():
                        break
                    await sentence_queue.put(sentence)
            finally:
                await sentence_queue.put(None)  # sentinel

        async def consumer():
            while not self._barge_in_event.is_set():
                # Poll queue with timeout so we can check barge-in
                try:
                    sentence = await asyncio.wait_for(
                        sentence_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                if sentence is None:
                    break

                spoken_sentences.append(sentence)

                try:
                    async for audio_chunk in self.tts.synthesize_stream(sentence):
                        if self._barge_in_event.is_set():
                            break
                        self.audio_output.write(audio_chunk)
                except Exception as e:
                    logger.error(f"TTS error: {e}")

            # Wait for playback to drain (unless barged-in)
            while self.audio_output.is_playing() and not self._barge_in_event.is_set():
                await asyncio.sleep(0.05)

        # Run producer and consumer concurrently
        producer_task = asyncio.create_task(producer())
        await consumer()

        # Ensure producer is cleaned up
        if not producer_task.done():
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass
        else:
            await producer_task  # propagate exceptions

        # On barge-in, immediately silence HAL
        if self._barge_in_event.is_set():
            self.audio_output.clear_buffer()

        # Record whatever was spoken
        full_response = " ".join(spoken_sentences)
        if full_response:
            self.conversation.add_assistant_message(full_response.strip())
            await broadcast_conversation()
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
        await self.set_state(config.State.IDLE)

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
