import logging
logger = logging.getLogger(__name__)

import os
import threading
import time
from pathlib import Path
from typing import Optional, Callable, Any, Dict
import numpy as np
import websockets

from colors import Colors

try:
    from deepgram import DeepgramClient
    from deepgram.listen.v1.socket_client import (
        ListenV1Results as ListenV1ResultsEvent,
        ListenV1SpeechStarted as ListenV1SpeechStartedEvent,
        ListenV1UtteranceEnd as ListenV1UtteranceEndEvent,
        V1SocketClient,
        AsyncV1SocketClient,
    )
    # ControlMessage / MediaMessage removed in SDK v7 — raw bytes sent directly
    ListenV1ControlMessage = None
    ListenV1MediaMessage = None
    DEEPGRAM_AVAILABLE = True
except Exception as e:
    DEEPGRAM_AVAILABLE = False
    ListenV1ResultsEvent = None
    ListenV1SpeechStartedEvent = None
    ListenV1UtteranceEndEvent = None
    ListenV1ControlMessage = None
    ListenV1MediaMessage = None

try:
    from dotenv import load_dotenv
    BASE_DIR = Path(__file__).resolve().parent.parent
    env_specific = BASE_DIR / ".enve"
    env_default = BASE_DIR / ".env"
    if env_specific.exists():
        load_dotenv(env_specific, override=True)
    if env_default.exists():
        load_dotenv(env_default, override=True)
except Exception:
    # If python-dotenv is not installed, silently continue; env vars may be set elsewhere
    pass

# --- Backend selection ---
STT_BACKEND = os.getenv("STT_BACKEND", "deepgram").lower()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_STT_MODEL = os.getenv("DEEPGRAM_STT_MODEL", "nova-3")
DEEPGRAM_LANGUAGE = os.getenv("DEEPGRAM_LANGUAGE")  # Optional override; defaults to source_language

SAMPLE_RATE: int = 16000

# ---------------------------------------------------------------------------
# Deepgram-based Transcriber (STT over API)
# ---------------------------------------------------------------------------
class DeepgramTranscriptionProcessor:
    """
    Lightweight transcription processor that streams audio to Deepgram's
    real-time API and relays partial/final transcripts through the same
    callback interface used by the local RealtimeSTT-backed processor.
    """

    def __init__(
            self,
            source_language: str = "en",
            realtime_transcription_callback: Optional[Callable[[str], None]] = None,
            full_transcription_callback: Optional[Callable[[str], None]] = None,
            potential_full_transcription_callback: Optional[Callable[[str], None]] = None,
            potential_full_transcription_abort_callback: Optional[Callable[[], None]] = None,
            potential_sentence_end: Optional[Callable[[str], None]] = None,
            before_final_sentence: Optional[Callable[[Optional[np.ndarray], Optional[str]], bool]] = None,
            silence_active_callback: Optional[Callable[[bool], None]] = None,
            on_recording_start_callback: Optional[Callable[[], None]] = None,
            is_orpheus: bool = False,
            local: bool = True,
            tts_allowed_event: Optional[threading.Event] = None,
            pipeline_latency: float = 0.5,
            recorder_config: Optional[Dict[str, Any]] = None,
        ) -> None:
        self.source_language = source_language
        self.realtime_transcription_callback = realtime_transcription_callback
        self.full_transcription_callback = full_transcription_callback
        self.potential_full_transcription_callback = potential_full_transcription_callback
        self.potential_full_transcription_abort_callback = potential_full_transcription_abort_callback
        self.potential_sentence_end = potential_sentence_end
        self.before_final_sentence = before_final_sentence
        self.silence_active_callback = silence_active_callback
        self.on_recording_start_callback = on_recording_start_callback
        self.on_tts_allowed_to_synthesize: Optional[Callable[[], None]] = None
        self.turn_detection = None  # Not used with Deepgram backend

        self.realtime_text: Optional[str] = None
        self.final_transcription: Optional[str] = None
        self.shutdown_performed: bool = False

        self.sample_rate = SAMPLE_RATE
        self.model = DEEPGRAM_STT_MODEL
        self.language = DEEPGRAM_LANGUAGE or self.source_language

        self._accumulated_parts = []
        self._last_final_time = 0.0

        self._client: Optional["DeepgramClient"] = None
        self._socket_ctx = None
        self._socket = None
        self._stop_event = threading.Event()
        self._socket_lock = threading.Lock()
        self._keepalive_thread: Optional[threading.Thread] = None
        self._keepalive_stop = threading.Event()

    # --- Internal helpers ---
    def _ensure_client(self):
        if not DEEPGRAM_AVAILABLE:
            raise ImportError("deepgram SDK not installed; install deepgram-sdk or select a different STT backend.")
        if not DEEPGRAM_API_KEY:
            raise RuntimeError("DEEPGRAM_API_KEY is not set; provide it in .enve/.env or environment variables.")
        if self._client is None:
            self._client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

    def _ensure_connection(self):
        if self._socket is not None:
            return
        self._ensure_client()
        logger.info("🎤🔌 Opening Deepgram streaming connection (model=%s, language=%s, endpointing=1200ms)", self.model, self.language)
        ctx = self._client.listen.v1.connect(
            model=self.model,
            language=self.language,
            encoding="linear16",
            sample_rate=str(self.sample_rate),
            interim_results="true",
            smart_format="true",
            punctuate="true",
            numerals="true",
            vad_events="true",
            endpointing="1200",
            utterance_end_ms="1200",
        )
        self._socket_ctx = ctx
        self._socket = ctx.__enter__()
        self._start_keepalive()

    def _close_socket(self):
        with self._socket_lock:
            if self._socket:
                self._stop_keepalive()
                try:
                    if hasattr(self._socket, "send_close_stream"):
                        self._socket.send_close_stream()
                except Exception:
                    pass
                try:
                    self._socket._websocket.close()
                except Exception:
                    pass
                self._socket = None
            if self._socket_ctx:
                try:
                    self._socket_ctx.__exit__(None, None, None)
                except Exception:
                    pass
                self._socket_ctx = None

    def _finalize_turn(self, final_text: str = "") -> None:
        """
        Dispatches the finalized complete user utterance when speech_final or silence occurs.
        """
        if not final_text and self._accumulated_parts:
            final_text = " ".join(self._accumulated_parts).strip()
        self._accumulated_parts.clear()

        if not final_text:
            return

        logger.info(f"👂✅ Deepgram complete turn finalized: {final_text}")
        self.final_transcription = final_text
        if self.before_final_sentence:
            try:
                self.before_final_sentence(None, final_text)
            except Exception as e:
                logger.warning(f"👂⚠️ Error in before_final_sentence callback: {e}")
        if self.full_transcription_callback:
            self.full_transcription_callback(final_text)

    def _handle_result(self, message: ListenV1ResultsEvent):
        if not message.channel or not message.channel.alternatives:
            return
        alt = message.channel.alternatives[0]
        transcript = alt.transcript
        if not transcript:
            return

        is_final = bool(getattr(message, "is_final", False))
        speech_final = bool(getattr(message, "speech_final", False))

        if is_final:
            clean_part = transcript.strip()
            if clean_part:
                if not self._accumulated_parts or self._accumulated_parts[-1] != clean_part:
                    self._accumulated_parts.append(clean_part)
            full_text = " ".join(self._accumulated_parts).strip()
            self.realtime_text = full_text
            self._last_final_time = time.time()

            if speech_final:
                # User finished speaking and pause threshold satisfied
                self._finalize_turn(full_text)
            else:
                logger.info(f"👂📝 Deepgram chunk final (in-progress): {clean_part} | Rolling total: '{full_text}'")
                if self.realtime_transcription_callback:
                    self.realtime_transcription_callback(full_text)
                if self.potential_full_transcription_callback:
                    self.potential_full_transcription_callback(full_text)
        else:
            interim_full = " ".join(self._accumulated_parts + [transcript.strip()]).strip()
            self.realtime_text = interim_full
            logger.debug(f"👂📝 Deepgram interim partial: {interim_full}")
            if self.realtime_transcription_callback:
                self.realtime_transcription_callback(interim_full)
            if self.potential_full_transcription_callback:
                self.potential_full_transcription_callback(interim_full)

    # --- Public API mirroring the local recorder-backed processor ---
    def transcribe_loop(self) -> None:
        """
        Opens a Deepgram streaming connection and blocks while consuming events.
        Designed to run inside a background thread via AudioInputProcessor.
        """
        if self.shutdown_performed:
            return
        while not self.shutdown_performed and not self._stop_event.is_set():
            try:
                self._ensure_connection()
            except Exception as e:
                logger.error(f"👂💥 Failed to start Deepgram streaming session: {e}")
                self.shutdown_performed = True
                break

            try:
                for message in self._socket:
                    if self._stop_event.is_set() or self.shutdown_performed:
                        break
                    if isinstance(message, ListenV1ResultsEvent):
                        self._handle_result(message)
                    elif isinstance(message, ListenV1SpeechStartedEvent):
                        if self.on_recording_start_callback:
                            self.on_recording_start_callback()
                        if self.silence_active_callback:
                            self.silence_active_callback(False)
                    elif isinstance(message, ListenV1UtteranceEndEvent):
                        if self.silence_active_callback:
                            self.silence_active_callback(True)
                        if self._accumulated_parts:
                            self._finalize_turn()
                        if self.potential_full_transcription_abort_callback:
                            self.potential_full_transcription_abort_callback()
            except websockets.exceptions.ConnectionClosedError as e:
                if not self.shutdown_performed:
                    logger.warning(f"👂⚠️ Deepgram connection closed, will retry: {e}")
            except Exception as e:
                if not self.shutdown_performed:
                    logger.error(f"👂💥 Deepgram streaming error: {e}", exc_info=True)
            finally:
                self._close_socket()
                if not self.shutdown_performed and not self._stop_event.is_set():
                    time.sleep(1.0)  # brief backoff before reconnect

    def feed_audio(self, chunk: bytes, audio_meta_data: Optional[Dict[str, Any]] = None) -> None:
        if self.shutdown_performed or self._stop_event.is_set():
            return
        try:
            self._ensure_connection()
            with self._socket_lock:
                if self._socket:
                    self._socket.send_media(chunk)
        except Exception as e:
            logger.error(f"👂💥 Failed to feed audio to Deepgram: {e}", exc_info=True)

    def abort_generation(self) -> None:
        self._accumulated_parts.clear()
        self.realtime_text = None
        try:
            with self._socket_lock:
                if self._socket and hasattr(self._socket, "send_finalize"):
                    self._socket.send_finalize()
        except Exception:
            pass

    def perform_final(self, audio_bytes: Optional[bytes] = None) -> None:
        self.abort_generation()

    def shutdown(self) -> None:
        logger.info("👂🔌 Shutting down Deepgram transcription processor...")
        self.shutdown_performed = True
        self._stop_event.set()
        self._close_socket()
        self._stop_keepalive()

    # Compatibility helpers for AudioInputProcessor
    def get_audio_copy(self) -> Optional[np.ndarray]:
        return None

    def get_last_audio_copy(self) -> Optional[np.ndarray]:
        return None

    def _keepalive_loop(self):
        while not self._keepalive_stop.is_set() and not self.shutdown_performed:
            try:
                with self._socket_lock:
                    if self._socket and hasattr(self._socket, "send_keep_alive"):
                        self._socket.send_keep_alive()
            except Exception:
                pass
            self._keepalive_stop.wait(timeout=8.0)

    def _start_keepalive(self):
        self._keepalive_stop.clear()
        if self._keepalive_thread and self._keepalive_thread.is_alive():
            return
        self._keepalive_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        self._keepalive_thread.start()

    def _stop_keepalive(self):
        self._keepalive_stop.set()
        if self._keepalive_thread and self._keepalive_thread.is_alive():
            self._keepalive_thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Factory to create the appropriate transcriber backend
# ---------------------------------------------------------------------------
def create_transcription_processor(
        source_language: str = "en",
        backend: Optional[str] = None,
        **kwargs: Any,
    ):
    """
    Returns a transcription processor instance based on the configured backend.

    Args:
        source_language: Language passed to the transcriber.
        backend: Name of the backend ("deepgram" or "realtimestt").
        **kwargs: Additional arguments forwarded to the processor constructor.
    """
    chosen_backend = (backend or STT_BACKEND or "deepgram").lower()
    if chosen_backend == "deepgram":
        return DeepgramTranscriptionProcessor(source_language=source_language, **kwargs)
    else:
        raise ValueError(f"Unsupported STT backend: {chosen_backend}. Only 'deepgram' is supported.")
