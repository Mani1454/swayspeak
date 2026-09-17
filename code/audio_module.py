import json
import logging
import os
import threading
import time
from queue import Queue, Full
from typing import Generator, Optional, Callable
from pathlib import Path

# Load .env variables
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
    pass

# Deepgram Import
try:
    from deepgram import DeepgramClient
    DEEPGRAM_AVAILABLE = True
except Exception:
    DEEPGRAM_AVAILABLE = False

logger = logging.getLogger(__name__)

# Configuration
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_TTS_MODEL = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en")
DEEPGRAM_TTS_SAMPLE_RATE = int(os.getenv("DEEPGRAM_TTS_SAMPLE_RATE", "24000"))
DEEPGRAM_TTS_SPEED = float(os.getenv("DEEPGRAM_TTS_SPEED", "0.82"))

class AudioProcessor:
    """
    Manages Text-to-Speech (TTS) synthesis.
    Cleaned version: Supports Deepgram API only with pacing control.
    """
    def __init__(self, engine: str = "deepgram"):
        """
        Initializes the AudioProcessor with the API client.
        """
        self.engine_name = engine.lower()
        self.on_first_audio_chunk_synthesize: Optional[Callable[[], None]] = None
        self.tts_inference_time = 0.0
        
        # Initialize Deepgram
        if self.engine_name == "deepgram":
            if not DEEPGRAM_AVAILABLE:
                raise ImportError("deepgram-sdk is required. pip install deepgram-sdk")
            if not DEEPGRAM_API_KEY:
                raise RuntimeError("DEEPGRAM_API_KEY is not set.")
            
            self.deepgram_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
            self.deepgram_tts_model = DEEPGRAM_TTS_MODEL
            self.deepgram_tts_speed = DEEPGRAM_TTS_SPEED
            logger.info(f"🔊 AudioProcessor initialized with Deepgram ({self.deepgram_tts_model}, speed={self.deepgram_tts_speed})")
        else:
            raise ValueError(f"Unsupported engine: {engine}. Only 'deepgram' is supported in this configuration.")

    def _fire_first_chunk_callback(self, generation_string: str):
        """Helper to trigger the first chunk callback safely."""
        if self.on_first_audio_chunk_synthesize:
            try:
                logger.info(f"👄🚀 {generation_string} Firing on_first_audio_chunk_synthesize (Deepgram).")
                self.on_first_audio_chunk_synthesize()
            except Exception as e:
                logger.error(f"👄💥 {generation_string} Error in on_first_audio_chunk_synthesize callback: {e}", exc_info=True)

    def synthesize(
            self,
            text: str,
            audio_chunks: Queue, 
            stop_event: threading.Event,
            generation_string: str = "",
        ) -> bool:
        """
        Synthesizes audio from a complete text string using Deepgram.
        """
        if self.engine_name != "deepgram":
            return False

        # Strictly isolate conversational_reply if a structured JSON payload was provided
        if isinstance(text, str) and text.strip().startswith("{") and "conversational_reply" in text:
            try:
                parsed_data = json.loads(text.strip())
                if isinstance(parsed_data, dict) and "conversational_reply" in parsed_data:
                    logger.info(f"👄🔒 {generation_string} TTS routing guard: Isolated conversational_reply from JSON, ignoring correction fields.")
                    text = parsed_data["conversational_reply"]
            except Exception as json_err:
                logger.debug(f"TTS JSON guard parse ignore: {json_err}")

        logger.info(f"👄▶️ {generation_string} Deepgram synthesis starting. Text: {text[:50]}...")
        
        if not text or not str(text).strip():
            logger.error(f"👄💥 {generation_string} Deepgram TTS aborted: input text empty.")
            return False

        try:
            # Stream directly from Deepgram API with speed parameter if supported
            kwargs = {
                "text": text,
                "model": self.deepgram_tts_model,
                "encoding": "linear16",
                "sample_rate": DEEPGRAM_TTS_SAMPLE_RATE,
                "container": "none",
                "request_options": {"chunk_size": 4096},
            }
            if self.deepgram_tts_speed and self.deepgram_tts_speed != 1.0:
                try:
                    stream = self.deepgram_client.speak.v1.audio.generate(
                        **kwargs,
                        speed=self.deepgram_tts_speed
                    )
                except Exception as speed_err:
                    logger.warning(f"👄⚠️ {generation_string} Speed parameter not supported by model {self.deepgram_tts_model}, falling back without speed: {speed_err}")
                    stream = self.deepgram_client.speak.v1.audio.generate(**kwargs)
            else:
                stream = self.deepgram_client.speak.v1.audio.generate(**kwargs)
        except Exception as e:
            logger.error(f"👄💥 {generation_string} Failed to start Deepgram TTS: {e}")
            return False

        first_chunk_seen = False
        try:
            for chunk in stream:
                if stop_event.is_set():
                    logger.info(f"👄🛑 {generation_string} Deepgram synthesis interrupted by stop_event.")
                    return False
                
                if not chunk:
                    continue

                try:
                    audio_chunks.put_nowait(chunk)
                except Full:
                    logger.warning(f"👄⚠️ {generation_string} Deepgram audio queue full, dropping chunk.")
                    continue
                
                if not first_chunk_seen:
                    first_chunk_seen = True
                    self._fire_first_chunk_callback(generation_string)
            
            logger.info(f"👄✅ {generation_string} Deepgram synthesis complete.")
            return True

        except Exception as e:
            logger.error(f"👄💥 {generation_string} Deepgram synthesis failed mid-stream: {e}", exc_info=True)
            return False

    def synthesize_generator(
            self,
            generator: Generator[str, None, None],
            audio_chunks: Queue,
            stop_event: threading.Event,
            generation_string: str = "",
        ) -> bool:
        """
        Accumulates text from a generator and sends it to Deepgram.
        """
        # Note: We maintain the accumulation logic because Deepgram REST API 
        # works best with full context for prosody, and the 'speech_pipeline_manager'
        # handles latency by sending the first sentence separately.
        
        combined_text = ""
        try:
            for piece in generator:
                if stop_event.is_set():
                    logger.info(f"👄🛑 {generation_string} Deepgram final synthesis aborted before start.")
                    return False
                combined_text += piece
        except Exception as e:
            logger.error(f"👄💥 {generation_string} Error consuming generator for Deepgram: {e}", exc_info=True)
            return False
        
        if stop_event.is_set():
            return False
            
        return self.synthesize(combined_text, audio_chunks, stop_event, generation_string)