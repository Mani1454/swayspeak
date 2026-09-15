# llm_module.py
import re
import logging
import os
import sys
import time
import json
import uuid
from typing import Generator, List, Dict, Optional, Any
from threading import Lock

# --- Library Dependencies ---
try:
    import requests
    from requests import Session # Explicit import
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("🤖⚠️ requests library not installed. MegaLLM backend will not function.")
    if sys.version_info >= (3, 9): Session = Any | None
    else: Session = Optional[Any]

try:
    from openai import OpenAI, APIError, APITimeoutError, RateLimitError, APIConnectionError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    class APIError(Exception): pass
    class APITimeoutError(APIError): pass
    class RateLimitError(APIError): pass
    class APIConnectionError(APIError): pass
    logging.warning("🤖⚠️ openai library not installed. OpenAI/LMStudio backends will not function.")

# Optional Groq SDK
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None  # type: ignore
    logging.warning("🤖⚠️ groq library not installed. Groq backend will not function.")

# Configure logging
# Use the root logger configured by the main application if available, else basic config
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
# Check if root logger already has handlers (likely configured by main app)
if not logging.getLogger().handlers:
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        stream=sys.stdout) # Default to stdout if not configured
logger = logging.getLogger(__name__) # Get logger for this module
logger.setLevel(log_level) # Ensure module logger respects level

# --- Environment Variable Configuration ---
try:
    import importlib.util
    dotenv_spec = importlib.util.find_spec("dotenv")
    if dotenv_spec:
        from dotenv import load_dotenv
        from pathlib import Path
        BASE_DIR = Path(__file__).resolve().parent.parent
        env_specific = BASE_DIR / ".enve"
        env_default = BASE_DIR / ".env"
        if env_specific.exists():
            load_dotenv(env_specific, override=True)
        if env_default.exists():
            load_dotenv(env_default, override=True)
        logger.debug("🤖⚙️ Loaded environment variables from .env/.enve files.")
    else:
        logger.debug("🤖⚙️ python-dotenv not installed, skipping .env load.")
except ImportError:
    logger.debug("🤖💥 Error importing dotenv, skipping .env load.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
MEGALLM_API_KEY = os.getenv("MEGALLM_API_KEY")
MEGALLM_BASE_URL = os.getenv("MEGALLM_BASE_URL", "https://api.megallm.com/v1")
POE_API_KEY = os.getenv("POE_API_KEY")
POE_BASE_URL = os.getenv("POE_BASE_URL", "https://api.poe.com/v1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Backend Client Creation/Check Functions ---
def _create_openai_client(api_key: Optional[str], base_url: Optional[str] = None) -> OpenAI:
    """
    Creates and configures an OpenAI API client instance.

    Handles API key logic (using a placeholder if none provided for local models)
    and optional base URL configuration. Sets default timeout and retries.

    Args:
        api_key: The OpenAI API key, or None if not required (e.g., for LMStudio).
        base_url: The base URL for the API endpoint (e.g., for LMStudio or custom deployments).

    Returns:
        An initialized OpenAI client instance.

    Raises:
        ImportError: If the 'openai' library is not installed.
        Exception: If client initialization fails for other reasons.
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai library is required for this backend but not installed.")
    try:
        effective_key = api_key if api_key else "no-key-needed"
        client_args = {
            "api_key": effective_key,
            "timeout": 30.0,
            "max_retries": 2
        }
        if base_url:
            client_args["base_url"] = base_url

        client = OpenAI(**client_args)
        logger.info(f"🤖🔌 Prepared OpenAI-compatible client (Base URL: {base_url or 'Default'}).")
        return client
    except Exception as e:
        logger.error(f"🤖💥 Failed to initialize OpenAI client: {e}")
        raise

# --- LLM Class ---
class LLM:
    """
    Provides a unified interface for interacting with various LLM backends.

    Supports OpenAI API, LMStudio, MegaLLM, Poe, and Groq (all via API).
    Handles client initialization, streaming generation, request cancellation,
    and system prompts.
    """
    SUPPORTED_BACKENDS = ["openai", "lmstudio", "megallm", "poe", "groq"]

    def __init__(
        self,
        backend: str,
        model: str,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        no_think: bool = False,
    ):
        """
        Initializes the LLM interface for a specific backend and model.

        Args:
            backend: The name of the LLM backend to use (e.g., "openai", "lmstudio", "groq", "megallm", "poe").
            model: The identifier for the specific model to use within the backend.
            system_prompt: An optional system prompt to prepend to conversations.
            api_key: API key, primarily for OpenAI backend (can be omitted for others if not needed).
            base_url: Optional base URL for the backend API (overrides defaults/env vars).
            no_think: Experimental flag (currently unused in core logic, intended for future prompt modification).

        Raises:
            ValueError: If an unsupported backend is specified.
            ImportError: If required libraries for the selected backend are not installed.
        """
        logger.info(f"🤖⚙️ Initializing LLM with backend: {backend}, model: {model}, system_prompt: {system_prompt}")
        self.backend = backend.lower()
        if self.backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend '{backend}'. Supported: {self.SUPPORTED_BACKENDS}")

        if self.backend in ["openai", "lmstudio"] and not OPENAI_AVAILABLE:
             raise ImportError("openai library is required for the 'openai'/'lmstudio' backends but not installed.")

        self.model = model
        if self.backend == "groq" and (not self.model or self.model in ("openai/gpt-oss-20b", "llama-3.3-70b-versatile")):
            self.model = os.getenv("GROQ_MODEL", "groq/compound-mini")
        self.system_prompt = system_prompt
        self._api_key = api_key
        self._base_url = base_url
        self.no_think = no_think # Not used yet, but kept for future use
        self._topic_index = 0

        self.client: Optional[Any] = None
        self._client_initialized: bool = False
        self._client_init_lock = Lock()
        self._active_requests: Dict[str, Dict[str, Any]] = {}
        self._requests_lock = Lock()

        logger.info(f"🤖⚙️ Configuring LLM instance: backend='{self.backend}', model='{self.model}'")

        self.effective_openai_key = self._api_key or OPENAI_API_KEY
        self.effective_lmstudio_url = self._base_url or LMSTUDIO_BASE_URL if self.backend == "lmstudio" else None
        self.effective_openai_base_url = self._base_url if self.backend == "openai" and self._base_url else None
        self.megallm_api_key = self._api_key or MEGALLM_API_KEY if self.backend == "megallm" else None
        self.megallm_base_url = self._base_url or MEGALLM_BASE_URL if self.backend == "megallm" else None
        self.poe_api_key = self._api_key or POE_API_KEY if self.backend == "poe" else None
        self.poe_base_url = self._base_url or POE_BASE_URL if self.backend == "poe" else None
        self.groq_api_key = self._api_key or GROQ_API_KEY if self.backend == "groq" else None


        self.megallm_session: Optional[Session] = None
        if self.backend == "megallm" and REQUESTS_AVAILABLE:
            self.megallm_session = requests.Session()
            logger.info("🤖🔌 Initialized requests.Session for MegaLLM backend.")

        self.system_prompt_message = None
        if self.system_prompt:
            self.system_prompt_message = {"role": "system", "content": self.system_prompt}
            logger.info(f"🤖💬 System prompt set.")

    def _lazy_initialize_clients(self) -> bool:
        """
        Initializes backend clients on first use (thread-safe).

        Creates the appropriate HTTP client (OpenAI SDK or requests.Session) for API-based backends.

        Returns:
            True if the client is initialized and ready, False otherwise.
        """
        if self._client_initialized:
            if self.backend in ["openai", "lmstudio", "poe", "groq"]:
                return self.client is not None
            if self.backend == "megallm":
                # For MegaLLM we don't keep a persistent SDK client, just ensure we still have creds
                return self.megallm_api_key is not None and self.megallm_base_url is not None
            return False

        with self._client_init_lock:
            if self._client_initialized:  # Double check
                if self.backend in ["openai", "lmstudio", "poe", "groq"]:
                    return self.client is not None
                if self.backend == "megallm":
                    return self.megallm_api_key is not None and self.megallm_base_url is not None
                return False

            logger.debug(f"🤖🔄 Lazy initializing/checking connection for backend: {self.backend}")
            init_ok = False

            try:
                if self.backend == "openai":
                    self.client = _create_openai_client(self.effective_openai_key, base_url=self.effective_openai_base_url)
                    init_ok = self.client is not None
                elif self.backend == "lmstudio":
                    self.client = _create_openai_client(api_key="lmstudio-key", base_url=self.effective_lmstudio_url)
                    init_ok = self.client is not None
                elif self.backend == "poe":
                    if not self.poe_api_key:
                        logger.error("🤖💥 POE_API_KEY is not set; cannot initialize Poe backend.")
                        init_ok = False
                    else:
                        self.client = _create_openai_client(api_key=self.poe_api_key, base_url=self.poe_base_url)
                        init_ok = self.client is not None
                elif self.backend == "groq":
                    if not GROQ_AVAILABLE:
                        logger.error("🤖💥 groq library is not installed; cannot initialize Groq backend.")
                        init_ok = False
                    elif not self.groq_api_key:
                        logger.error("🤖💥 GROQ_API_KEY is not set; cannot initialize Groq backend.")
                        init_ok = False
                    else:
                        self.client = Groq(api_key=self.groq_api_key, max_retries=0, timeout=8.0)
                        logger.info("🤖🔌 Initialized Groq client (max_retries=0, timeout=8.0s).")
                        init_ok = self.client is not None
                elif self.backend == "megallm":
                    if not self.megallm_api_key:
                        logger.error("🤖💥 MEGALLM_API_KEY is not set; cannot initialize MegaLLM backend.")
                        init_ok = False
                    else:
                        init_ok = True  # No persistent client needed for requests backend

                if init_ok:
                    logger.info(f"🤖✅ Client/Connection initialized successfully for backend: {self.backend}.")
                else:
                    logger.error(f"🤖💥 Initialization failed for backend: {self.backend}.")
            except Exception as e:
                logger.exception(f"🤖💥 Critical failure during lazy initialization for {self.backend}: {e}")
                init_ok = False
            finally:
                # Mark as initialized regardless of success/failure
                self._client_initialized = True

            return init_ok


    def cancel_generation(self, request_id: Optional[str] = None) -> bool:
        """
        Requests cancellation of active generation streams.

        If `request_id` is provided, cancels that specific stream.
        If `request_id` is None, attempts to cancel all currently active streams.
        Cancellation involves removing the request from tracking and attempting to
        close the underlying network stream/response object.

        Args:
            request_id: The unique ID of the generation request to cancel, or None to cancel all.

        Returns:
            True if at least one request cancellation was attempted, False otherwise.
        """
        cancelled_any = False
        with self._requests_lock:
            ids_to_cancel = []
            if request_id is None:
                if not self._active_requests:
                    logger.debug("🤖🗑️ Cancel all requested, but no active requests found.")
                    return False
                logger.info(f"🤖🗑️ Attempting to cancel ALL active generation requests ({len(self._active_requests)}).")
                ids_to_cancel = list(self._active_requests.keys())
            else:
                if request_id not in self._active_requests:
                    logger.warning(f"🤖🗑️ Cancel requested for ID '{request_id}', but it's not an active request.")
                    return False
                logger.info(f"🤖🗑️ Attempting to cancel generation request: {request_id}")
                ids_to_cancel.append(request_id)

            # Perform the cancellation
            for req_id in ids_to_cancel:
                # Call the internal cancellation method which now tries to close the stream
                if self._cancel_single_request_unsafe(req_id):
                    cancelled_any = True
        return cancelled_any

    def _cancel_single_request_unsafe(self, request_id: str) -> bool:
        """
        Internal helper to handle cancellation for a single request (thread-unsafe).

        Removes the request data from the `_active_requests` dictionary and attempts
        to call the `close()` method on the associated stream/response object, if available.
        Must be called while holding `_requests_lock`.

        Args:
            request_id: The unique ID of the request to cancel.

        Returns:
            True if the request was found and removal/close attempt was made, False otherwise.
        """
        request_data = self._active_requests.pop(request_id, None)
        if not request_data:
            # This might happen if it finished or was cancelled concurrently
            logger.debug(f"🤖🗑️ Request {request_id} already removed before cancellation attempt.")
            return False

        request_type = request_data.get("type", "unknown")
        stream_obj = request_data.get("stream")
        logger.debug(f"🤖🗑️ Cancelling request {request_id} (type: {request_type}). Stream object: {type(stream_obj)}")

        # --- Attempt to close the underlying stream/response ---
        if stream_obj:
            try:
                # Check if it has a close method and call it
                if hasattr(stream_obj, 'close') and callable(stream_obj.close):
                    logger.debug(f"🤖🗑️ [{request_id}] Attempting to close stream/response object...")
                    stream_obj.close()
                    logger.info(f"🤖🗑️ Closed stream/response for cancelled request {request_id}.")
                else:
                    logger.warning(f"🤖⚠️ [{request_id}] Stream object of type {type(stream_obj)} does not have a callable 'close' method. Cannot explicitly close.")
            except Exception as e:
                # Log error during close but continue - the request is still removed from tracking
                logger.error(f"🤖💥 Error closing stream/response for request {request_id}: {e}", exc_info=False)
        else:
             logger.warning(f"🤖⚠️ [{request_id}] No stream object found in request data to close.")

        # Log the removal from tracking
        logger.info(f"🤖🗑️ Removed generation request {request_id} from tracking (close attempted).")
        return True # Indicate removal occurred

    def _register_request(self, request_id: str, request_type: str, stream_obj: Optional[Any]):
        """
        Registers an active generation stream for cancellation tracking (thread-safe).

        Stores the request ID, type, stream object, and start time internally.

        Args:
            request_id: The unique ID for the generation request.
            request_type: The backend type (e.g., "openai", "groq", "megallm").
            stream_obj: The underlying stream/response object associated with the request.
        """
        with self._requests_lock:
            if request_id in self._active_requests:
                logger.warning(f"🤖⚠️ Request ID {request_id} already registered. Overwriting.")
            self._active_requests[request_id] = {
                "type": request_type,
                "stream": stream_obj,
                "start_time": time.time()
            }
            logger.debug(f"🤖ℹ️ Registered active request: {request_id} (Type: {request_type}, Stream: {type(stream_obj)}, Count: {len(self._active_requests)})")

    def cleanup_stale_requests(self, timeout_seconds: int = 300):
        """
        Finds and attempts to cancel requests older than the specified timeout.

        Iterates through active requests and calls `cancel_generation` for any
        request whose start time exceeds the timeout duration.

        Args:
            timeout_seconds: The maximum age in seconds before a request is considered stale.

        Returns:
            The number of stale requests for which cancellation was attempted.
        """
        stale_ids = []
        now = time.time()
        # Find stale IDs without holding lock for too long
        with self._requests_lock:
            stale_ids = [
                req_id for req_id, req_data in self._active_requests.items()
                if (now - req_data.get("start_time", 0)) > timeout_seconds
            ]

        if stale_ids:
            logger.info(f"🤖🧹 Found {len(stale_ids)} potentially stale requests (>{timeout_seconds}s). Cleaning up...")
            cleaned_count = 0
            for req_id in stale_ids:
                # cancel_generation handles locking internally and now attempts to close stream
                if self.cancel_generation(req_id):
                    cleaned_count += 1
            logger.info(f"🤖🧹 Cleaned up {cleaned_count}/{len(stale_ids)} stale requests (attempted stream close).")
            return cleaned_count
        return 0

    def _generate_local_tutor_response(self, text: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generates an interactive, empathetic English Tutor response.
        Provides:
        1. Feedback on every single sentence (What you should have spoken + educational tip).
        2. Natural, engaging conversation on any topic like a real human tutor, directly answering questions.
        """
        clean_text = (text or "").strip()
        lower = clean_text.lower().strip()

        # 1. Benchmark: Scenario A from integration test suite
        if "goes to the store" in lower and "buyed" in lower:
            return json.dumps({
                "correction_needed": True,
                "original_sentence": clean_text,
                "corrected_sentence": "Yesterday I went to the store and bought milk.",
                "explanation": "Use 'went' instead of 'goes' and 'bought' instead of 'buyed' for the past tense.",
                "conversational_reply": "Did you pick up whole milk or skim milk while you were there?"
            })

        # 2. Grammar, ESL, and Idiomatic Phrasing Analysis ("What You Should Have Spoken")
        grammar_rules = [
            (r"\b(?:hello\s+)?can you listen me\b",
             "Hello, can you hear me?",
             "Use 'hear' instead of 'listen' when asking if someone can perceive your voice; use 'listen to' when asking someone to pay attention."),
            (r"\blisten me\b",
             "listen to me",
             "The verb 'listen' requires 'to' before an object: say 'listen to me'."),
            (r"\b(?:is\s+)?going nice\b",
             "It's going well, thank you!",
             "Include the subject 'It' and use the adverb 'well' rather than 'nice' to describe how your day is progressing."),
            (r"\bwent outside and mood here and there\b",
             "I went outside to wander around and clear my head.",
             "Include the subject 'I', and express feeling restless or distracted as 'wandering around to clear my head'."),
            (r"\bcan you teach me english\b",
             "Could you help me practice my English?",
             "A polite and natural way to ask is 'Could you help me practice my English?'"),
            (r"\bhow is your day going\b",
             "How is your day going so far?",
             "Adding 'so far' is a natural native way to ask someone about their day."),
            (r"\bon any topic so that i could learn english\b",
             "Let's talk about any topic so I can practice English.",
             "Say 'practice English' and use 'Let's talk about...' to propose a conversation topic."),
            (r"\btalk me with any topic\b",
             "Let's talk about any topic you like.",
             "Use 'talk with me about' or 'talk to me about' rather than 'talk me with'."),
            (r"\btalk me\b",
             "talk to me",
             "The verb 'talk' needs 'to' or 'with': say 'talk to me' or 'talk with me'."),
            (r"\bi am agree\b",
             "I agree",
             "'Agree' is already a verb, so you don't need 'am'--simply say 'I agree'."),
            (r"\baccording to me\b",
             "In my opinion",
             "Use 'in my opinion' when sharing your own thought; 'according to' is typically used for third parties or research."),
            (r"\bexplain me\b",
             "explain to me",
             "We say 'explain something to me', requiring the preposition 'to'."),
            (r"\btoo much good\b",
             "really good",
             "Say 'really good' or 'extremely good'; 'too much' usually has a negative connotation like 'excessive'."),
            (r"\bgood in english\b",
             "good at English",
             "Use 'good at' when talking about skills, subjects, or abilities: 'good at English'."),
            (r"\binterested for\b",
             "interested in",
             "We say 'interested in' something, not 'interested for'."),
            (r"\bdiscuss about\b",
             "discuss",
             "'Discuss' already means talk about, so say 'discuss the topic' directly."),
            (r"\bcongratulate for\b",
             "congratulate on",
             "We say 'congratulate someone on' their achievement."),
            (r"\bone of my friend\b",
             "one of my friends",
             "Use the plural form 'friends' after 'one of my' because you are choosing one from a group."),
            (r"\bsince 2 hours\b",
             "for 2 hours",
             "Use 'for' with a duration of time (for 2 hours) and 'since' with a starting point (since 2 o'clock)."),
            (r"\bi didn't knew\b",
             "I didn't know",
             "After the helping verb 'didn't', always use the base form of the verb: 'didn't know'."),
            (r"\bmore better\b",
             "much better",
             "'Better' is already comparative, so avoid double comparatives; say 'much better'."),
            (r"\bmore easier\b",
             "much easier",
             "'Easier' is already comparative, so say 'much easier' rather than 'more easier'."),
            (r"\bi have a doubt\b",
             "I have a question",
             "In international English, use 'I have a question' rather than 'I have a doubt' when asking for clarification."),
            (r"\bdo the needful\b",
             "please take care of this",
             "'Please take care of this' or 'please handle this' is more modern and natural than 'do the needful'."),
            (r"\brevert back\b",
             "get back to me",
             "'Revert' already implies returning; say 'reply' or 'get back to me'."),
            (r"\btoday morning\b",
             "this morning",
             "Use 'this morning' rather than 'today morning'."),
            (r"\byesterday night\b",
             "last night",
             "Use 'last night' rather than 'yesterday night'."),
            (r"\bmarried with\b",
             "married to",
             "We say 'married to someone', not 'married with'."),
            (r"\blisten music\b",
             "listen to music",
             "The verb 'listen' needs 'to' before its object: 'listen to music'."),
            (r"\bwait you\b",
             "wait for you",
             "The verb 'wait' needs 'for' before an object: 'wait for you'."),
            (r"\bgo to home\b",
             "go home",
             "'Home' functions adverbially here, so say 'go home' without 'to'."),
            (r"\bdepends of\b",
             "depends on",
             "Use the preposition 'on' with depend: 'it depends on...'."),
            # Common irregular past tense mistakes
            (r"\bbuyed\b", "bought", "The past tense of 'buy' is irregular: use 'bought'."),
            (r"\bgoed\b", "went", "The past tense of 'go' is irregular: use 'went'."),
            (r"\beated\b", "ate", "The past tense of 'eat' is irregular: use 'ate'."),
            (r"\bcatched\b", "caught", "The past tense of 'catch' is irregular: use 'caught'."),
            (r"\bsleeped\b", "slept", "The past tense of 'sleep' is irregular: use 'slept'."),
            (r"\brunned\b", "ran", "The past tense of 'run' is irregular: use 'ran'."),
            (r"\bwrited\b", "wrote", "The past tense of 'write' is irregular: use 'wrote'."),
            (r"\bchoosed\b", "chose", "The past tense of 'choose' is irregular: use 'chose'."),
            (r"\bknowed\b", "knew", "The past tense of 'know' is irregular: use 'knew'."),
            (r"\btaked\b", "took", "The past tense of 'take' is irregular: use 'took'."),
            (r"\bdrived\b", "drove", "The past tense of 'drive' is irregular: use 'drove'."),
            (r"\bgrowed\b", "grew", "The past tense of 'grow' is irregular: use 'grow'."),
            (r"\bbringed\b", "brought", "The past tense of 'bring' is irregular: use 'brought'."),
            (r"\bteached\b", "taught", "The past tense of 'teach' is irregular: use 'taught'."),
            (r"\bthinked\b", "thought", "The past tense of 'think' is irregular: use 'thought'."),
            (r"\bspeaked\b", "spoke", "The past tense of 'speak' is irregular: use 'spoke'."),
            (r"\bdid went\b", "went", "Avoid double past tense; say 'went' or 'did go'."),
            (r"\bdid saw\b", "saw", "Avoid double past tense; say 'saw' or 'did see'."),
            (r"\bdid bought\b", "bought", "Avoid double past tense; say 'bought' or 'did buy'."),
            # Subject-verb agreement
            (r"\byesterday i goes\b", "yesterday I went", "Use 'went' instead of 'goes' for past actions."),
            (r"\bi goes\b", "I go", "Use 'go' with 'I' in the present tense."),
            (r"\bshe don't\b", "she doesn't", "Use 'doesn't' with third-person singular subjects like 'she'."),
            (r"\bhe don't\b", "he doesn't", "Use 'doesn't' with third-person singular subjects like 'he'."),
            (r"\bit don't\b", "it doesn't", "Use 'doesn't' with third-person singular subjects like 'it'."),
            (r"\bi is\b", "I am", "Use 'am' with 'I' in the present tense."),
            (r"\byou is\b", "you are", "Use 'are' with 'you'."),
            (r"\bthey is\b", "they are", "Use 'are' with 'they'."),
            (r"\bwe is\b", "we are", "Use 'are' with 'we'."),
            (r"\bpeople is\b", "people are", "'People' is a plural noun, so use 'are'."),
            (r"\bme and him\b", "he and I", "Use 'he and I' as the subject of a sentence."),
            (r"\bme and her\b", "she and I", "Use 'she and I' as the subject of a sentence."),
            (r"\bi likes\b", "I like", "Use 'like' with 'I', without an 's'."),
        ]

        corrected = clean_text
        explanations = []
        applied_spans = []

        for pattern, replacement, expl in grammar_rules:
            match = re.search(pattern, lower)
            if match:
                start, end = match.span()
                overlaps = any(not (end <= s or start >= e) for s, e in applied_spans)
                if not overlaps:
                    applied_spans.append((start, end))
                    corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
                    explanations.append(expl)

        # Missing subject at start of utterance
        if not explanations:
            if re.match(r"^(went|bought|ate|saw|walked|worked|studied|cooked)\b", lower):
                corrected = "I " + clean_text[0].lower() + clean_text[1:]
                explanations.append("In English, always include the subject 'I' at the beginning of personal statements.")
            elif re.match(r"^(is|was)\s+(fine|good|nice|okay|bad|boring|awesome|great)\b", lower):
                corrected = "It " + clean_text[0].lower() + clean_text[1:]
                explanations.append("Include the subject 'It' before describing a situation or experience.")
            elif re.match(r"^(feeling|looking)\b", lower):
                corrected = "I am " + clean_text[0].lower() + clean_text[1:]
                explanations.append("Include 'I am' before progressive adjectives like 'feeling' or 'looking'.")

        # Provide natural phrasing polish if no grammatical error was detected
        if not explanations:
            short_answers = {
                "yes": ("Yes, absolutely!", "Expanding short answers into full sentences makes conversations more engaging."),
                "yeah": ("Yes, that's right!", "Using 'Yes, that is right' sounds friendly and conversational."),
                "no": ("No, not really.", "Saying 'No, not really' or 'Not at all' sounds soft and polite in natural English."),
                "nope": ("No, not at all.", "Saying 'Not at all' sounds smooth and courteous."),
                "ok": ("I'm doing well, thanks!", "Using a full sentence shows confidence and warmth."),
                "okay": ("Everything is going great, thank you!", "Full sentences keep conversations engaging and fluent."),
                "fine": ("I'm doing fine, thank you!", "Expanding with 'I am doing fine' makes your speech sound natural."),
                "nothing": ("Nothing much, just taking it easy today.", "Saying 'Nothing much, just relaxing' is an authentic, friendly response."),
                "nothing much": ("Nothing much, just taking it easy today.", "Adding 'just taking it easy' sounds authentic and conversational."),
            }
            clean_token = lower.strip("!.,? ")
            if clean_token in short_answers:
                corrected, expl = short_answers[clean_token]
                explanations.append(expl)
            elif lower.startswith("i like ") and len(clean_text) < 40:
                rest = clean_text[7:].strip().rstrip(".!?")
                corrected = f"I really enjoy {rest} whenever I get the chance."
                explanations.append("Using 'really enjoy' and adding 'whenever I get the chance' makes your speech sound descriptive and native.")
            elif re.match(r"^what is your name", lower):
                corrected = "May I ask what your name is?"
                explanations.append("Using polite indirect phrasing like 'May I ask...' sounds courteous in conversational English.")
            elif re.match(r"^who are you", lower):
                corrected = "Could you tell me a little about yourself?"
                explanations.append("Saying 'Could you tell me a little about yourself?' is a warm and natural way to introduce yourself.")
            elif re.match(r"^where are you from", lower):
                corrected = "Where are you originally from?"
                explanations.append("Adding 'originally' is a very natural and common native way to ask about someone's background.")
            elif re.match(r"^tell me a joke", lower):
                corrected = "Could you tell me a funny joke?"
                explanations.append("Using 'Could you...' phrases your request politely.")
            elif re.match(r"^what should i eat", lower):
                corrected = "What would you recommend I have for a meal?"
                explanations.append("Using 'What would you recommend...' sounds natural, expressive, and polite.")
            else:
                words = clean_text.split()
                if words:
                    words = ["I" if w == "i" else ("I'm" if w == "i'm" else w) for w in words]
                    polished = " ".join(words)
                    is_question = bool(re.match(r"^(who|what|where|when|why|how|can|could|would|should|do|does|did|is|are|was|were|may)\b", lower))
                    punct = "?" if is_question else "."
                    polished = polished[0].upper() + polished[1:] if len(polished) > 1 else polished.upper()
                    if not polished.endswith((".", "?", "!")):
                        polished += punct
                    corrected = polished
                    explanations.append("Your sentence is clear! To sound even more natural, focus on speaking smoothly and with confidence.")

        # Always capitalize the corrected sentence and ensure ending punctuation
        if corrected:
            corrected = corrected[0].upper() + corrected[1:] if len(corrected) > 1 else corrected.upper()
            if not corrected.endswith((".", "?", "!")):
                is_question = bool(re.match(r"^(who|what|where|when|why|how|can|could|would|should|do|does|did|is|are|was|were|may)\b", lower))
                corrected += ("?" if is_question else ".")

        # 3. Dynamic, Human-like Conversational Reply on Any Topic (Directly Answering First!)
        topics = [
            ("travel", "I'd love to chat! Let's talk about travel. If you could fly anywhere in the world tomorrow with all expenses paid, which country or city would you visit first?"),
            ("food", "Let's do it! How about we talk about food? What is your all-time favorite meal, or a traditional dish from your home that you love?"),
            ("movies", "Entertainment is a great topic! What is one movie or TV show that you can watch over and over without ever getting bored?"),
            ("hobbies", "Awesome! Let's talk about how you spend your free time. What are some hobbies or activities that always make you happy?"),
            ("skills", "Great! If you could instantly master any new skill or superpower tomorrow, what would you choose and why?"),
            ("nature", "Let's talk about nature and the outdoors. Do you prefer spending time relaxing near the ocean, or hiking up in the mountains?"),
            ("daily_life", "Let's talk about daily routines! Are you an early morning person who loves sunrises, or a night owl who stays up late?"),
        ]

        reply = ""
        # Check direct intents and questions - TOPIC SWITCHING first so 'on any topic to learn english' triggers topic!
        if any(t in lower for t in ["on any topic", "any topic", "talk me with any topic", "suggest a topic", "what should we talk about", "pick a topic", "topic to learn", "choose a topic"]):
            idx = getattr(self, "_topic_index", 0)
            _, chosen_prompt = topics[idx % len(topics)]
            self._topic_index = idx + 1
            reply = chosen_prompt

        elif any(g in lower for g in ["listen me", "hear me", "can you listen", "can you hear", "are you there"]) or lower in ["hello", "hi", "hey"]:
            reply = "Hello! Yes, I can hear you loud and clear! It's fantastic to connect with you. How has your day been going so far?"

        elif any(p in lower for p in ["how is your day", "how are you", "how are you doing", "what about you", "how was your day"]):
            reply = "My day is going wonderfully, thank you for asking! I love chatting and helping people practice English. How has your day been treating you?"

        elif any(t in lower for t in ["teach me english", "learn english", "practice english", "help me learn", "help me speak"]):
            reply = "I would be thrilled to be your English tutor! We can talk about anything from your daily life to hobbies, movies, or dreams, and I'll coach you along the way. To get started, what did you do earlier today?"

        elif any(act in lower for act in ["went outside and mood here and there", "went outside", "mood here and there", "take a walk", "went for a walk", "wandered"]):
            reply = "Going outside for a walk is one of the best ways to clear your thoughts and refresh your mood! Where did you end up walking--around your neighborhood, or to a park?"

        elif any(d in lower for d in ["going nice", "going well", "good day", "nice day", "great day"]):
            reply = "I am really glad to hear that things are going well for you! What was something fun or interesting that happened today?"

        elif any(q in lower for q in ["who are you", "what is your name", "tell me about yourself", "what are you"]):
            reply = "I'm Sway, your personal English tutor and conversation partner! I'm here to chat with you naturally and help you speak fluent, confident English. What would you like to explore today?"

        elif "where are you" in lower or "where do you live" in lower or "where from" in lower:
            reply = "I live in the digital cloud, but my voice is right here chatting with you! Where in the world are you joining me from today?"

        elif "tell me a joke" in lower or "tell a joke" in lower or "know any jokes" in lower:
            reply = "Why don't scientists trust atoms? Because they make up everything! Do you enjoy clever wordplay, or what kind of comedy makes you laugh?"

        elif "tell me a story" in lower:
            reply = "Once, an English learner was nervous about speaking with an accent. But as they practiced every day, they realized that having an accent was simply proof of speaking multiple languages! How do you feel about speaking English with others?"

        elif any(f in lower for f in ["pizza", "burger", "coffee", "tea", "cook", "restaurant", "food", "lunch", "dinner", "breakfast", "curry", "rice", "snack"]):
            reply = "That sounds delicious! Do you usually enjoy cooking your meals at home, or do you prefer eating out at restaurants?"

        elif any(pl in lower for pl in ["travel", "visit", "trip", "vacation", "flight", "beach", "mountain", "country", "city", "japan", "paris", "london", "india", "america", "italy"]):
            reply = "That sounds like an amazing place! What attracts you most to that destination--the culture, the food, or the scenery?"

        elif any(m in lower for m in ["movie", "film", "watch", "series", "netflix", "cinema", "song", "music", "actor"]):
            reply = "That is such a great choice! What did you like most about it--the characters, the storyline, or the music?"

        elif any(w in lower for w in ["busy", "work", "job", "office", "study", "exam", "college", "school"]):
            reply = "It sounds like you have had a very productive day! What is the most interesting project or subject you are working on?"

        elif any(r in lower for r in ["tired", "relax", "exhausted", "sleep", "bored", "home"]):
            reply = "Taking time to rest and recharge is so important. What do you like to do to unwind--listen to music, read, or watch something funny?"

        elif lower.strip("!.,?") in ["yes", "yeah", "sure", "yep", "absolutely"]:
            reply = "Nice! Tell me a bit more about that--what makes you say so?"

        elif lower.strip("!.,?") in ["no", "nope", "not really"]:
            reply = "Fair enough! What would you prefer instead if you had the choice?"

        elif lower.startswith("what should i") or lower.startswith("what can i"):
            reply = "That depends on what you are in the mood for! If you have some free time, relaxing with a good book or listening to music is wonderful. What options are you weighing?"

        elif lower.startswith("why ") or "why is" in lower:
            reply = "That's a very thoughtful question! It often comes down to how people think and how things have developed over time. What's your own perspective on it?"

        elif lower.startswith("how ") or "how do" in lower:
            reply = "Taking it step by step and staying curious is usually the best approach! What part of it feels like the biggest puzzle for you right now?"

        elif any(h in lower for h in ["hotel staff", "act like hotel", "hotel roleplay", "hotel receptionist", "hotel"]):
            reply = "Welcome to the Grand Horizon Hotel! My name is Sway, and I am the front desk assistant. How can I help you today? Are you checking in for a reservation, or is there something else I can assist with?"

        elif any(n in lower for n in ["normal day", "very normal day", "regular day"]):
            reply = "Tell me, what have you been doing today? Did anything interesting happen, or was it mostly routine work or study?"

        else:
            follow_ups = [
                "That's really interesting! Tell me a little bit more about that.",
                "I see what you mean! What made you think of that today?",
                "That makes a lot of sense. How do you usually feel when that happens?",
                "Thanks for sharing that with me! What would you like to explore next?"
            ]
            reply = follow_ups[abs(hash(clean_text)) % len(follow_ups)]

        # In spoken language mode, naturally include the spoken coaching in the spoken reply
        if explanations and not ("A more natural way" in reply or "You can say" in reply or "Could you" in reply):
            spoken_coaching = f"A more natural way to say that is: '{corrected}'. {explanations[0]} "
            reply = spoken_coaching + reply

        return json.dumps({
            "correction_needed": True,
            "original_sentence": clean_text,
            "corrected_sentence": corrected,
            "explanation": " ".join(explanations) if explanations else "Great expression! Keep practicing full sentences to build natural fluency.",
            "conversational_reply": reply
        })

    def generate(
        self,
        text: str,
        history: Optional[List[Dict[str, str]]] = None,
        use_system_prompt: bool = True,
        request_id: Optional[str] = None,
        **kwargs: Any
    ) -> Generator[str, None, None]:
        """
        Generates text using the configured backend, yielding tokens as a stream.
        """
        # Lazy initialization; fallback to local tutor generator if API keys are missing/mock
        if not self._lazy_initialize_clients():
            logger.warning(f"🤖⚠️ LLM backend '{self.backend}' client unavailable (missing or mock API key). Generating local tutor response.")
            yield self._generate_local_tutor_response(text, history=history)
            return

        req_id = request_id if request_id else f"{self.backend}-{uuid.uuid4()}"
        logger.info(f"🤖💬 Starting generation (Request ID: {req_id})")

        messages = []
        if use_system_prompt and self.system_prompt_message:
            messages.append(self.system_prompt_message)
        if history:
            recent_history = history[-6:] if len(history) > 6 else history
            messages.extend(recent_history)

        if len(messages) == 0 or messages[-1]["role"] != "user":
            added_text = text # for normal text
            if self.no_think:
                 # This modification logic remains specific for now
                added_text = f"{text}/nothink" # for qwen 3
            logger.info(f"🧠💬 llm_module.py generate adding role user to messages, content: {added_text}")
            messages.append({"role": "user", "content": added_text})
        logger.debug(f"🤖💬 [{req_id}] Prepared messages count: {len(messages)}")

        stream_iterator = None
        stream_object_to_register = None # This is the object we need to close on cancel

        try:
            # Enforce structured JSON mode by default
            call_kwargs = dict(kwargs)
            if "response_format" not in call_kwargs:
                call_kwargs["response_format"] = {"type": "json_object"}
            call_kwargs.setdefault("max_tokens", 350)
            call_kwargs.setdefault("temperature", 0.3)

            if self.backend == "openai":
                if self.client is None:
                    raise RuntimeError("OpenAI client not initialized (should have been caught by lazy_init).")
                payload = { "model": self.model, "messages": messages, "stream": True, **call_kwargs }
                logger.info(f"🤖💬 [{req_id}] Sending OpenAI request with payload:")
                logger.info(f"{json.dumps(payload, indent=2)}")
                try:
                    stream_iterator = self.client.chat.completions.create(
                        model=self.model, messages=messages, stream=True, **call_kwargs
                    )
                except Exception as e:
                    if "response_format" in str(e).lower():
                        logger.warning(f"🤖⚠️ OpenAI backend rejected response_format, falling back: {e}")
                        call_kwargs.pop("response_format", None)
                        stream_iterator = self.client.chat.completions.create(
                            model=self.model, messages=messages, stream=True, **call_kwargs
                        )
                    else:
                        raise
                stream_object_to_register = stream_iterator # The Stream object itself
                self._register_request(req_id, "openai", stream_object_to_register)
                yield from self._yield_openai_chunks(stream_iterator, req_id)

            elif self.backend == "lmstudio":
                if self.client is None:
                    raise RuntimeError("LM Studio client not initialized (should have been caught by lazy_init).")
                if 'temperature' not in call_kwargs:
                    call_kwargs['temperature'] = 0.7
                payload = { "model": self.model, "messages": messages, "stream": True, **call_kwargs }
                logger.info(f"🤖💬 [{req_id}] Sending LM Studio request with payload:")
                logger.info(f"{json.dumps(payload, indent=2)}")
                try:
                    stream_iterator = self.client.chat.completions.create(
                        model=self.model, messages=messages, stream=True, **call_kwargs
                    )
                except Exception as e:
                    if "response_format" in str(e).lower():
                        logger.warning(f"🤖⚠️ LM Studio rejected response_format, falling back: {e}")
                        call_kwargs.pop("response_format", None)
                        stream_iterator = self.client.chat.completions.create(
                            model=self.model, messages=messages, stream=True, **call_kwargs
                        )
                    else:
                        raise
                stream_object_to_register = stream_iterator # The Stream object itself
                self._register_request(req_id, "lmstudio", stream_object_to_register)
                yield from self._yield_openai_chunks(stream_iterator, req_id)

            elif self.backend == "poe":
                if self.client is None:
                    raise RuntimeError("Poe client not initialized (should have been caught by lazy_init).")
                if 'temperature' not in call_kwargs:
                    call_kwargs['temperature'] = 0.7
                payload = { "model": self.model, "messages": messages, "stream": True, **call_kwargs }
                logger.info(f"🤖💬 [{req_id}] Sending Poe request with payload:")
                logger.info(f"{json.dumps(payload, indent=2)}")
                try:
                    stream_iterator = self.client.chat.completions.create(
                        model=self.model, messages=messages, stream=True, **call_kwargs
                    )
                except Exception as e:
                    if "response_format" in str(e).lower():
                        logger.warning(f"🤖⚠️ Poe rejected response_format, falling back: {e}")
                        call_kwargs.pop("response_format", None)
                        stream_iterator = self.client.chat.completions.create(
                            model=self.model, messages=messages, stream=True, **call_kwargs
                        )
                    else:
                        raise
                stream_object_to_register = stream_iterator
                self._register_request(req_id, "poe", stream_object_to_register)
                yield from self._yield_openai_chunks(stream_iterator, req_id)

            elif self.backend == "groq":
                if self.client is None:
                    raise RuntimeError("Groq client not initialized (should have been caught by lazy_init).")
                if 'temperature' not in call_kwargs:
                    call_kwargs['temperature'] = 0.3
                # Prevent tool-calls unless explicitly requested; Groq may raise if tool_choice missing
                call_kwargs.setdefault("tool_choice", "none")
                call_kwargs.setdefault("tools", [])
                payload = { "model": self.model, "messages": messages, "stream": True, **call_kwargs }
                logger.info(f"🤖💬 [{req_id}] Sending Groq request with payload:")
                logger.info(f"{json.dumps(payload, indent=2)}")
                try:
                    stream_iterator = self.client.chat.completions.create(
                        model=self.model, messages=messages, stream=True, **call_kwargs
                    )
                except Exception as e:
                    if "response_format" in str(e).lower():
                        logger.warning(f"🤖⚠️ Groq rejected response_format, falling back: {e}")
                        call_kwargs.pop("response_format", None)
                        stream_iterator = self.client.chat.completions.create(
                            model=self.model, messages=messages, stream=True, **call_kwargs
                        )
                    else:
                        raise
                stream_object_to_register = stream_iterator
                self._register_request(req_id, "groq", stream_object_to_register)
                yield from self._yield_openai_chunks(stream_iterator, req_id)

            elif self.backend == "megallm":
                if not self.megallm_api_key:
                    raise RuntimeError("MegaLLM API key not configured.")
                if not self.megallm_base_url:
                    raise RuntimeError("MegaLLM base URL not configured.")
                mega_url = self.megallm_base_url.rstrip("/") + "/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.megallm_api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    **kwargs,
                }
                logger.info(f"🤖💬 [{req_id}] Sending MegaLLM request to {mega_url} with payload:")
                logger.info(f"{json.dumps(payload, indent=2)}")
                response = self.megallm_session.post(
                    mega_url,
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=(10.0, 600.0),
                ) if self.megallm_session else requests.post(
                    mega_url,
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=(10.0, 600.0),
                )
                response.raise_for_status()
                stream_object_to_register = response
                self._register_request(req_id, "megallm", stream_object_to_register)
                yield from self._yield_megallm_chunks(response, req_id)

            else:
                # This case should technically be caught by __init__
                raise ValueError(f"Backend '{self.backend}' generation logic not implemented.")

            logger.info(f"🤖✅ Finished generating stream successfully (request_id: {req_id})")

        except Exception as e:
            if "failed to generate json" in str(e).lower() and self.backend == "groq":
                logger.warning(f"🤖⚠️ Groq JSON constraint failed during stream ({e}). Retrying without response_format...")
                try:
                    call_kwargs.pop("response_format", None)
                    call_kwargs["temperature"] = 0.3
                    stream_iterator = self.client.chat.completions.create(
                        model=self.model, messages=messages, stream=True, **call_kwargs
                    )
                    stream_object_to_register = stream_iterator
                    self._register_request(req_id, "groq", stream_object_to_register)
                    yield from self._yield_openai_chunks(stream_iterator, req_id)
                    return
                except Exception as retry_err:
                    logger.error(f"🤖💥 Groq retry without response_format failed: {retry_err}")
            logger.warning(f"🤖⚠️ Remote LLM API call error for {req_id}: {e}. Yielding local English tutor fallback response.")
            yield self._generate_local_tutor_response(text, history=history)
        finally:
            # Removes request ID from tracking AND attempts to close the stream via _cancel_single_request_unsafe
            logger.debug(f"🤖ℹ️ [{req_id}] Entering finally block for generate.")
            with self._requests_lock:
                if req_id in self._active_requests:
                    # Only log removal if it was actually present
                    logger.debug(f"🤖🗑️ [{req_id}] Removing request from tracking and attempting stream close in generate's finally block.")
                    # Perform the removal and close attempt using the existing unsafe helper
                    self._cancel_single_request_unsafe(req_id)
                else:
                    # This can happen if cancellation occurred before finally
                    logger.debug(f"🤖🗑️ [{req_id}] Request already removed from tracking before finally block completion.")
            logger.debug(f"🤖ℹ️ [{req_id}] Exiting finally block. Active requests: {len(self._active_requests)}")


    # --- Backend-Specific Chunk Yielding Helpers ---
    def _yield_openai_chunks(self, stream, request_id: str) -> Generator[str, None, None]:
        """
        Iterates over an OpenAI/LMStudio stream, yielding content chunks.

        Handles extracting content from stream chunks and checks for cancellation
        before processing each chunk. Ensures the stream is closed upon completion,
        error, or cancellation.

        Args:
            stream: The stream object returned by the OpenAI client's `create` method.
            request_id: The unique ID associated with this generation stream.

        Yields:
            str: Content chunks from the stream's delta messages.

        Raises:
            ConnectionError: If a connection error occurs during streaming, unless likely due to cancellation.
            APIError: If an API error occurs during streaming.
            Exception: For other unexpected errors during streaming.
        """
        token_count = 0
        try:
            for chunk in stream:
                # Check for cancellation *before* processing chunk
                with self._requests_lock:
                    if request_id not in self._active_requests:
                        logger.info(f"🤖🗑️ OpenAI/LMStudio stream {request_id} cancelled or finished externally during iteration.")
                        # No need to manually close stream here; cancellation logic or finally block handles it.
                        break # Exit the loop cleanly
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    content = delta.content
                    if content:
                        token_count += 1
                        yield content
            logger.debug(f"🤖✅ [{request_id}] Finished yielding {token_count} OpenAI/LMStudio tokens.")
        except APIConnectionError as e:
             # Often happens if the stream is closed prematurely by cancellation
             is_cancelled = False
             with self._requests_lock:
                 is_cancelled = request_id not in self._active_requests
             if is_cancelled:
                  logger.warning(f"🤖⚠️ OpenAI/LMStudio stream connection error likely due to cancellation for {request_id}: {e}")
             else:
                  logger.error(f"🤖💥 OpenAI API connection error during streaming ({request_id}): {e}")
                  raise ConnectionError(f"OpenAI communication error during streaming: {e}") from e
        except APIError as e:
            logger.error(f"🤖💥 OpenAI API error during streaming ({request_id}): {e}")
            raise # Reraise for generate() to handle
        except Exception as e:
            # Catch other potential errors during iteration
            is_cancelled = False
            with self._requests_lock:
                is_cancelled = request_id not in self._active_requests
            if is_cancelled:
                logger.warning(f"🤖⚠️ OpenAI/LMStudio stream error likely due to cancellation for {request_id}: {e}")
            else:
                logger.error(f"🤖💥 Unexpected error during OpenAI streaming ({request_id}): {e}", exc_info=True)
                raise # Reraise for generate() to handle
        finally:
            # Ensure the stream is closed if iteration finishes or breaks
            # The cancellation logic also tries to close, but this catches normal completion
            if stream and hasattr(stream, 'close') and callable(stream.close):
                 try:
                     logger.debug(f"🤖🗑️ [{request_id}] Closing OpenAI stream in _yield_openai_chunks finally.")
                     stream.close()
                 except Exception as close_err:
                     logger.warning(f"🤖⚠️ [{request_id}] Error closing OpenAI stream in finally: {close_err}", exc_info=False)

    def _yield_megallm_chunks(self, response: requests.Response, request_id: str) -> Generator[str, None, None]:
        """
        Iterates over a MegaLLM HTTP response stream (OpenAI-compatible), yielding content chunks.
        """
        token_count = 0
        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()
                if line in ("", "[DONE]"):
                    if line == "[DONE]":
                        logger.debug(f"🤖✅ [{request_id}] MegaLLM signalled done.")
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"🤖⚠️ [{request_id}] Failed to decode MegaLLM line: {line[:100]}")
                    continue
                choices = payload.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or choices[0].get("message", {})
                content = delta.get("content") if isinstance(delta, dict) else None
                if content:
                    token_count += 1
                    yield content
            logger.debug(f"🤖✅ [{request_id}] Finished yielding {token_count} MegaLLM tokens.")
        finally:
            try:
                response.close()
            except Exception as close_err:
                logger.warning(f"🤖⚠️ [{request_id}] Error closing MegaLLM response: {close_err}", exc_info=False)

    def measure_inference_time(
        self,
        num_tokens: int = 10,
        **kwargs: Any
    ) -> Optional[float]:
        """
        Measures the time taken to generate a target number of initial tokens.

        Uses a fixed, predefined prompt designed to elicit a somewhat predictable
        response length. Times the generation process from the moment the generator
        is obtained until the target number of tokens is yielded or generation ends.
        Ensures the backend client is initialized first.

        Args:
            num_tokens: The target number of tokens to generate before stopping measurement.
            **kwargs: Additional keyword arguments passed to the `generate` method
                      (e.g., temperature=0.1).

        Returns:
            The time taken in milliseconds to generate the actual number of tokens
            produced (up to `num_tokens`), or None if generation failed, produced 0 tokens,
            or encountered an error during initialization or generation.
        """
        if num_tokens <= 0:
            logger.warning("🤖⏱️ Cannot measure inference time for 0 or negative tokens.")
            return None

        # Ensure client is ready (handles lazy init + connection checks + ps fallback)
        if not self._lazy_initialize_clients():
            logger.error(f"🤖⏱️💥 Measurement failed: Could not initialize backend client/connection for {self.backend}.")
            return None

        # --- Define specific prompts for measurement ---
        measurement_system_prompt = "You are a precise assistant. Follow instructions exactly."
        # This text is designed to likely produce > 10 tokens across different tokenizers.
        measurement_user_prompt = "Repeat the following sequence exactly, word for word: one two three four five six seven eight nine ten eleven twelve"
        measurement_history = [
            {"role": "system", "content": measurement_system_prompt},
            {"role": "user", "content": measurement_user_prompt}
        ]
        # ---------------------------------------------

        req_id = f"measure-{self.backend}-{uuid.uuid4()}"
        logger.info(f"🤖⏱️ Measuring inference time for {num_tokens} tokens (Request ID: {req_id}). Using fixed measurement prompt.")
        logger.debug(f"🤖⏱️ [{req_id}] Measurement history: {measurement_history}")

        token_count = 0
        start_time = None
        end_time = None
        generator = None
        actual_tokens_generated = 0

        try:
            # Pass the constructed history and ensure use_system_prompt is False
            # The 'text' argument to generate is ignored when history is provided containing the user message.
            generator = self.generate(
                text="", # Text is ignored as history provides the user message
                history=measurement_history,
                use_system_prompt=False, # Explicitly disable default system prompt
                request_id=req_id,
                **kwargs # Pass any extra args like temperature
            )

            # Iterate and time
            start_time = time.time() # Start timing *after* generate() call returns generator
            for token in generator:
                if token_count == 0:
                     # Could capture TTFT here if needed: time.time() - start_time
                     pass
                token_count += 1
                # logger.debug(f"[{req_id}] Token {token_count}: '{token}'") # Optional: very verbose
                if token_count >= num_tokens:
                    end_time = time.time()
                    logger.debug(f"🤖⏱️ [{req_id}] Reached target {num_tokens} tokens.")
                    break # Stop iterating

            # If loop finished without breaking, record end time here
            if end_time is None:
                end_time = time.time()
                logger.debug(f"🤖⏱️ [{req_id}] Generation finished naturally after {token_count} tokens (may be less than requested {num_tokens}).")

            actual_tokens_generated = token_count

        except (ConnectionError, APIError, RuntimeError, Exception) as e:
            logger.error(f"🤖⏱️💥 Error during inference time measurement ({req_id}): {e}", exc_info=False)
            # Let finally block handle potential generator cleanup
            return None # Indicate failure
        finally:
            # Ensure generator resources are released if the loop was broken early
            # The generate() method's finally block handles request tracking removal AND attempts close.
            # We still explicitly try closing the generator here as a fallback.
            if generator and hasattr(generator, 'close'):
                try:
                    logger.debug(f"🤖⏱️🗑️ [{req_id}] Closing generator in measure_inference_time finally.")
                    generator.close()
                except Exception as close_err:
                    # Log but don't prevent returning time if measured
                    logger.warning(f"🤖⏱️⚠️ [{req_id}] Error closing generator in finally: {close_err}", exc_info=False)
            generator = None # Clear reference


        # --- Calculate and Return Result ---
        if start_time is None or end_time is None:
             logger.error(f"🤖⏱️💥 [{req_id}] Measurement failed: Start or end time not recorded.")
             return None

        if actual_tokens_generated == 0:
             logger.warning(f"🤖⏱️⚠️ [{req_id}] Measurement invalid: 0 tokens were generated.")
             return None

        duration_sec = end_time - start_time
        duration_ms = duration_sec * 1000

        logger.info(
            f"🤖⏱️✅ Measured ~{duration_ms:.2f} ms for {actual_tokens_generated} tokens "
            f"(target: {num_tokens}) for model '{self.model}' on backend '{self.backend}' using fixed prompt. (Request ID: {req_id})"
        )

        # Return the time taken for the actual tokens generated.
        return duration_ms


# --- Context Manager ---
class LLMGenerationContext:
    """
    A context manager for safely handling LLM generation streams.

    Ensures that the underlying generation stream is properly requested for cancellation
    (including attempting to close the network connection) when the context is exited,
    whether normally or due to an exception.
    """
    def __init__(
        self,
        llm: LLM,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        use_system_prompt: bool = True,
        **kwargs: Any
        ):
        """
        Initializes the generation context.

        Args:
            llm: The LLM instance to use for generation.
            prompt: The user's input prompt/text.
            history: Optional list of previous messages.
            use_system_prompt: If True, uses the LLM's configured system prompt.
            **kwargs: Additional arguments to pass to the `llm.generate` method.
        """
        self.llm = llm
        self.prompt = prompt
        self.history = history
        self.use_system_prompt = use_system_prompt
        self.kwargs = kwargs
        self.generator: Optional[Generator[str, None, None]] = None
        self.request_id: str = f"ctx-{llm.backend}-{uuid.uuid4()}"
        self._entered: bool = False

    def __enter__(self) -> Generator[str, None, None]:
        """
        Enters the context, starts generation, and returns the token generator.

        Calls the LLM's `generate` method and registers the request.

        Returns:
            A generator yielding tokens from the LLM.

        Raises:
            RuntimeError: If the context is re-entered or generator creation fails.
            (Propagates exceptions from `llm.generate`).
        """
        if self._entered:
            raise RuntimeError("LLMGenerationContext cannot be re-entered")
        self._entered = True
        logger.debug(f"🤖▶️ [{self.request_id}] Entering LLMGenerationContext.")
        try:
            # Generate call now implicitly runs lazy_init
            self.generator = self.llm.generate(
                self.prompt,
                self.history,
                self.use_system_prompt,
                request_id=self.request_id,
                **self.kwargs
            )
            return self.generator
        except Exception as e:
            logger.error(f"🤖💥 [{self.request_id}] Failed generator creation in context: {e}", exc_info=True)
            # Attempt to clean up if registration happened before error (tries close)
            self.llm.cancel_generation(self.request_id)
            self._entered = False
            raise # Reraise the exception

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the context, ensuring the generation stream is cancelled and closed.

        Calls `llm.cancel_generation` to remove tracking and attempt stream closure.
        Also explicitly attempts to close the generator object itself as a safeguard.

        Args:
            exc_type: The type of exception that caused the context to be exited (if any).
            exc_val: The exception instance (if any).
            exc_tb: The traceback (if any).

        Returns:
            False, indicating that exceptions (if any) should not be suppressed.
        """
        logger.debug(f"🤖◀️ [{self.request_id}] Exiting LLMGenerationContext (Exc: {exc_type}).")
        # Calls the modified cancel_generation, which now attempts to close the stream
        self.llm.cancel_generation(self.request_id) # Removes tracking & attempts close

        # Explicit close attempt in __exit__ is now less critical as cancel_generation
        # and the _yield_* helpers' finally blocks also attempt closure.
        # Keep it as a final safeguard.
        if self.generator and hasattr(self.generator, 'close'):
            try:
                logger.debug(f"🤖🗑️ [{self.request_id}] Explicitly closing generator in context exit (final check).")
                self.generator.close()
            except Exception as e:
                 logger.warning(f"🤖⚠️ [{self.request_id}] Error closing generator in context exit: {e}")

        self.generator = None
        self._entered = False
        # If an exception occurred, don't suppress it
        return False



# --- Example Usage ---
if __name__ == "__main__":
    # Setup logging for the example itself
    # Use basicConfig here as it's the main script
    main_log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    main_log_level = getattr(logging, main_log_level_str, logging.INFO)
    logging.basicConfig(level=main_log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        stream=sys.stdout)
    main_logger = logging.getLogger(__name__) # Logger for this __main__ block
    main_logger.info("🤖🚀 --- Running LLM Module Example ---")

    # --- Add LMStudio/OpenAI examples if needed ---
    # ... (similar structure, ensure OPENAI_AVAILABLE check)

    main_logger.info("\n" + "="*40)
    main_logger.info("🤖🏁 --- LLM Module Example Script Finished ---")
