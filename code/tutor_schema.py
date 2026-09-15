# tutor_schema.py
import re
import json
import logging
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

class EnglishTutorResponse(BaseModel):
    """
    Strict response schema for English Tutor output.
    """
    correction_needed: bool = Field(
        default=False,
        description="True if the user made a grammar, vocabulary, or phrasing mistake, False otherwise."
    )
    original_sentence: str = Field(
        default="",
        description="The user's sentence containing the grammatical or phrasing error, or empty string if no correction."
    )
    corrected_sentence: str = Field(
        default="",
        description="The corrected, natural English version of the sentence, or empty string if no correction."
    )
    explanation: str = Field(
        default="",
        description="A concise, encouraging explanation of the correction, or empty string if no correction."
    )
    conversational_reply: str = Field(
        default="",
        description="A natural, supportive spoken response to continue the conversation."
    )


def parse_and_validate_tutor_response(raw_text: str, user_text: str = "") -> EnglishTutorResponse:
    """
    Safely parses and validates the LLM output into an EnglishTutorResponse.
    
    Handles:
    - Markdown code fences (```json ... ``` or ``` ... ```)
    - Stray text around the JSON object
    - CamelCase vs snake_case field variations
    - Incomplete or malformed JSON with a graceful fallback to ensure the chat pipeline never breaks.
    """
    if not raw_text or not isinstance(raw_text, str):
        return EnglishTutorResponse(
            correction_needed=False,
            original_sentence=user_text,
            corrected_sentence="",
            explanation="",
            conversational_reply="I'm here! What would you like to talk about today?"
        )

    cleaned = raw_text.strip()

    # 1. Strip markdown code fences if present
    if "```" in cleaned:
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if fence_match:
            cleaned = fence_match.group(1).strip()

    # 2. Extract outermost JSON object if surrounding commentary exists
    if not (cleaned.startswith("{") and cleaned.endswith("}")):
        json_match = re.search(r"(\{[\s\S]*\})", cleaned)
        if json_match:
            cleaned = json_match.group(1).strip()

    # 3. Parse JSON
    data: Optional[Dict[str, Any]] = None
    try:
        data = json.loads(cleaned)
    except Exception as parse_err:
        logger.warning(f"Failed to parse LLM output as JSON ({parse_err}). Attempting regex extraction. Raw: {raw_text[:100]}...")
        # Attempt regex extraction of schema fields if JSON had missing commas or trailing quotes
        reply_m = re.search(r'"conversational_reply"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_text)
        corr_m = re.search(r'"corrected_sentence"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_text)
        exp_m = re.search(r'"explanation"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_text)
        needed_m = re.search(r'"correction_needed"\s*:\s*(true|false)', raw_text, re.IGNORECASE)

        if reply_m:
            recovered_reply = reply_m.group(1).encode().decode('unicode_escape', errors='replace')
            recovered_corr = corr_m.group(1).encode().decode('unicode_escape', errors='replace') if corr_m else ""
            recovered_exp = exp_m.group(1).encode().decode('unicode_escape', errors='replace') if exp_m else ""
            recovered_needed = needed_m.group(1).lower() == "true" if needed_m else bool(recovered_corr)
            logger.info(f"Successfully recovered tutor response via regex! Reply: {recovered_reply[:60]}...")
            return EnglishTutorResponse(
                correction_needed=recovered_needed,
                original_sentence=user_text,
                corrected_sentence=recovered_corr,
                explanation=recovered_exp,
                conversational_reply=recovered_reply
            )

        # Fallback if not even conversational_reply is extracted: strip any JSON tokens so TTS doesn't speak braces
        cleaned_fallback = re.sub(r'[{}\[\]"]|"(?:correction_needed|original_sentence|corrected_sentence|explanation|conversational_reply)":', '', raw_text).strip()
        if not cleaned_fallback or len(cleaned_fallback) < 4:
            cleaned_fallback = "That sounds good! Tell me a little bit more about that."

        return EnglishTutorResponse(
            correction_needed=False,
            original_sentence=user_text,
            corrected_sentence="",
            explanation="",
            conversational_reply=cleaned_fallback
        )

    if not isinstance(data, dict):
        return EnglishTutorResponse(
            correction_needed=False,
            original_sentence=user_text,
            corrected_sentence="",
            explanation="",
            conversational_reply=str(data)
        )

    # 4. Normalize common camelCase aliases
    alias_mapping = {
        "correctionNeeded": "correction_needed",
        "originalSentence": "original_sentence",
        "correctedSentence": "corrected_sentence",
        "conversationalReply": "conversational_reply",
        "reply": "conversational_reply",
        "response": "conversational_reply",
    }
    for camel, snake in alias_mapping.items():
        if camel in data and snake not in data:
            data[snake] = data[camel]

    # 5. Validate with Pydantic
    try:
        return EnglishTutorResponse.model_validate(data)
    except ValidationError as val_err:
        logger.warning(f"Pydantic validation warning ({val_err}). Normalizing fields.")
        return EnglishTutorResponse(
            correction_needed=bool(data.get("correction_needed", False)),
            original_sentence=str(data.get("original_sentence") or ""),
            corrected_sentence=str(data.get("corrected_sentence") or ""),
            explanation=str(data.get("explanation") or ""),
            conversational_reply=str(data.get("conversational_reply") or raw_text.strip())
        )
