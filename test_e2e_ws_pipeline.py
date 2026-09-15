#!/usr/bin/env python3
"""
test_e2e_ws_pipeline.py
Automated End-to-End WebSocket and Pipeline Integration Test for SwaySpeak English Tutor.

Tests:
1. Scenario A: "Happy Path" (Grammatical Errors)
   - Input: "Yesterday I goes to the store and buyed milk."
   - Verifies: correction_needed == True, fields extracted, TTS receives ONLY conversational_reply.
2. Scenario B: "Negative Path" (Perfect Grammar)
   - Input: "I went to the store yesterday to buy some milk."
   - Verifies: correction_needed == False, TTS receives ONLY conversational_reply.
3. Scenario C: Resiliency & Fallback (Malformed LLM Output)
   - Input: "Wow, great sentence! Oh wait, here is some markdown: [garbled text]"
   - Verifies: No server crash, fallback sets correction_needed == False, conversational_reply preserved.
"""

import sys
import os
import json
import time
import asyncio
from pathlib import Path
from unittest.mock import MagicMock

# Ensure code directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
CODE_DIR = BASE_DIR / "code"
sys.path.insert(0, str(CODE_DIR))

# Ensure stdout supports UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tutor_schema import EnglishTutorResponse, parse_and_validate_tutor_response
import server


# --- Mock Transcriber & Audio Input Processor for isolated, deterministic testing ---
class MockTranscriber:
    def __init__(self):
        self.potential_sentence_end = None
        self.on_tts_allowed_to_synthesize = None
        self.potential_full_transcription_callback = None
        self.potential_full_transcription_abort_callback = None
        self.full_transcription_callback = None
        self.before_final_sentence = None


class MockAudioInputProcessor:
    def __init__(self, language="en"):
        self.transcriber = MockTranscriber()
        self.realtime_callback = None
        self.recording_start_callback = None
        self.silence_active_callback = None
        self.interrupted = False

    async def process_chunk_queue(self, audio_chunks):
        # Keeps running as background task during connection
        while True:
            await asyncio.sleep(0.01)

    def abort_generation(self):
        pass

    def set_input_sample_rate(self, sr):
        pass

    def shutdown(self):
        pass

# Patch server.AudioInputProcessor to use our mock during app lifespan
server.AudioInputProcessor = MockAudioInputProcessor


def run_scenario_a():
    """Scenario A: The 'Happy Path' (Grammatical Errors)"""
    print("\n" + "="*70)
    print("RUNNING SCENARIO A: The 'Happy Path' (Grammatical Errors)")
    print("="*70)

    user_text = "Yesterday I goes to the store and buyed milk."
    mock_llm_json = {
        "correction_needed": True,
        "original_sentence": "Yesterday I goes to the store and buyed milk.",
        "corrected_sentence": "Yesterday I went to the store and bought milk.",
        "explanation": "Use 'went' instead of 'goes' and 'bought' instead of 'buyed' for the past tense.",
        "conversational_reply": "Did you pick up whole milk or skim milk while you were there?"
    }
    raw_llm_output = json.dumps(mock_llm_json)

    # 1. Test Schema Parsing
    parsed: EnglishTutorResponse = parse_and_validate_tutor_response(raw_llm_output, user_text=user_text)
    assert parsed.correction_needed is True, "Assertion Failed: correction_needed must be True"
    assert parsed.original_sentence == "Yesterday I goes to the store and buyed milk.", "Assertion Failed: original_sentence mismatch"
    assert parsed.corrected_sentence == "Yesterday I went to the store and bought milk.", "Assertion Failed: corrected_sentence mismatch"
    assert "went" in parsed.explanation, "Assertion Failed: explanation must mention 'went'"
    assert parsed.conversational_reply == "Did you pick up whole milk or skim milk while you were there?", "Assertion Failed: conversational_reply mismatch"
    print("[PASS] Assertion 1 & 2: EnglishTutorResponse evaluated correction_needed=True and extracted all fields.")

    # 2. Test Pipeline Manager & Audio Routing
    from speech_pipeline_manager import SpeechPipelineManager
    manager = SpeechPipelineManager(tts_engine="deepgram", llm_provider="groq")

    synthesized_texts = []
    def mock_synthesize(text, audio_chunks, stop_event, generation_string=""):
        synthesized_texts.append(text)
        if manager.audio.on_first_audio_chunk_synthesize:
            manager.audio.on_first_audio_chunk_synthesize()
        audio_chunks.put_nowait(b"\x00\x00" * 40)
        return True

    def mock_llm_generate(text, history=None, use_system_prompt=True, **kwargs):
        # Simulate token-by-token streaming of the JSON
        for chunk in [raw_llm_output[:25], raw_llm_output[25:60], raw_llm_output[60:]]:
            yield chunk

    manager.audio.synthesize = mock_synthesize
    manager.llm.generate = mock_llm_generate

    # Trigger generation
    manager.process_prepare_generation(user_text)

    # Wait for LLM and TTS workers to finish
    timeout = time.time() + 5.0
    while time.time() < timeout:
        if manager.running_generation and manager.running_generation.audio_quick_finished:
            break
        time.sleep(0.05)

    assert len(synthesized_texts) == 1, f"Expected 1 synthesis call, got {len(synthesized_texts)}"
    tts_text = synthesized_texts[0]
    print(f"       Synthesized Audio Text: \"{tts_text}\"")

    # Strict Dual-Stream Isolation Assertions:
    assert tts_text == parsed.conversational_reply, "Assertion Failed: TTS text must strictly match conversational_reply"
    assert "correction_needed" not in tts_text, "Assertion Failed: TTS must not contain JSON keys"
    assert parsed.explanation not in tts_text, "Assertion Failed: TTS must not contain grammar explanation"
    assert parsed.corrected_sentence not in tts_text, "Assertion Failed: TTS must not contain corrected sentence"
    assert "{" not in tts_text and "}" not in tts_text, "Assertion Failed: TTS must not contain raw JSON syntax"
    print("[PASS] Assertion 3: TTS routing logic strictly captures ONLY conversational_reply for audio synthesis.")

    # Cleanup manager threads
    manager.shutdown_event.set()
    print("[SCENARIO A PASSED SUCCESSFULLY]")


def run_scenario_b():
    """Scenario B: The 'Negative Path' (Perfect Grammar)"""
    print("\n" + "="*70)
    print("RUNNING SCENARIO B: The 'Negative Path' (Perfect Grammar)")
    print("="*70)

    user_text = "I went to the store yesterday to buy some milk."
    mock_llm_json = {
        "correction_needed": False,
        "original_sentence": "",
        "corrected_sentence": "",
        "explanation": "",
        "conversational_reply": "Sounds like a productive trip! What kind of milk did you choose?"
    }
    raw_llm_output = json.dumps(mock_llm_json)

    # 1. Test Schema Parsing
    parsed: EnglishTutorResponse = parse_and_validate_tutor_response(raw_llm_output, user_text=user_text)
    assert parsed.correction_needed is False, "Assertion Failed: correction_needed must be False for perfect grammar"
    assert parsed.original_sentence == "", "Assertion Failed: original_sentence should be empty"
    assert parsed.corrected_sentence == "", "Assertion Failed: corrected_sentence should be empty"
    assert parsed.explanation == "", "Assertion Failed: explanation should be empty"
    assert parsed.conversational_reply == "Sounds like a productive trip! What kind of milk did you choose?"
    print("[PASS] Assertion 1: EnglishTutorResponse evaluated correction_needed=False.")

    # 2. Test Pipeline Manager & Audio Routing
    from speech_pipeline_manager import SpeechPipelineManager
    manager = SpeechPipelineManager(tts_engine="deepgram", llm_provider="groq")

    synthesized_texts = []
    def mock_synthesize(text, audio_chunks, stop_event, generation_string=""):
        synthesized_texts.append(text)
        if manager.audio.on_first_audio_chunk_synthesize:
            manager.audio.on_first_audio_chunk_synthesize()
        audio_chunks.put_nowait(b"\x00\x00" * 40)
        return True

    def mock_llm_generate(text, history=None, use_system_prompt=True, **kwargs):
        for chunk in [raw_llm_output[:30], raw_llm_output[30:]]:
            yield chunk

    manager.audio.synthesize = mock_synthesize
    manager.llm.generate = mock_llm_generate

    manager.process_prepare_generation(user_text)

    timeout = time.time() + 5.0
    while time.time() < timeout:
        if manager.running_generation and manager.running_generation.audio_quick_finished:
            break
        time.sleep(0.05)

    assert len(synthesized_texts) == 1, f"Expected 1 synthesis call, got {len(synthesized_texts)}"
    tts_text = synthesized_texts[0]
    print(f"       Synthesized Audio Text: \"{tts_text}\"")

    assert tts_text == parsed.conversational_reply, "Assertion Failed: TTS text must match conversational_reply"
    print("[PASS] Assertion 2: TTS routing captures only conversational_reply.")

    manager.shutdown_event.set()
    print("[SCENARIO B PASSED SUCCESSFULLY]")


def run_scenario_c():
    """Scenario C: Resiliency & Fallback (Malformed LLM Output)"""
    print("\n" + "="*70)
    print("RUNNING SCENARIO C: Resiliency & Fallback (Malformed LLM Output)")
    print("="*70)

    user_text = "I likes coding."
    corrupted_raw_output = "Wow, great sentence! Oh wait, here is some markdown: [garbled text]"

    # 1. Test Schema Fallback
    try:
        parsed: EnglishTutorResponse = parse_and_validate_tutor_response(corrupted_raw_output, user_text=user_text)
        print("[PASS] Assertion 1: parse_and_validate_tutor_response safely caught the JSON parse exception.")
    except Exception as e:
        raise AssertionError(f"Function crashed instead of handling exception: {e}")

    assert parsed.correction_needed is False, "Assertion Failed: fallback correction_needed must be False"
    assert "Wow, great sentence!" in parsed.conversational_reply, "Assertion Failed: conversational_reply must retain raw text"
    print("[PASS] Assertion 2 & 3: Fallback object sets correction_needed=False and extracts text into conversational_reply.")

    # 2. Test Pipeline Manager with Malformed Output
    from speech_pipeline_manager import SpeechPipelineManager
    manager = SpeechPipelineManager(tts_engine="deepgram", llm_provider="groq")

    synthesized_texts = []
    def mock_synthesize(text, audio_chunks, stop_event, generation_string=""):
        synthesized_texts.append(text)
        if manager.audio.on_first_audio_chunk_synthesize:
            manager.audio.on_first_audio_chunk_synthesize()
        audio_chunks.put_nowait(b"\x00\x00" * 40)
        return True

    def mock_llm_generate(text, history=None, use_system_prompt=True, **kwargs):
        yield corrupted_raw_output

    manager.audio.synthesize = mock_synthesize
    manager.llm.generate = mock_llm_generate

    try:
        manager.process_prepare_generation(user_text)
        timeout = time.time() + 5.0
        while time.time() < timeout:
            if manager.running_generation and manager.running_generation.audio_quick_finished:
                break
            time.sleep(0.05)
        print("[PASS] Assertion 4: Speech Pipeline Manager did not crash on malformed LLM output.")
    except Exception as e:
        raise AssertionError(f"Pipeline crashed on malformed output: {e}")

    assert len(synthesized_texts) == 1
    assert "Wow, great sentence!" in synthesized_texts[0]
    print(f"       Synthesized Audio Text: \"{synthesized_texts[0]}\"")
    print("[PASS] Assertion 5: TTS routing safely synthesized fallback conversational_reply.")

    manager.shutdown_event.set()
    print("[SCENARIO C PASSED SUCCESSFULLY]")


def run_e2e_websocket_session():
    """End-to-End WebSocket Session Simulation with FastAPI TestClient"""
    print("\n" + "="*70)
    print("RUNNING E2E WEBSOCKET SESSION SIMULATION (Client to Server)")
    print("="*70)

    from fastapi.testclient import TestClient
    from server import app

    # Inject mock AudioInputProcessor into app.state before lifespan startup
    app.state.AudioInputProcessor = MockAudioInputProcessor()

    # Create TestClient
    with TestClient(app) as client:
        # Override SpeechPipelineManager dependencies on app.state
        manager = app.state.SpeechPipelineManager

        scenario_a_payload = {
            "correction_needed": True,
            "original_sentence": "Yesterday I goes to the store and buyed milk.",
            "corrected_sentence": "Yesterday I went to the store and bought milk.",
            "explanation": "Use 'went' instead of 'goes' and 'bought' instead of 'buyed' for the past tense.",
            "conversational_reply": "Did you pick up whole milk or skim milk while you were there?"
        }

        def mock_llm_stream(text, history=None, use_system_prompt=True, **kwargs):
            yield json.dumps(scenario_a_payload)

        synthesized = []
        def mock_tts_synth(text, audio_chunks, stop_event, generation_string=""):
            synthesized.append(text)
            if manager.audio.on_first_audio_chunk_synthesize:
                manager.audio.on_first_audio_chunk_synthesize()
            audio_chunks.put_nowait(b"\x00\x00" * 100)
            return True

        manager.llm.generate = mock_llm_stream
        manager.audio.synthesize = mock_tts_synth

        # Connect WebSocket exactly as frontend app.js does
        with client.websocket_connect("/ws") as ws:
            print("[WS] Client connected to /ws successfully.")

            # Send client clear_history
            ws.send_json({"type": "clear_history"})
            print("[WS] Sent clear_history.")
            time.sleep(0.5)

            # Get the connection-specific callbacks instance
            # In server.py, callbacks are attached to AudioInputProcessor during websocket_endpoint
            callbacks = app.state.AudioInputProcessor.transcriber

            user_sentence = "Yesterday I goes to the store and buyed milk."
            print(f"[WS] Simulating user turn end: \"{user_sentence}\"")

            # Trigger turn completion
            callbacks.before_final_sentence(b"", user_sentence)
            callbacks.full_transcription_callback(user_sentence)

            # Receive WebSocket messages from server
            received_messages = []
            final_answer_msg = None
            tts_chunks_received = 0

            timeout = time.time() + 8.0
            while time.time() < timeout:
                try:
                    data = ws.receive_json()
                    received_messages.append(data)
                    msg_type = data.get("type")
                    print(f"       [WS Ingest] Received message type: {msg_type}")

                    if msg_type == "tts_chunk":
                        tts_chunks_received += 1
                    elif msg_type == "final_assistant_answer":
                        final_answer_msg = data
                        break
                except Exception as e:
                    time.sleep(0.05)

            assert final_answer_msg is not None, "Assertion Failed: Client did not receive final_assistant_answer"
            content = final_answer_msg.get("content")
            assert isinstance(content, dict), f"Assertion Failed: content must be a structured dict, got {type(content)}"
            assert content["correction_needed"] is True, "Assertion Failed: correction_needed must be True"
            assert content["original_sentence"] == user_sentence
            assert content["corrected_sentence"] == "Yesterday I went to the store and bought milk."
            assert "went" in content["explanation"]
            assert content["conversational_reply"] == scenario_a_payload["conversational_reply"]
            assert tts_chunks_received > 0, "Assertion Failed: Client must receive at least 1 tts_chunk"

            print("[PASS] Client successfully received structured JSON final_assistant_answer and tts_chunk over WebSocket.")
            print(f"       Structured Content Received by Frontend:\n{json.dumps(content, indent=4)}")

    print("[E2E WEBSOCKET SIMULATION PASSED SUCCESSFULLY]")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("STARTING SWAYSPEAK ENGLISH TUTOR E2E INTEGRATION SUITE")
    print("="*70)

    try:
        run_scenario_a()
        run_scenario_b()
        run_scenario_c()
        run_e2e_websocket_session()

        print("\n" + "="*70)
        print("ALL TESTS PASSED WITH EXIT CODE 0!")
        print("="*70 + "\n")
        sys.exit(0)
    except AssertionError as ae:
        print(f"\n[TEST FAILURE] AssertionError: {ae}")
        sys.exit(1)
    except Exception as ex:
        import traceback
        traceback.print_exc()
        print(f"\n[TEST ERROR] Unexpected Exception: {ex}")
        sys.exit(1)
