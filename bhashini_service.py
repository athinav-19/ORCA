"""
bhashini_service.py - Bhashini Cloud API Integration Service
ISRO SIH Problem Statement 176: Marine Multi-Agent System (Project ORCA)

Handles:
1. Base64 Audio Decoding & Speech-to-Text (ASR) via Bhashini ULCA/Dhruva Pipeline.
2. Neural Machine Translation (NMT) between Indian Coastal Languages and English.
3. Text-to-Speech (TTS) generating Base64 encoded audio payloads for voice responses.
4. Robust exception handling and fallback resilience preventing system halts.
"""

import os
import re
import json
import base64
import requests
from typing import Optional, Dict, Union
from dotenv import load_dotenv




class BhashiniService:
    """
    Cloud API Client for Government of India / MeitY Bhashini Platform.
    Requires:
      - BHASHINI_USER_ID (Udyat Key)
      - BHASHINI_API_KEY (Inference Key)
    """

    DEFAULT_INFERENCE_URL = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
    DEFAULT_CONFIG_URL = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"

    def __init__(self, user_id: Optional[str] = None, api_key: Optional[str] = None):
        # Load environment variables
        load_dotenv()

        # Fetch keys from environment with support for standard and alias names
        self.user_id = (
            user_id
            or os.getenv("BHASHINI_USER_ID")
            or os.getenv("BHASHINI_UDYAT_KEY")
            or os.getenv("UDYAT_KEY")
        )
        self.api_key = (
            api_key
            or os.getenv("BHASHINI_API_KEY")
            or os.getenv("BHASHINI_INFERENCE_KEY")
            or os.getenv("INFERENCE_KEY")
        )

        # Raise ValueError if credentials are missing
        if not self.user_id or not self.api_key:
            raise ValueError(
                "Missing Bhashini credentials. Please define BHASHINI_USER_ID (Udyat Key) "
                "and BHASHINI_API_KEY (Inference Key) in your .env file."
            )

        # Standard Bhashini request headers
        self.headers: Dict[str, str] = {
            "userID": self.user_id,
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }

        self.inference_url: str = os.getenv("BHASHINI_INFERENCE_URL", self.DEFAULT_INFERENCE_URL)
        self.config_url: str = os.getenv("BHASHINI_CONFIG_URL", self.DEFAULT_CONFIG_URL)
        self.pipeline_id: Optional[str] = os.getenv("BHASHINI_PIPELINE_ID")

    @staticmethod
    def _clean_base64_audio(audio_data: Union[str, bytes]) -> str:
        """Strips data-URI scheme header if present and cleans whitespace."""
        if not audio_data:
            return ""
        if isinstance(audio_data, bytes):
            try:
                decoded = audio_data.decode("utf-8").strip()
                if re.match(r"^[A-Za-z0-9+/=\r\n]+$", decoded):
                    clean = decoded
                else:
                    return base64.b64encode(audio_data).decode("utf-8")
            except Exception:
                return base64.b64encode(audio_data).decode("utf-8")
        else:
            clean = str(audio_data).strip()

        if "," in clean:
            clean = clean.split(",", 1)[1]
        return clean.replace("\n", "").replace("\r", "")

    def speech_to_text(self, base64_audio: Union[str, bytes], source_lang: str = "ta") -> str:
        """
        Method 1: Speech-to-Text (ASR)
        Sends Base64 audio payload to Bhashini ASR endpoint and returns transcribed text.
        """
        if not base64_audio:
            return ""

        clean_audio = self._clean_base64_audio(base64_audio)
        if not clean_audio:
            return ""

        source_lang_code = source_lang.lower().strip()

        payload = {
            "pipelineTasks": [
                {
                    "taskType": "asr",
                    "config": {
                        "language": {
                            "sourceLanguage": source_lang_code
                        },
                        "audioFormat": "wav"
                    }
                }
            ],
            "inputData": {
                "audio": [
                    {
                        "audioContent": clean_audio
                    }
                ]
            }
        }

        if self.pipeline_id:
            payload["pipelineRequestConfig"] = {"pipelineId": self.pipeline_id}

        try:
            response = requests.post(
                self.inference_url,
                headers=self.headers,
                json=payload,
                timeout=20
            )

            if response.status_code == 200:
                resp_json = response.json()
                pipeline_resp = resp_json.get("pipelineResponse", [])
                if pipeline_resp:
                    output_list = pipeline_resp[0].get("output", [])
                    if output_list and "source" in output_list[0]:
                        transcribed = output_list[0]["source"].strip()
                        return transcribed
                print(f"[BhashiniService ASR] Unexpected response schema: {resp_json}")
            else:
                print(f"[BhashiniService ASR] HTTP {response.status_code}: {response.text[:200]}")

        except requests.exceptions.Timeout:
            print("[BhashiniService ASR Error] Request timed out after 20 seconds.")
        except requests.exceptions.RequestException as req_err:
            print(f"[BhashiniService ASR Error] Network or HTTP error: {req_err}")
        except Exception as err:
            print(f"[BhashiniService ASR Error] Unexpected exception during speech_to_text: {err}")

        # Fallback return empty string if ASR fails
        return ""

    def translate(self, text: str, source_lang: str, target_lang: str = "en") -> str:
        """
        Method 2: Neural Machine Translation (NMT)
        Translates regional text into English (for backend agents), and vice versa (for final output).
        """
        if not text or not text.strip():
            return ""

        clean_text = text.strip()
        src = source_lang.lower().strip()
        tgt = target_lang.lower().strip()

        # If source and target language are identical, return text directly
        if src == tgt:
            return clean_text

        payload = {
            "pipelineTasks": [
                {
                    "taskType": "translation",
                    "config": {
                        "language": {
                            "sourceLanguage": src,
                            "targetLanguage": tgt
                        }
                    }
                }
            ],
            "inputData": {
                "input": [
                    {
                        "source": clean_text
                    }
                ]
            }
        }

        if self.pipeline_id:
            payload["pipelineRequestConfig"] = {"pipelineId": self.pipeline_id}

        try:
            response = requests.post(
                self.inference_url,
                headers=self.headers,
                json=payload,
                timeout=20
            )

            if response.status_code == 200:
                resp_json = response.json()
                pipeline_resp = resp_json.get("pipelineResponse", [])
                if pipeline_resp:
                    output_list = pipeline_resp[0].get("output", [])
                    if output_list and "target" in output_list[0]:
                        translated_text = output_list[0]["target"].strip()
                        return translated_text
                print(f"[BhashiniService Translation] Unexpected response schema: {resp_json}")
            else:
                print(f"[BhashiniService Translation] HTTP {response.status_code}: {response.text[:200]}")

        except requests.exceptions.Timeout:
            print("[BhashiniService Translation Error] Request timed out after 20 seconds.")
        except requests.exceptions.RequestException as req_err:
            print(f"[BhashiniService Translation Error] Network or HTTP error: {req_err}")
        except Exception as err:
            print(f"[BhashiniService Translation Error] Unexpected exception during translate: {err}")

        # Fallback to returning original text if translation times out or fails
        return clean_text

    def text_to_speech(self, text: str, target_lang: str = "ta", gender: str = "female") -> Optional[str]:
        """
        Method 3: Text-to-Speech (TTS)
        Converts regional text advisory into Base64 encoded audio string.
        Fallback to returning raw text/None if TTS fails.
        """
        if not text or not text.strip():
            return None

        clean_text = text.strip()
        tgt = target_lang.lower().strip()

        payload = {
            "pipelineTasks": [
                {
                    "taskType": "tts",
                    "config": {
                        "language": {
                            "sourceLanguage": tgt
                        },
                        "gender": gender
                    }
                }
            ],
            "inputData": {
                "input": [
                    {
                        "source": clean_text
                    }
                ]
            }
        }

        if self.pipeline_id:
            payload["pipelineRequestConfig"] = {"pipelineId": self.pipeline_id}

        try:
            response = requests.post(
                self.inference_url,
                headers=self.headers,
                json=payload,
                timeout=25
            )

            if response.status_code == 200:
                resp_json = response.json()
                pipeline_resp = resp_json.get("pipelineResponse", [])
                if pipeline_resp:
                    audio_list = pipeline_resp[0].get("audio", [])
                    if audio_list and "audioContent" in audio_list[0]:
                        audio_base64 = audio_list[0]["audioContent"]
                        return audio_base64
                print(f"[BhashiniService TTS] Unexpected response schema: {resp_json}")
            else:
                print(f"[BhashiniService TTS] HTTP {response.status_code}: {response.text[:200]}")

        except requests.exceptions.Timeout:
            print("[BhashiniService TTS Error] Request timed out after 25 seconds.")
        except requests.exceptions.RequestException as req_err:
            print(f"[BhashiniService TTS Error] Network or HTTP error: {req_err}")
        except Exception as err:
            print(f"[BhashiniService TTS Error] Unexpected exception during text_to_speech: {err}")

        return None