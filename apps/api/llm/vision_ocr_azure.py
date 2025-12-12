"""
Azure OpenAI Vision OCR Client (ingestion-only).

Purpose:
- Some DOCX pages are embedded as images (non-selectable text).
- We extract DOCX-embedded images and OCR them during ingestion (never at runtime).

Constraints:
- Temperature fixed at 0 for maximal stability.
- Output is raw transcribed text only (no interpretation).
- Produces traceable anchors via image SHA256 + paragraph anchor.
"""

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from typing import Optional

try:
    from openai import AsyncAzureOpenAI
except ImportError:  # pragma: no cover
    AsyncAzureOpenAI = None


@dataclass(frozen=True)
class VisionOcrConfig:
    azure_endpoint: str
    azure_api_key: str
    azure_api_version: str
    vision_deployment: str
    timeout: int = 60

    @classmethod
    def from_env(cls) -> "VisionOcrConfig":
        return cls(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            azure_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            vision_deployment=os.getenv(
                "AZURE_OPENAI_VISION_DEPLOYMENT_NAME",
                os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""),
            ),
            timeout=int(os.getenv("LLM_TIMEOUT", "60")),
        )

    def is_configured(self) -> bool:
        return bool(self.azure_endpoint and self.azure_api_key and self.vision_deployment)


@dataclass(frozen=True)
class OcrResult:
    """
    OCR result for a single image.

    - image_sha256 is used as a stable evidence anchor component.
    - text_ar is the raw transcription.
    """

    image_sha256: str
    text_ar: str
    error: Optional[str] = None


class VisionOcrClient:
    """
    Minimal OCR client using Azure OpenAI Responses API.

    Note:
    - This is ingestion-only and should never be invoked at runtime /ask.
    """

    def __init__(self, config: Optional[VisionOcrConfig] = None):
        self.config = config or VisionOcrConfig.from_env()
        self._client: Optional[AsyncAzureOpenAI] = None

    async def _get_client(self) -> AsyncAzureOpenAI:
        if self._client is None:
            if AsyncAzureOpenAI is None:
                raise ImportError("openai package not installed. Install `openai`.")
            self._client = AsyncAzureOpenAI(
                azure_endpoint=self.config.azure_endpoint,
                api_key=self.config.azure_api_key,
                api_version=self.config.azure_api_version,
                timeout=self.config.timeout,
            )
        return self._client

    async def ocr_image(self, image_bytes: bytes) -> OcrResult:
        """
        OCR an image to Arabic text.

        Args:
            image_bytes: Raw image bytes (png/jpg).

        Returns:
            OcrResult: raw text with stable hash.
        """
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()

        if not self.config.is_configured():
            return OcrResult(
                image_sha256=image_sha256,
                text_ar="",
                error="Vision OCR is not configured (missing AZURE_OPENAI_* env vars).",
            )

        try:
            client = await self._get_client()
            b64 = base64.b64encode(image_bytes).decode("ascii")
            data_url = f"data:image/png;base64,{b64}"

            prompt = (
                "أنت نظام OCR فقط.\n"
                "المهمة: انسخ النص العربي كما يظهر في الصورة حرفيًا.\n"
                "قواعد صارمة:\n"
                "- لا تفسر ولا تلخص ولا تعيد الصياغة.\n"
                "- حافظ على ترتيب الأسطر والفقرات قدر الإمكان.\n"
                "- لا تضف أي كلمات غير موجودة في الصورة.\n"
                "- إذا كان هناك نص إنجليزي عناوين (مثل Acceptance) انسخه كما هو.\n"
                "- أرجع النص فقط بدون أي مقدمة.\n"
            )

            resp = await client.responses.create(
                model=self.config.vision_deployment,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    }
                ],
                temperature=0.0,
                max_output_tokens=4096,
            )

            # openai SDK convenience
            text = getattr(resp, "output_text", None)
            if isinstance(text, str) and text.strip():
                return OcrResult(image_sha256=image_sha256, text_ar=text.strip())

            # Fallback parsing
            out_text = ""
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        out_text += t
            return OcrResult(image_sha256=image_sha256, text_ar=out_text.strip())
        except Exception as e:  # pragma: no cover
            return OcrResult(image_sha256=image_sha256, text_ar="", error=str(e))


