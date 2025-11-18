"""
interrupt_handler.py

Async InterruptHandler extension (example/demo) for LiveKit Agents.

Behavior:
- Ignores filler words (configurable + extended Hindi list) only while the agent is speaking.
- Treats filler words as valid speech when the agent is silent.
- Ignores low-confidence ASR segments while agent is speaking.
- Detects stop-keywords and invokes agent pause/stop immediately.
- Optional runtime HTTP config server (aiohttp) to update ignored words / threshold.
"""

import asyncio
import logging
import os
import re
from typing import List, Optional, Dict, Any

# optional aiohttp for runtime config (bonus)
try:
    from aiohttp import web
    _AIOHTTP_AVAILABLE = True
except Exception:
    _AIOHTTP_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("interrupt_handler")

# Default ignored words (EN + minimal HI) + ENV override support
DEFAULT_IGNORED = os.getenv("IGNORED_WORDS", "uh,umm,hmm,haan,mm,erm,uhh,uhm").split(",")

# Extended Hindi filler list (appended without replacing defaults/environment)
EXTENDED_HINDI_FILLERS = [
    "arre", "arey", "acha", "achha", "ohh", "theek",
    "sahi", "yaar", "bas"
]

DEFAULT_CONF_THRESH = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

# STOP words kept minimal
STOP_KEYWORDS = {"stop", "wait", "pause", "hold", "no"}


class InterruptHandler:
    def __init__(
        self,
        agent,
        ignored_words: Optional[List[str]] = None,
        confidence_threshold: float = DEFAULT_CONF_THRESH,
        http_config_port: Optional[int] = None,
    ):
        """
        agent: object that implements:
            - on(event_name: str, callback)
            - pause_tts() or stop_speaking()
        """

        # MUST BE FIRST — store reference to agent
        self.agent = agent

        # Base ignored list = user-provided OR defaults
        base_ignored = ignored_words or DEFAULT_IGNORED

        # Append extended Hindi fillers
        full_ignored_list = list(base_ignored) + EXTENDED_HINDI_FILLERS

        # Normalize and store all ignored words
        self.ignored_words = set(self._clean_token(w) for w in full_ignored_list)

        self.confidence_threshold = float(confidence_threshold)
        self.agent_speaking = False
        self._lock = asyncio.Lock()

        # metrics
        self.metrics = {
            "ignored_interruption": 0,
            "valid_interruption": 0,
            "low_confidence_ignored": 0,
        }

        # register event listeners
        agent.on("agent_speaking_changed", self._on_agent_speaking_changed)
        agent.on("transcription", self._on_transcription)

        # optional HTTP config server
        self._http_app = None
        self._http_runner = None
        if http_config_port is not None:
            if not _AIOHTTP_AVAILABLE:
                logger.warning("aiohttp not available; runtime config HTTP server disabled")
            else:
                asyncio.create_task(self._start_http_server(http_config_port))

    # --------------------------
    # Event handlers
    # --------------------------

    async def _on_agent_speaking_changed(self, is_speaking: bool):
        async with self._lock:
            self.agent_speaking = bool(is_speaking)
        logger.debug("agent_speaking set to %s", self.agent_speaking)

    async def _on_transcription(self, segment: Dict[str, Any]):
        text = (segment.get("text") or "").strip()
        conf = float(segment.get("confidence", 1.0))
        start_time = segment.get("start_time")
        end_time = segment.get("end_time")

        if not text:
            return

        tokens = [self._normalize_token(t) for t in re.findall(r"\w+", text.lower())]

        # 1. Low confidence noise
        if conf < self.confidence_threshold and self.agent_speaking:
            self.metrics["low_confidence_ignored"] += 1
            self.metrics["ignored_interruption"] += 1
            self._log_ignored(text, conf, start_time, end_time, reason="LOW_CONF_DURING_AGENT")
            return

        # 2. Filler-only speech
        if tokens and all(self._is_ignored_token(tok) for tok in tokens):
            if self.agent_speaking:
                self.metrics["ignored_interruption"] += 1
                self._log_ignored(text, conf, start_time, end_time, reason="FILLER_ONLY")
                return
            else:
                self.metrics["valid_interruption"] += 1
                self._log_valid(text, conf, start_time, end_time, reason="FILLER_WHEN_AGENT_SILENT")
                return

        # 3. STOP keyword detection
        if any(self._is_stop_keyword(tok) for tok in tokens):
            self.metrics["valid_interruption"] += 1
            self._log_valid(text, conf, start_time, end_time, reason="STOP_KEYWORD")
            await self._invoke_agent_stop()
            return

        # 4. Normal valid speech
        self.metrics["valid_interruption"] += 1
        self._log_valid(text, conf, start_time, end_time, reason="NORMAL_SPEECH")

    # --------------------------
    # Agent stop handling
    # --------------------------
    async def _invoke_agent_stop(self):
        try:
            if hasattr(self.agent, "pause_tts"):
                coro = self.agent.pause_tts()
                if asyncio.iscoroutine(coro):
                    await coro
            elif hasattr(self.agent, "stop_speaking"):
                coro = self.agent.stop_speaking()
                if asyncio.iscoroutine(coro):
                    await coro
            else:
                logger.warning("Agent lacks pause_tts/stop_speaking method.")
        except Exception as e:
            logger.exception("Error during agent stop: %s", e)

    # --------------------------
    # Helpers
    # --------------------------
    def _is_ignored_token(self, token: str) -> bool:
        return token in self.ignored_words

    def _is_stop_keyword(self, token: str) -> bool:
        return token in STOP_KEYWORDS

    def _clean_token(self, token: str) -> str:
        return re.sub(r"[^a-z]", "", (token or "").lower())

    def _normalize_token(self, token: str) -> str:
        """
        Normalize:
        - Lowercase alphabet only
        - Collapse repeated letters: "haaannn" → "han"
        """
        t = self._clean_token(token)
        if not t:
            return t
        out = [t[0]]
        for ch in t[1:]:
            if ch != out[-1]:
                out.append(ch)
        return "".join(out)

    # --------------------------
    # Logging
    # --------------------------
    def _log_ignored(self, text, conf, start_time, end_time, reason=""):
        logger.info("[IGNORED] reason=%s conf=%.3f text=%s start=%s end=%s",
                    reason, conf, text, start_time, end_time)

    def _log_valid(self, text, conf, start_time, end_time, reason=""):
        logger.info("[VALID] reason=%s conf=%.3f text=%s start=%s end=%s",
                    reason, conf, text, start_time, end_time)

    # --------------------------
    # Optional HTTP runtime config server
    # --------------------------
    async def _start_http_server(self, port: int):
        if not _AIOHTTP_AVAILABLE:
            return

        self._http_app = web.Application()
        self._http_app.add_routes([
            web.get('/config', self._http_get_config),
            web.post('/config', self._http_post_config),
        ])

        self._http_runner = web.AppRunner(self._http_app)
        await self._http_runner.setup()

        site = web.TCPSite(self._http_runner, '0.0.0.0', port)
        await site.start()
        logger.info("InterruptHandler HTTP config server running on port %d", port)

    async def _http_get_config(self, request):
        return web.json_response({
            "ignored_words": sorted(list(self.ignored_words)),
            "confidence_threshold": self.confidence_threshold,
            "metrics": self.metrics,
        })

    async def _http_post_config(self, request):
        try:
            payload = await request.json()
        except Exception:
            return web.Response(status=400, text="invalid json")

        updated = False

        if 'ignored_words' in payload and isinstance(payload['ignored_words'], list):
            new_words = [self._clean_token(w) for w in payload['ignored_words']]
            self.ignored_words = set(new_words + EXTENDED_HINDI_FILLERS)
            updated = True

        if 'confidence_threshold' in payload:
            try:
                self.confidence_threshold = float(payload['confidence_threshold'])
                updated = True
            except Exception:
                return web.Response(status=400, text="invalid confidence_threshold")

        if updated:
            return web.json_response({
                "status": "ok",
                "ignored_words": sorted(list(self.ignored_words)),
                "confidence_threshold": self.confidence_threshold,
            })

        return web.json_response({"status": "no_changes"})
