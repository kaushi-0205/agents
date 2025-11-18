# main.py demo runner (offline simulation)
import asyncio
from interrupt_handler import InterruptHandler

class DemoAgent:
    """
    Minimal demo agent that supports:
      - on(event, callback)
      - emit(event, payload)
      - pause_tts() (coroutine)
    This is only for local/offline demonstration and unit testing.
    """
    def __init__(self):
        self._listeners = {}

    def on(self, event, callback):
        self._listeners.setdefault(event, []).append(callback)

    async def emit(self, event, payload):
        for cb in self._listeners.get(event, []):
            if asyncio.iscoroutinefunction(cb):
                await cb(payload)
            else:
                cb(payload)

    async def pause_tts(self):
        # Demo action when TTS is stopped by the handler
        print("\n>>> DEMO: Agent TTS STOPPED <<<\n")

async def run_demo():
    agent = DemoAgent()
    handler = InterruptHandler(agent, ignored_words=['uh','umm','hmm','haan'], confidence_threshold=0.6, http_config_port=None)

    print("\n=== DEMO START ===\n")

    # 1. Agent starts speaking
    print("Agent speaking...")
    await agent.emit("agent_speaking_changed", True)

    # ---- IGNORED TEST ----
    print("\nUser says: 'uh'")
    await agent.emit("transcription", {
        "text": "uh",
        "confidence": 0.98,
        "start_time": 0,
        "end_time": 1
    })

    # ---- VALID INTERRUPTION ----
    print("\nUser says: 'umm stop'")
    await agent.emit("transcription", {
        "text": "umm stop",
        "confidence": 0.92,
        "start_time": 2,
        "end_time": 3
    })

    # Agent stops speaking (handler will have called pause_tts)
    await agent.emit("agent_speaking_changed", False)

    # ---- FILLER WHEN AGENT SILENT ----
    print("\nUser says: 'uh' (agent silent)")
    await agent.emit("transcription", {
        "text": "uh",
        "confidence": 0.90,
        "start_time": 4,
        "end_time": 5
    })

    # ---- LOW CONFIDENCE TEST ----
    print("\nAgent speaking again...")
    await agent.emit("agent_speaking_changed", True)
    print("User says noise with low confidence...")
    await agent.emit("transcription", {
        "text": "sdfkjwe",   # nonsense
        "confidence": 0.10,
        "start_time": 6,
        "end_time": 7
    })

    print("\n=== DEMO END ===\n")

if __name__ == "__main__":
    asyncio.run(run_demo())
