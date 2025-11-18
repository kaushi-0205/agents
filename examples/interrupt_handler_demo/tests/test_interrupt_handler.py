import asyncio
import pytest
from interrupt_handler_demo.interrupt_handler import InterruptHandler

class MockAgent:
    def __init__(self):
        self._listeners = {}

    def on(self, event_name: str, callback):
        self._listeners.setdefault(event_name, []).append(callback)

    async def emit(self, event_name: str, payload):
        for cb in self._listeners.get(event_name, []):
            if asyncio.iscoroutinefunction(cb):
                await cb(payload)
            else:
                cb(payload)

    async def pause_tts(self):
        # simulate async method
        await asyncio.sleep(0)

@pytest.mark.asyncio
async def test_filler_ignored_while_speaking():
    agent = MockAgent()
    handler = InterruptHandler(agent)
    await agent.emit('agent_speaking_changed', True)
    await agent.emit('transcription', {'text': 'uh', 'confidence': 0.95})
    assert handler.metrics['ignored_interruption'] >= 1

@pytest.mark.asyncio
async def test_low_confidence_ignored_while_speaking():
    agent = MockAgent()
    handler = InterruptHandler(agent, confidence_threshold=0.5)
    await agent.emit('agent_speaking_changed', True)
    await agent.emit('transcription', {'text': 'hello', 'confidence': 0.2})
    assert handler.metrics['low_confidence_ignored'] >= 1

@pytest.mark.asyncio
async def test_stop_keyword_triggers_pause():
    agent = MockAgent()
    called = {'paused': False}
    async def pause_wrapper():
        called['paused'] = True
    agent.pause_tts = pause_wrapper
    handler = InterruptHandler(agent)
    await agent.emit('transcription', {'text': 'please stop', 'confidence': 0.9})
    await asyncio.sleep(0.01)
    assert handler.metrics['valid_interruption'] >= 1

@pytest.mark.asyncio
async def test_filler_when_agent_silent_counts_as_valid():
    agent = MockAgent()
    handler = InterruptHandler(agent)
    await agent.emit('agent_speaking_changed', False)
    prev = handler.metrics['valid_interruption']
    await agent.emit('transcription', {'text': 'uh', 'confidence': 0.9})
    assert handler.metrics['valid_interruption'] >= prev + 1
