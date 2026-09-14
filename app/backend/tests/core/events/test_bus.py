from uuid import uuid4

import pytest
from apps.api.app.core.events.bus import EventBus
from apps.api.app.domain.events.base import DomainEvent


class SampleEvent(DomainEvent):
    payload: str

class AnotherEvent(DomainEvent):
    count: int

@pytest.fixture
def event_bus():
    return EventBus()

@pytest.mark.asyncio
async def test_event_bus_subscription_and_publish(event_bus):
    received_events = []
    
    async def handler(event: DomainEvent):
        received_events.append(event)
        
    event_bus.subscribe(SampleEvent, handler)
    
    event = SampleEvent(event_id=str(uuid4()), payload="test_data")
    await event_bus.publish(event)
    
    assert len(received_events) == 1
    assert received_events[0].payload == "test_data"

@pytest.mark.asyncio
async def test_event_bus_multiple_handlers(event_bus):
    counter = {"value": 0}
    
    async def handler_one(event: DomainEvent):
        counter["value"] += 1
        
    async def handler_two(event: DomainEvent):
        counter["value"] += 1
        
    event_bus.subscribe(SampleEvent, handler_one)
    event_bus.subscribe(SampleEvent, handler_two)
    
    event = SampleEvent(event_id=str(uuid4()), payload="test")
    await event_bus.publish(event)
    
    assert counter["value"] == 2

@pytest.mark.asyncio
async def test_event_bus_isolated_topics(event_bus):
    received = {"sample": 0, "another": 0}
    
    async def sample_handler(event: DomainEvent):
        received["sample"] += 1
        
    async def another_handler(event: DomainEvent):
        received["another"] += 1
        
    event_bus.subscribe(SampleEvent, sample_handler)
    event_bus.subscribe(AnotherEvent, another_handler)
    
    await event_bus.publish(SampleEvent(event_id=str(uuid4()), payload="test"))
    
    assert received["sample"] == 1
    assert received["another"] == 0

@pytest.mark.asyncio
async def test_event_bus_handler_exception_isolation(event_bus):
    """Ensure that one failing handler does not prevent others from running."""
    counter = {"value": 0}
    
    async def failing_handler(event: DomainEvent):
        raise ValueError("Intentional failure")
        
    async def successful_handler(event: DomainEvent):
        counter["value"] += 1
        
    event_bus.subscribe(SampleEvent, failing_handler)
    event_bus.subscribe(SampleEvent, successful_handler)
    
    event = SampleEvent(event_id=str(uuid4()), payload="test")
    
    # Should not raise exception to the caller
    await event_bus.publish(event)
    
    # The successful handler should still have executed
    assert counter["value"] == 1
