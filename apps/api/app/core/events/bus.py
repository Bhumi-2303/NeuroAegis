import asyncio
import logging
from typing import Callable, Dict, List, Any, Coroutine, Type
from apps.api.app.domain.events.base import DomainEvent

logger = logging.getLogger(__name__)

# Type alias for an async event handler
EventHandler = Callable[[DomainEvent], Coroutine[Any, Any, None]]

class EventBus:
    """
    Asynchronous internal publish-subscribe event bus.
    Facilitates decoupled communication between system modules via DomainEvents.
    """
    def __init__(self):
        # Maps event class names to a list of handler coroutines
        self._subscribers: Dict[str, List[EventHandler]] = {}

    def subscribe(self, event_type: Type[DomainEvent], handler: EventHandler) -> None:
        """
        Subscribe an async handler to a specific domain event type.
        
        Args:
            event_type: The class of the DomainEvent to subscribe to.
            handler: An async function that accepts the event instance.
        """
        event_name = event_type.__name__
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(handler)

    async def publish(self, event: DomainEvent) -> None:
        """
        Publish an event to all subscribed handlers concurrently.
        
        Args:
            event: The instantiated DomainEvent to dispatch.
        """
        event_name = event.event_name
        handlers = self._subscribers.get(event_name, [])
        
        if not handlers:
            return

        # Execute all handlers concurrently
        tasks = [asyncio.create_task(handler(event)) for handler in handlers]
        
        # return_exceptions=True ensures one failing handler doesn't crash the bus
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Handler {handlers[idx].__name__} failed for event {event_name}: {result}")
