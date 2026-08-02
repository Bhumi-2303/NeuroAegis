from datetime import datetime, timezone
from pydantic import BaseModel, Field

class DomainEvent(BaseModel):
    """
    Base class for all internal domain events within NeuroAegis.
    Events represent state changes or triggers across bounded contexts.
    """
    event_id: str = Field(..., description="Unique identifier for the event occurrence")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def event_name(self) -> str:
        """Return the class name to be used as the topic/routing key."""
        return self.__class__.__name__
