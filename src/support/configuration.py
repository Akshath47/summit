from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig

class Configuration(BaseModel):
    """
    Minimal config shared across the graph.
    Only stores per-user values that must persist across nodes.
    """

    user_id: str = Field(..., description="Unique identifier for the user")
    timezone: str = Field("UTC", description="User's timezone for scheduling events")

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> "Configuration":
        """
        Extracts Configuration from a RunnableConfig object.
        Falls back to defaults if values are missing.
        """
        configurable = config.get("configurable", {})
        user_id_raw = configurable.get("user_id")
        user_id = user_id_raw if user_id_raw else "default-user"
        timezone = configurable.get("timezone", "UTC")
        
        return cls(user_id=user_id, timezone=timezone)
