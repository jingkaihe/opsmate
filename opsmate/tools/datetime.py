from datetime import datetime, timedelta
from pydantic import BaseModel, Field, model_validator
import pytz
from typing import ClassVar, Annotated
from opsmate.dino import dtool, dino
import structlog

logger = structlog.get_logger(__name__)


class DatetimeRange(BaseModel):
    start: str = Field(
        description="The start time of the query in %Y-%m-%dT%H:%M:%SZ format",
    )
    end: str = Field(
        description="The end time of the query in %Y-%m-%dT%H:%M:%SZ format",
    )

    _FMT: ClassVar[str] = "%Y-%m-%dT%H:%M:%SZ"

    @model_validator(mode="after")
    def validate_start_end(cls, v):
        try:
            datetime.strptime(v.start, cls._FMT)
        except ValueError:
            raise ValueError(f"Invalid start date format: {v.start}")

        try:
            datetime.strptime(v.end, cls._FMT)
        except ValueError:
            raise ValueError(f"Invalid end date format: {v.end}")

        return v


@dtool
async def current_time() -> str:
    """
    Get the current time in %Y-%m-%dT%H:%M:%SZ format
    """
    return datetime.now(pytz.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dtool
@dino(
    model="gpt-4o-mini",
    response_model=DatetimeRange,
    tools=[current_time],
)
async def datetime_extraction(
    text: Annotated[str, "The text to extract the datetime range from"]
) -> DatetimeRange:
    """
    You are tasked to extract the datetime range from the text

    The `current_time` tool must be called to understand the current time
    """
    logger.info("datetime_extraction", text=text)
    return text
