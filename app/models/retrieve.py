from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class DateRange(BaseModel):
    start_date: Optional[str] = Field(..., description="start_date")
    end_date: Optional[str] = Field(..., description="end_date")


class NamedEntity(BaseModel):
    name: str = Field(..., description="name")


class SearchParams(BaseModel):
    question: str = Field(..., description="question")
    date_range: DateRange = Field(..., description="date_range")
    authors: List[NamedEntity] = Field(default_factory=list, description="authors")
    speakers: List[NamedEntity] = Field(default_factory=list, description="speakers")
    guests: List[NamedEntity] = Field(default_factory=list, description="guests")
    content_types: List[Literal["article", "transcript", "script"]] = Field(
        default_factory=list,
        description="content_types",
    )
    program: Optional[str] = Field(default=None, description="program")
