from pydantic import BaseModel, Field
from typing import List, Optional


class DateRange(BaseModel): 
    start_date: Optional[str] = Field(..., description="start_date")
    end_date: Optional[str] = Field(..., description="end_date")


class Author(BaseModel):
    name: str = Field(..., description="name")


class SearchParams(BaseModel):
    question: str = Field(..., description="question")
    date_range: DateRange = Field(..., description="date_range")
    authors: List[Author] = Field(..., description="authors")
