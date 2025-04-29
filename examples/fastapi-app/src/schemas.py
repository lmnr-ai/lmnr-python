from pydantic import BaseModel
from enum import Enum


class TicketCategory(str, Enum):
    REFUND = "REFUND"
    BUG = "BUG"
    QUESTION = "QUESTION"
    OTHER = "OTHER"


class Ticket(BaseModel):
    title: str
    description: str
    customer_email: str


class TicketClassification(BaseModel):
    category: TicketCategory
    reasoning: str
