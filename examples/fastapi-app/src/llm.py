from openai import OpenAI
import os
from dotenv import load_dotenv
from schemas import Ticket, TicketCategory, TicketClassification

load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def model_classify_ticket(ticket: Ticket) -> TicketClassification:
    response = client.beta.chat.completions.parse(
        model="gpt-4.1-nano",
        response_format=TicketClassification,
        messages=[
            {
                "role": "system",
                "content": """You are a support ticket classifier.
Classify the ticket that a user sends you""",
            },
            {
                "role": "user",
                "content": f"""Title: {ticket.title}
Description: {ticket.description}
Customer Email: {ticket.customer_email}""",
            },
        ],
    )

    return response.choices[0].message.parsed or TicketClassification(
        category=TicketCategory.OTHER,
        reasoning=response.choices[0].message.content,
    )
