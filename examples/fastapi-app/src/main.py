from fastapi import FastAPI, HTTPException
from schemas import Ticket, TicketClassification
from dotenv import load_dotenv
from lmnr import Laminar
from llm import model_classify_ticket

load_dotenv(override=True)

Laminar.initialize()

app = FastAPI()


@app.post("/api/v1/tickets/classify", response_model=TicketClassification)
async def classify_ticket(ticket: Ticket):
    try:
        classification = model_classify_ticket(ticket)
        print(classification)
        return classification
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
