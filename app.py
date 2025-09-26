from typing import Dict, List, Any
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage, AIMessage

# Asegúrate de que en agent.py la variable compilada se llame agent_app
from agent import agent_app

# --- Inicialización de FastAPI ---
app = FastAPI()

# Diccionario para almacenar el historial de cada conversación en memoria
conversations: Dict[str, Dict] = {}

## ----------------- RUTAS DE LA API ----------------- ##

@app.post("/conversation")
async def create_conversation():
    """
    Inicia una nueva conversación.
    """
    conversation_id = str(uuid.uuid4())
    conversations[conversation_id] = {
        "history": [],
        "remaining": 10
    }
    print(f"Nueva conversación creada con ID: {conversation_id}")
    return {"conversation_id": conversation_id}


@app.post("/message")
async def send_message(request: Request):
    """Procesa un mensaje de usuario y devuelve la respuesta con sus fuentes."""
    data = await request.json()
    conversation_id = data.get("conversation_id")
    user_input = data.get("user_input")

    if not conversation_id or conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="ID de conversación no válido.")
    if not user_input:
        raise HTTPException(status_code=400, detail="La entrada del usuario no puede estar vacía.")

    conv_data = conversations[conversation_id]

    if conv_data["remaining"] <= 0:
        return {
            "answer": "Lo siento, has alcanzado el límite de mensajes.",
            "remaining_messages": 0,
            "sources": []
        }

    try:
        history = conv_data["history"]
        history.append(HumanMessage(content=user_input))

        ## <-- MEJORA 1: Simplificación de la entrada
        # El grafo solo necesita 'messages' para su punto de entrada.
        # LangGraph se encarga de poblar los otros campos del estado internamente.
        inputs = {"messages": history}
        
        print(f"Invocando agente para la conversación {conversation_id}...")
        
        # Invocamos el agente. No es necesario inicializar todos los campos del estado.
        final_state = agent_app.invoke(inputs, {"recursion_limit": 100})
        
        final_response_message = final_state["messages"][-1]
        final_response_content = final_response_message.content if isinstance(final_response_message, AIMessage) else ""

        ## <-- MEJORA 2: Manejo seguro y formato de las fuentes
        # Usamos .get() para evitar un KeyError si la ruta conversacional no produce chunks.
        source_documents = final_state.get("retrieved_chunks", [])
        
        # Formateamos las fuentes para que el frontend reciba un JSON limpio.
        sources_for_frontend = [
            {"content": doc.page_content, "metadata": doc.metadata} for doc in source_documents
        ]

        if final_response_content:
            history.append(AIMessage(content=final_response_content))
        
        conv_data["remaining"] -= 1
        
        print(f"Respuesta generada para {conversation_id}. Mensajes restantes: {conv_data['remaining']}")
        
        return {
            "answer": final_response_content,
            "remaining_messages": conv_data["remaining"],
            "sources": sources_for_frontend ## <-- MEJORA 3: Enviamos las fuentes ya formateadas
        }
    except Exception as e:
        print(f"💥 Ocurrió un error en la conversación {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Ocurrió un error interno al procesar tu solicitud.")


## ----------------- RUTA PARA SERVIR EL FRONTEND ----------------- ##

@app.get("/")
async def serve_frontend():
    """Sirve el archivo principal del frontend (index.html)."""
    return FileResponse('frontend/index.html')


if __name__ == "__main__":
    import uvicorn
    print(">> Starting Uvicorn server on http://0.0.0.0:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)