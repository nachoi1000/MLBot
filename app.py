import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

# Importamos la instancia del agente ya compilada desde agent.py
from agent import agent_app

# --- Inicialización de FastAPI ---
app = FastAPI()

# Ya no necesitamos gestionar el historial aquí, LangGraph lo hace.
# Podemos mantener un diccionario si queremos guardar metadatos, como el límite de mensajes.
conversations_metadata: dict = {}

## ----------------- RUTAS DE LA API ----------------- ##

@app.post("/conversation")
async def create_conversation():
    """Inicia una nueva conversación y devuelve su ID."""
    conversation_id = str(uuid.uuid4())
    # Guardamos metadatos si es necesario
    conversations_metadata[conversation_id] = {"remaining": 10}
    print(f"Nueva conversación creada con ID: {conversation_id}")
    return {"conversation_id": conversation_id}


@app.post("/message")
async def send_message(request: Request):
    """Procesa un mensaje de usuario para una conversación existente."""
    data = await request.json()
    conversation_id = data.get("conversation_id")
    user_input = data.get("user_input")

    # --- Validación ---
    if not conversation_id or conversation_id not in conversations_metadata:
        raise HTTPException(status_code=404, detail="ID de conversación no válido o no encontrado.")
    if not user_input:
        raise HTTPException(status_code=400, detail="La entrada del usuario no puede estar vacía.")

    # --- Lógica del Agente ---
    try:
        # La clave es pasar el conversation_id como thread_id en la configuración.
        # LangGraph usará este ID para recuperar el historial correcto.
        config = {"configurable": {"thread_id": conversation_id}}

        # El input para el grafo es solo la nueva pregunta.
        # LangGraph cargará el chat_history automáticamente desde su memoria.
        inputs = {"input": user_input, "chat_history": []} # pasamos el history vacio al inicio

        print(f"Invocando agente para la conversación {conversation_id}...")

        # Usamos .invoke() para obtener la respuesta final directamente. Es más simple para una API.
        final_state = agent_app.invoke(inputs, config)
        
        # La respuesta es el último mensaje en el historial actualizado del estado del grafo.
        ai_response = final_state["chat_history"][-1].content
        
        print(f"Respuesta generada para {conversation_id}.")
        
        return {"answer": ai_response}

    except Exception as e:
        print(f"💥 Ocurrió un error en la conversación {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Ocurrió un error interno: {e}")


## ----------------- RUTA PARA SERVIR EL FRONTEND ----------------- ##

@app.get("/")
async def serve_frontend():
    """Sirve el archivo principal del frontend (index.html)."""
    return FileResponse('frontend/index.html')


if __name__ == "__main__":
    import uvicorn
    print(">> Starting Uvicorn server on http://0.0.0.0:8000")
    # Asegúrate de que el nombre del archivo es 'app' si el archivo se llama 'app.py'
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)