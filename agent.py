import os
from dotenv import load_dotenv
from typing import List, TypedDict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from utils.retriever_factory import MarkdownParentRetrieverSetup
from utils.file_manager import FileManager

# --- 1. Configuración Inicial (igual que antes) ---
load_dotenv()
openai_model = os.getenv("OPENAI_MODEL")
llm = ChatOpenAI(model_name=openai_model ,temperature=0)
MD_FILE_PATH = "data/PDF-GenAI-Challenge_2.md"
file_manager = FileManager()
rag_system_prompt = file_manager.load_md_file("prompt/rag.md")

# Retriever
# Creas una instancia y obtienes el retriever con una sola llamada
retriever_factory = MarkdownParentRetrieverSetup(
    file_path="data/PDF-GenAI-Challenge_2.md",
    collection_name="rag_langgraph_prod"
)
retriever = retriever_factory.get_retriever()

# --- 2. Definir el Estado del Grafo ---
# El estado es un diccionario que contiene toda la información
# que se pasa entre los nodos del grafo.
class GraphState(TypedDict):
    input: str # La pregunta más reciente del usuario
    chat_history: List[BaseMessage] # El historial de la conversación
    question: str # La pregunta reescrita y autónoma
    documents: List[str] # Los documentos recuperados

# --- 3. Definir los Nodos del Grafo ---

def rewrite_question(state: GraphState):
    """Reescribe la pregunta del usuario para que sea autónoma, usando el historial."""
    print("--- 🧠 REESCRIBIENDO PREGUNTA ---")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Dada una conversación y una última pregunta del usuario, reformula la última pregunta para que sea una pregunta autónoma."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    rewriter_chain = prompt | llm
    rephrased_question = rewriter_chain.invoke(state)
    return {"question": rephrased_question.content}

def retrieve_documents(state: GraphState):
    """Recupera documentos relevantes usando la pregunta reescrita."""
    print("--- 📚 RECUPERANDO DOCUMENTOS ---")
    documents = retriever.invoke(state["question"])
    return {"documents": documents}

def generate_answer(state: GraphState):
    """Genera una respuesta usando la pregunta y los documentos recuperados."""
    print("--- 🗣️ GENERANDO RESPUESTA ---")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", rag_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    rag_chain = prompt | llm
    
    # Formateamos los documentos para pasarlos al prompt
    context_str = "\n".join([doc.page_content for doc in state["documents"]])
    
    # Generamos la respuesta
    generation = rag_chain.invoke({
        "context": context_str,
        "input": state["input"],
        "chat_history": state["chat_history"]
    })
    
    # Actualizamos el historial para la siguiente ronda
    new_history = state["chat_history"] + [
        HumanMessage(content=state["input"]),
        generation
    ]
    
    return {"chat_history": new_history}


# --- 4. Construir el Grafo ---
print("🏗️  Construyendo el grafo...")
workflow = StateGraph(GraphState)

# Añadir los nodos al grafo
workflow.add_node("rewriter", rewrite_question)
workflow.add_node("retriever", retrieve_documents)
workflow.add_node("generator", generate_answer)

# Definir el flujo de ejecución (las aristas)
workflow.set_entry_point("rewriter")
workflow.add_edge("rewriter", "retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", END) # El grafo termina después de generar la respuesta

# Compilar el grafo en una aplicación ejecutable
# Usamos un `MemorySaver` para que gestione el historial automáticamente por nosotros
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
print("✅ Grafo compilado y listo.")

# --- 5. Ejecutar el Grafo ---
# Usamos un `thread_id` para identificar y mantener la conversación.
config = {"configurable": {"thread_id": "mi-conversacion-1"}}

# Primera pregunta
first_query = "Explica el concepto de 'dropout'"
print(f"\n👤 Usuario: {first_query}")
for event in app.stream({"input": first_query, "chat_history": []}, config=config):
    # El stream nos permite ver los resultados de cada nodo a medida que se ejecutan
    pass # La respuesta final estará en el historial actualizado

# Obtenemos la última respuesta del historial
final_history = app.get_state(config).values["chat_history"]
print(f"🤖 AI: {final_history[-1].content}")


# Segunda pregunta (repregunta)
second_query = "¿Y por qué es importante?"
print(f"\n👤 Usuario: {second_query}")
for event in app.stream({"input": second_query}, config=config): # Ya no pasamos el historial, LangGraph lo gestiona
    pass

final_history = app.get_state(config).values["chat_history"]
print(f"🤖 AI: {final_history[-1].content}")