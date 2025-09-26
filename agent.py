import logging
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_agent_graph():
    """
    Función que encapsula la creación y compilación del grafo del agente.
    """
    load_dotenv()
    llm = ChatOpenAI(model_name=os.getenv("OPENAI_MODEL"), temperature=0)
    file_manager = FileManager()
    rag_system_prompt = file_manager.load_md_file("prompt/rag.md")

    # --- Retriever ---
    retriever_factory = MarkdownParentRetrieverSetup(
        file_path="data/PDF-GenAI-Challenge_2.md",
        collection_name="rag_langgraph_prod_v2"
    )
    retriever = retriever_factory.get_retriever()

    # --- Definición de Estado y Nodos (sin cambios) ---
    class GraphState(TypedDict):
        input: str
        chat_history: List[BaseMessage]
        question: str
        documents: List[str]

    def rewrite_question(state: GraphState):
        logging.info("--- 🧠 REESCRIBIENDO PREGUNTA ---")
        # ... (código del nodo sin cambios)
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
        logging.info("--- 📚 RECUPERANDO DOCUMENTOS ---")
        documents = retriever.invoke(state["question"])
        return {"documents": documents}

    def generate_answer(state: GraphState):
        logging.info("--- 🗣️ GENERANDO RESPUESTA ---")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", rag_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        rag_chain = prompt | llm
        context_str = "\n".join([doc.page_content for doc in state["documents"]])
        generation = rag_chain.invoke({
            "context": context_str,
            "input": state["input"],
            "chat_history": state["chat_history"]
        })
        # IMPORTANTE: Ya no actualizamos el historial aquí, LangGraph lo hará por nosotros.
        return {"chat_history": [generation]} # Devolvemos solo el nuevo mensaje

    # --- Construcción y Compilación del Grafo ---
    workflow = StateGraph(GraphState)
    workflow.add_node("rewriter", rewrite_question)
    workflow.add_node("retriever", retrieve_documents)
    workflow.add_node("generator", generate_answer)

    workflow.set_entry_point("rewriter")
    workflow.add_edge("rewriter", "retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", END)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    print("✅ Grafo compilado y listo para ser usado.")
    return app

# Creamos la instancia del agente una sola vez cuando el módulo se carga
agent_app = create_agent_graph()

# El bloque if __name__ == "__main__": es útil para probar este archivo de forma aislada
if __name__ == "__main__":
    print("Probando el agente de forma aislada...")
    config = {"configurable": {"thread_id": "test-thread-1"}}
    # Primera pregunta
    for _ in agent_app.stream({"input": "Explica el concepto de 'dropout'", "chat_history": []}, config):
        pass
    final_state = agent_app.get_state(config)
    print(f"🤖 AI: {final_state.values['chat_history'][-1].content}")
    # Segunda pregunta
    for _ in agent_app.stream({"input": "¿y por qué es importante?"}, config):
        pass
    final_state = agent_app.get_state(config)
    print(f"🤖 AI: {final_state.values['chat_history'][-1].content}")