import logging
import os
import operator
from dotenv import load_dotenv
from typing import Annotated, List, TypedDict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
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

    # --- DEFINICIÓN DEL ESTADO DEL GRAFO ---
    class GraphState(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
        question: str
        retrieved_chunks: List[Document]

    def rewrite_question(state: GraphState):
        """
        Reescribe la pregunta del usuario para que sea autónoma, usando el historial.
        Si es el primer mensaje de la conversación, lo usa directamente.
        """
        logging.info("--- 🧠 REESCRIBIENDO PREGUNTA ---")
        
        # Condición: ¿Es el primer mensaje de la conversación?
        # Si la longitud de 'messages' es 1, significa que solo contiene la primera pregunta del usuario.
        if len(state["messages"]) == 1:
            user_question = state["messages"][0].content
            print(f"    Primer mensaje, usando directamente: '{user_question}'")
            return {"question": user_question}
        
        # Si hay más de un mensaje, significa que es una repregunta y necesita contexto.
        user_question = state["messages"][-1].content
        chat_history = state["messages"][:-1]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Dada la siguiente conversación y una pregunta, reformula la pregunta para que sea una pregunta autónoma que pueda ser entendida sin el historial del chat. NO respondas la pregunta, solo reformúlala."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "Pregunta del usuario: {question}"),
            ]
        )
        
        rewriter_chain = prompt | llm
        
        response = rewriter_chain.invoke({
            "chat_history": chat_history,
            "question": user_question
        })
        
        print(f"    Pregunta original: '{user_question}'")
        print(f"    Pregunta reescrita: '{response.content}'")

        return {"question": response.content}

    def retrieve_documents(state: GraphState):
        logging.info("--- 📚 RECUPERANDO DOCUMENTOS ---")
        documents = retriever.invoke(state["question"])
        return {"retrieved_chunks": documents}

    def generate_answer(state: GraphState):
        logging.info("--- 🗣️ GENERANDO RESPUESTA ---")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", rag_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{rewritten_question}"),
            ]
        )
        
        rag_chain = prompt | llm
        
        context_str = "\n".join([doc.page_content for doc in state["retrieved_chunks"]])
        
        response = rag_chain.invoke({
            "context": context_str,
            "chat_history": state["messages"][:-1],
            "rewritten_question": state["question"]
        })
        # IMPORTANTE: Ya no actualizamos el historial aquí, LangGraph lo hará por nosotros.
        return {"messages": [AIMessage(content=response.content)]}

    # --- Construcción y Compilación del Grafo ---
    workflow = StateGraph(GraphState)
    workflow.add_node("rewriter", rewrite_question)
    workflow.add_node("retriever", retrieve_documents)
    workflow.add_node("generator", generate_answer)

    workflow.set_entry_point("rewriter")
    workflow.add_edge("rewriter", "retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", END)
    
    app = workflow.compile()
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