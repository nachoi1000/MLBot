import os
import re
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()

class MarkdownParentRetrieverSetup:
    """
    Una clase para encapsular la creación y configuración de un ParentDocumentRetriever
    a partir de un archivo Markdown con estructura de encabezados.
    """
    def __init__(self, file_path: str, collection_name: str):
        """
        Inicializa la configuración.

        Args:
            file_path (str): La ruta al archivo .md.
            collection_name (str): Nombre para la colección en ChromaDB.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo no fue encontrado en la ruta: {file_path}")
            
        self.file_path = file_path
        self.collection_name = collection_name
        self.retriever = None

    def _clean_text(self, text: str) -> str:
        """Limpia caracteres extraños del texto."""
        text = text.replace('↪', '').replace('glyph[triangleright]', ' ')
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def get_retriever(self) -> ParentDocumentRetriever:
        """
        Crea, indexa y devuelve el ParentDocumentRetriever configurado.
        Si ya fue creado, lo devuelve directamente.
        """
        if self.retriever:
            print("✅ Retriever ya estaba inicializado. Devolviendo instancia existente.")
            return self.retriever

        # --- 1. Cargar y dividir en documentos PADRE ---
        print(f"📚 Cargando y procesando el archivo: {self.file_path}...")
        with open(self.file_path, "r", encoding="utf-8") as f:
            md_text = self._clean_text(f.read())

        parent_headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"), # Ajustado para incluir ### como un nivel de padre
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=parent_headers_to_split_on, strip_headers=False
        )
        parent_documents = markdown_splitter.split_text(md_text)
        print(f"📄 Documento dividido en {len(parent_documents)} documentos padre.")

        # --- 2. Definir el splitter para los documentos HIJO ---
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=100
        )

        # --- 3. Configurar almacenes y Vectorstore ---
        print("🧠 Configurando Vectorstore y Docstore...")
        vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=OpenAIEmbeddings()
        )
        docstore = InMemoryStore()

        # --- 4. Crear e indexar el Retriever ---
        print("🛠️  Creando el ParentDocumentRetriever...")
        self.retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=child_splitter,
        )
        
        print("⏳ Indexando documentos (esto puede tardar un momento)...")
        self.retriever.add_documents(parent_documents, ids=None)
        print("✅ Indexación completa. ¡Retriever listo para usar!")

        return self.retriever

