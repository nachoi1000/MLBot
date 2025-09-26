# MLBot
**MLBot** es un asistente que responde consultas centradas en el libro "An Introduction to Statistical Learning with Applications in Python" inteligente construido con LangGraph. Utiliza la API de OpenAI para funcionar como un agente experto en el libro para que puedas consultar con un experto dudas respecto al libro.

[![Langchain](https://img.shields.io/badge/LangChain-LangGraph-blue.svg)](https://langchain-ai.github.io/langgraph/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

# Procesamiento de documento
## Pasar de .pdf a .md
Se utilizo mediante google colab una notebook con GPU para utilizar docling.
El codigo de la notebook esta en el archivo colab_notebook.ipynb.
En esta estapa ademas del texto del .pdf se "scrapeo" la gran cantidad de formulas matematicas que tenia el documento, con un formato LaTex, gracias a docling.
Con las imagenes, decidi no considerarlas en el procesamiento, ya que en el documento, cada imagen tenia bajo una descripcion y explicacion del contenido que reflejaba la imagen.

Luego de obtener el archivo .md, se obtiene la estructura del .pdf (table of contents) y se limpian los headers (#, ## como ###) del .md anterior, para arnmar los headers del .md basandome en la table of contents. Aqui utilice la notebook pdftoc_to_md.ipynb para generar el documento **PDF-GenAI-Challenge_2.md** el cual fue utilizado para luego chunkear, vectorizar y almacenar en una base de datos vectorial.

## Generacion de Chunk, Vector, Almcenambiento en Base de datos Vectorial y Estategia de Retrieval
La estrategia de procesamiento elegida es "parent-child".
Al ser un documento tecnico por ende:
Necesitas trozos pequeños (chunks) para la búsqueda: Para que la búsqueda semántica sea precisa, los fragmentos de texto (y sus embeddings) deben ser muy específicos. Si un usuario pregunta sobre regularización L1, quieres encontrar el párrafo exacto que lo define, no un capítulo entero de 20 páginas sobre regresión.

Necesitas trozos grandes (chunks) para la generación: Una vez que encuentras ese párrafo específico, el LLM necesita más contexto para dar una buena respuesta. ¿A qué modelo se aplica? ¿Cuál era el problema que se intentaba solucionar? Este contexto se encuentra en las secciones y capítulos circundantes.

Al tener un documento en un formato .md con los headers bien definidos decidi implementar un retrieval de Parent-Child.

Para ello los chunks fueron geenrados con la siguioente logica:
1- Se realizo una limpieza de cacracteres en el .md
2- Se implemento un MarkdownHeaderTextSplitter para obtener los Parent Documents.
3- Se implemento un RecursiveCharacterTextSplitter para generar los Child Documents
4- se utilizo Chroma como vectorstore, el cual utiliza text-embedding-3-small como modelo de embedding por default.
5- El retriever utilizado fue ParentDocumentRetriever

Se utilizaron componentes de Langchain en todo este proceso.

El detalle de este proceso se ve en el archivo **retrieve_factory.py** 


# Solucion Elegida:

Para la solucion de este challenge se implemento un agente con un flujo de RAG directo.

![alt text](docs/agent_mermaid_diagram.png) 

El agente sigue la logica:

- Recibir Pregunta: El usuario envía una consulta.

- Rewriter: Mediante LLM se genera un "enriquericmiento" del input del usuario para reformular la pregunta y que tenga mayor significado con la historia de la conversacion.

- Retriever: El sistema siempre utiliza la pregunta, posterior salida del Rewriter, para buscar los documentos más relevantes en la base de datos vectorial.

- Generator:
Los documentos encontrados se inyectan en el prompt junto con la pregunta original.
El LLM recibe el prompt aumentado y genera una respuesta basada exclusivamente en la información proporcionada.

- Retornar respuesta como chunks utilizados (si es necesario).

Este agente fue implementado con **LangGraph**, debido a la simpleza como a la flexibilidad de poder escalar la solucion a que implemente algun tipo de flujo mas complejo, o la utilizacion de distintas otras tools.

# Evaluacion del modelo
Para la evaluacion del modelo se realizaron 2 tests. Buscando medir **Fidelidad**, **Precision del Contexto**, **Recuperacion** y **Relevancia** utilizando **RAGAS**

Fidelidad (Faithfulness) → faithfulness

Relevancia (Answer Relevancy) → answer_relevancy

Precisión del Contexto (Context Precision) → context_precision

Recuperación del Contexto (Context Recall) → context_recall

Para esto primero mediante la utilizacion de un llm genere 2 listas, una enfocada a formulas matematicas y otra enfocada mas a conceptos generales. Genere las preguntas y su gorund_truth, 4 por cada capitulo, tanto en ingles como en español.

Luego utilice el agente para obtener las respuesta y los chunks.

Y con esta informacion, almacenada en .json implemente las evaluaciones solicitadas.
Las metricas son buenas, pero si hubo casos donde problemáticos (donde el JSON de entrada estaba mal formado o incompleto), aparece el campo error con IndexError: list index out of range, el cual afectó las metricas.

Se pueden ver estos detalles en los archivos dentro de la carpeta **tests**

---

## 🛠️ Tech Stack

* **Orquestación del Agente**: [Langgraph](https://langchain-ai.github.io/langgraph/)
* **Servidor Web**: [FastAPI](https://fastapi.tiangolo.com/)
* **Modelos de Lenguaje**: [OpenAI API](https://platform.openai.com/)
* **Búsquedas Externas**: [SerpAPI](https://serpapi.com/)
* **Base de Datos Vectorial**: [Chroma](https://github.com/chroma-core/chroma)
* **Lenguaje**: Python 3.10+

---

## ⚙️ Configuración e Instalación

Sigue estos pasos para poner en marcha el proyecto.

### Prerrequisitos

Asegúrate de tener instalado lo siguiente:
* [Python 3.10+](https://www.python.org/downloads/)

### Pasos

1.  **Clona el repositorio:**
    ```bash
    git clone [https://github.com/nachoi1000/scanntech_rag.git](https://github.com/nachoi1000/scanntech_rag.git)
    cd scanntech_rag
    ```

2.  **Crea y configura tu archivo de entorno:**
    Crea un archivo llamado `.env` en la raíz del proyecto a partir de la plantilla `env.Sample`. Deberás completar las siguientes variables:
    ```env
    # Claves de APIs (ver la sección 'Obtener Claves de API')
    OPENAI_API_KEY="tu_api_key_de_openai"

    ```

3.  **Ejecuta el start file**:
    
    Dependiendo el sistema operativo ejecuta el start.sh o start.ps1.
    el mismo archivo se encargara de:
    - Levantar el entorno de python.
    - Ejecutar el app.py para ejecutar la aplicacion.
    

---

## 🔑 Obtener Claves de API

Para que el proyecto funcione, necesita acceso a servicios externos a través de claves de API.

### OpenAI API Key

Necesitarás una clave de OpenAI para que el agente pueda pensar y procesar el lenguaje.
1.  Ve a [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
2.  Inicia sesión y crea una nueva "secret key".
3.  Copia la clave y pégala en la variable `OPENAI_API_KEY` de tu archivo `.env`.

---

## 🚀 Ejecutar el Proyecto

Los scripts de inicio construyen las imágenes de Docker y levantan los contenedores de la aplicación y la base de datos.

* En **Linux** o **macOS**:
    ```bash
    ./start.sh
    ```

* En **Windows**:
    ```powershell
    ./start.ps1
    ```

Una vez ejecutado, los servicios estarán disponibles en:

* **Aplicación MLBot**: `http://localhost:8000`

---
