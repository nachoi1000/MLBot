# scanntech_rag
GenAI Challenge

# Procesamiento de documento
se utilizo mediante google colab una notebook con GPU para utilizar docling.
El codigo de la notebook esta en el archivo colab_notebook.ipynb

# Solucion Elegida:
Flujo RAG Directo
Recibir Pregunta: El usuario envía una consulta.

Búsqueda (Retrieval): El sistema siempre utiliza la pregunta para buscar los documentos más relevantes en la base de datos vectorial.

Aumento (Augmentation): Los documentos encontrados se inyectan en el prompt junto con la pregunta original.

Generación (Generation): El LLM recibe el prompt aumentado y genera una respuesta basada exclusivamente en la información proporcionada.

# Porcesamiento de Datos Elegido:
La estrategia de procesamiento elegida es "parent-child".
Al ser un documento tecnico por ende:
Necesitas trozos pequeños (chunks) para la búsqueda: Para que la búsqueda semántica sea precisa, los fragmentos de texto (y sus embeddings) deben ser muy específicos. Si un usuario pregunta sobre regularización L1, quieres encontrar el párrafo exacto que lo define, no un capítulo entero de 20 páginas sobre regresión.

Necesitas trozos grandes (chunks) para la generación: Una vez que encuentras ese párrafo específico, el LLM necesita más contexto para dar una buena respuesta. ¿A qué modelo se aplica? ¿Cuál era el problema que se intentaba solucionar? Este contexto se encuentra en las secciones y capítulos circundantes.