import streamlit as st
from rag import RAGSystem
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from dotenv import load_dotenv

load_dotenv()

# Configuración Langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "chatbot-rag"

# Inicialización
if "rag" not in st.session_state:
    st.session_state.rag = RAGSystem()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Plantilla de prompt
PROMPT_TEMPLATE = """
Responde la pregunta basándote solo en el siguiente contexto:
{context}

Pregunta: {input}
"""

# Interfaz
st.title("Chatbot RAG con PDF")

# Sidebar para configuraciones del modelo
with st.sidebar:
    st.header("Configuración del Modelo")
    
    # Controles deslizantes para los parámetros del modelo
    temperature = st.slider("Temperatura", min_value=0.0, max_value=1.0, value=0.7, step=0.01,
                           help="Controla la aleatoriedad: valores más bajos = respuestas más deterministas")
    
    top_p = st.slider("Top P", min_value=0.1, max_value=1.0, value=0.9, step=0.01,
                     help="Muestreo de núcleo: considera solo los tokens cuya probabilidad acumulada supera este valor")
    
    top_k = st.slider("Top K", min_value=1, max_value=100, value=50,
                     help="Limita la selección a los K tokens más probables en cada paso")
    
    max_length = st.slider("Longitud máxima", min_value=64, max_value=4096, value=2048,
                          help="Longitud máxima de tokens en la respuesta generada")
    
    repeat_penalty = st.slider("Penalización de repetición", min_value=1.0, max_value=2.0, value=1.1, step=0.01,
                              help="Penaliza tokens repetidos para evitar repeticiones")
    
    num_ctx = st.slider("Contexto de memoria", min_value=512, max_value=4096, value=2048,
                       help="Tamaño del contexto de memoria del modelo")
    
    model_name = st.selectbox("Modelo", options=["llama3", "mistral", "phi3", "gemma"], index=0)

# Carga de PDF
uploaded_file = st.file_uploader("Sube un PDF", type="pdf")
if uploaded_file:
    with open("./data/documento.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    chunks = st.session_state.rag.process_pdf("./data/documento.pdf")
    st.success(f"PDF procesado con {chunks} chunks")

# Historial de chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de usuario
if prompt := st.chat_input("Haz una pregunta sobre el documento"):
    # Añadir mensaje de usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Procesar consulta
    docs = st.session_state.rag.query(prompt)
    
    # Configurar LLM con parámetros seleccionados
    llm = Ollama(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_ctx=num_ctx,
        repeat_penalty=repeat_penalty,
        num_predict=max_length,
        stop=["<|endoftext|>", "<|eot_id|>"]  # Tokens de parada comunes
    )
    
    # Crear cadena de generación
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    
    # Generar respuesta con indicador de progreso
    with st.spinner("Generando respuesta..."):
        response = document_chain.invoke({
            "input": prompt,
            "context": docs
        })
    
    # Mostrar respuesta
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})