import streamlit as st
import feedparser
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Terminal de Sentimiento", layout="wide")

# ==========================================
# 1. EXTRACCIÓN DIRECTA (SIN LIMITANTES)
# ==========================================
def obtener_noticias_yahoo(ticker, limite=30):
    """
    Se conecta al feed RSS de Yahoo Finance y extrae las noticias directamente,
    manteniendo el contexto amplio del mercado sin filtros restrictivos.
    """
    url_rss = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    
    try:
        # Intentamos descargar y leer el RSS
        feed = feedparser.parse(url_rss)
        if not feed.entries:
            return []
    except Exception as e:
        # Si Yahoo falla o no hay internet, mostramos un mensaje amigable
        st.error(f"Error de comunicación con el servidor de datos: {e}")
        return []

    titulares = []
    # Tomamos directamente las noticias hasta alcanzar el límite
    for entrada in feed.entries[:limite]:
        titulares.append({
            "titulo": entrada.title,
            "link": entrada.link,
            "fecha": entrada.published
        })
            
    return titulares

# ==========================================
# 2. CARGA DEL MODELO EN CACHÉ
# ==========================================
@st.cache_resource
def cargar_modelo_y_tokenizador(ruta_modelo):
    """
    Carga y mantiene el modelo en memoria. Incluye prevención de errores.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        modelo = AutoModelForSequenceClassification.from_pretrained(ruta_modelo)
        return tokenizer, modelo
    except Exception as e:
        st.error(f"Falla crítica al cargar el modelo: {e}")
        return None, None

# ==========================================
# 3. MOTOR DE INFERENCIA
# ==========================================
def clasificar_titular(texto, tokenizer, modelo):
    """
    Procesa el texto y devuelve la predicción numérica.
    """
    inputs = tokenizer(texto, padding=True, truncation=True, max_length=128, return_tensors="pt")
    
    with torch.no_grad():
        outputs = modelo(**inputs)
        
    logits = outputs.logits
    clase_predicha = np.argmax(logits.numpy(), axis=1)[0]
    
    return clase_predicha

# ==========================================
# 4. INTERFAZ GRÁFICA DE USUARIO (UI)
# ==========================================
st.title("📊 Terminal de Sentimiento Financiero")
st.markdown("Analiza el sentimiento de las últimas noticias del mercado en tiempo real.")

ruta_de_tu_modelo = "JEAR317/finbert-sentimiento-diplomado"
tokenizer, modelo = cargar_modelo_y_tokenizador(ruta_de_tu_modelo)

# Solo mostramos el buscador si el modelo cargó exitosamente
if tokenizer and modelo:
    ticker_input = st.text_input("🔍 Ingresa un Ticker (Ej. AAPL, TSLA, MSFT) y presiona Enter:", "").upper()

    if ticker_input:
        with st.spinner(f"Analizando el mercado para {ticker_input}..."):
            
            noticias = obtener_noticias_yahoo(ticker_input, limite=30)
            
            if not noticias:
                st.warning(f"No se encontraron noticias recientes para el ticker {ticker_input}.")
            else:
                noticias_bearish = []
                noticias_bullish = []
                noticias_neutral = []
                
                for noticia in noticias:
                    prediccion = clasificar_titular(noticia['titulo'], tokenizer, modelo)
                    
                    if prediccion == 0:
                        noticias_bearish.append(noticia)
                    elif prediccion == 1:
                        noticias_bullish.append(noticia)
                    else:
                        noticias_neutral.append(noticia)
                
                st.markdown("---")
                col_bear, col_bull, col_neut = st.columns(3)
                
                with col_bear:
                    st.subheader("📉 Bearish")
                    for noti in noticias_bearish[:3]:
                        st.error(f"**[{noti['titulo']}]({noti['link']})** \n*🕒 {noti['fecha']}*")
                        
                with col_bull:
                    st.subheader("📈 Bullish")
                    for noti in noticias_bullish[:3]:
                        st.success(f"**[{noti['titulo']}]({noti['link']})** \n*🕒 {noti['fecha']}*")
                        
                with col_neut:
                    st.subheader("⚖️ Neutral")
                    for noti in noticias_neutral[:3]:
                        st.info(f"**[{noti['titulo']}]({noti['link']})** \n*🕒 {noti['fecha']}*")
else:
    st.error("La aplicación no puede continuar debido a una falla al cargar el modelo.")
# ==========================================
# 4. INTERFAZ GRÁFICA DE USUARIO (UI)
# ==========================================
st.title("📊 Terminal de Sentimiento Financiero")
st.markdown("Analiza el sentimiento de las últimas noticias del mercado en tiempo real.")

# Recuerda poner aquí la ruta final de tu modelo en Hugging Face
ruta_de_tu_modelo = "tu-usuario/nombre-de-tu-modelo" 
tokenizer, modelo = cargar_modelo_y_tokenizador(ruta_de_tu_modelo)

# Solo mostramos el buscador si el modelo cargó exitosamente
if tokenizer and modelo:
    
    # Creamos dos columnas: una ancha (4) para el buscador y una estrecha (1) para el botón
    col_search, col_btn = st.columns([4, 1])
    
    with col_search:
        ticker_input = st.text_input("🔍 Ingresa un Ticker (Ej. AAPL, TSLA, MSFT):", "").upper()
        
    with col_btn:
        # Agregamos espacios vacíos para que el botón se alinee verticalmente con la caja de texto
        st.write("")
        st.write("")
        # Este botón forzará la recarga de la página al ser presionado
        btn_actualizar = st.button("🔄 Actualizar Noticias")

    # La lógica se ejecuta si el usuario escribió un ticker (ya sea al presionar Enter o dar clic al botón)
    if ticker_input:
        with st.spinner(f"Analizando el mercado para {ticker_input}..."):
            
            noticias = obtener_noticias_yahoo(ticker_input, limite=30)
            
            if not noticias:
                st.warning(f"No se encontraron noticias recientes para el ticker {ticker_input}.")
            else:
                noticias_bearish = []
                noticias_bullish = []
                noticias_neutral = []
                
                for noticia in noticias:
                    prediccion = clasificar_titular(noticia['titulo'], tokenizer, modelo)
                    
                    if prediccion == 0:
                        noticias_bearish.append(noticia)
                    elif prediccion == 1:
                        noticias_bullish.append(noticia)
                    else:
                        noticias_neutral.append(noticia)
                
                st.markdown("---")
                col_bear, col_bull, col_neut = st.columns(3)
                
                with col_bear:
                    st.subheader("📉 Bearish")
                    for noti in noticias_bearish[:3]:
                        st.error(f"**[{noti['titulo']}]({noti['link']})** \n*🕒 {noti['fecha']}*")
                        
                with col_bull:
                    st.subheader("📈 Bullish")
                    for noti in noticias_bullish[:3]:
                        st.success(f"**[{noti['titulo']}]({noti['link']})** \n*🕒 {noti['fecha']}*")
                        
                with col_neut:
                    st.subheader("⚖️ Neutral")
                    for noti in noticias_neutral[:3]:
                        st.info(f"**[{noti['titulo']}]({noti['link']})** \n*🕒 {noti['fecha']}*")
else:
    st.error("La aplicación no puede continuar debido a una falla al cargar el modelo.")
