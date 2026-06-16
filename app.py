import streamlit as st
import feedparser
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Importamos librerías nativas de Python para manejo de tiempo
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Terminal de Sentimiento", layout="wide")

# ==========================================
# 1. EXTRACCIÓN Y FORMATEO DE TIEMPO
# ==========================================
def obtener_noticias_yahoo(ticker, limite=30):
    """
    Se conecta al feed RSS de Yahoo Finance, extrae las noticias directamente
    y formatea la fecha a la zona horaria de Ciudad de México.
    """
    url_rss = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    
    try:
        feed = feedparser.parse(url_rss)
        if not feed.entries:
            return []
    except Exception as e:
        st.error(f"Error de comunicación con el servidor de datos: {e}")
        return []

    titulares = []
    
    for entrada in feed.entries[:limite]:
        # ----- LÓGICA DE CONVERSIÓN DE FECHA -----
        try:
            # 1. Convertimos el texto del RSS a un objeto de tiempo (UTC)
            tiempo_servidor = parsedate_to_datetime(entrada.published)
            
            # 2. Transformamos a la zona horaria de CDMX
            tiempo_cdmx = tiempo_servidor.astimezone(ZoneInfo("America/Mexico_City"))
            
            # 3. Le damos el formato solicitado: HH:MM:SS dd-mm-aaaa
            fecha_legible = tiempo_cdmx.strftime("%H:%M:%S %d-%m-%Y")
        except Exception:
            # Si Yahoo llega a mandar un formato raro, mostramos el original para que no falle la app
            fecha_legible = entrada.published
        # -----------------------------------------

        titulares.append({
            "titulo": entrada.title,
            "link": entrada.link,
            "fecha": fecha_legible  # Guardamos nuestra nueva fecha formateada
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

# Reemplaza esto con tu ruta real de Hugging Face
ruta_de_tu_modelo = "JEAR317/finbert-sentimiento-diplomado"
tokenizer, modelo = cargar_modelo_y_tokenizador(ruta_de_tu_modelo)

if tokenizer and modelo:
    # Buscador principal (Ejecuta la búsqueda al presionar Enter)
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
                        # La fecha ahora se mostrará como: 07:31:02 16-06-2026
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
