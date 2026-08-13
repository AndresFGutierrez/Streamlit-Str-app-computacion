# ============================================================
# CHATBOT GRATUITO CON MACHINE LEARNING + STREAMLIT
# Archivo único: app.py
# No usa OpenAI
# No usa API Keys
# No requiere modelos de pago
# Modelo ML: Multinomial Naive Bayes implementado desde cero
# ============================================================

import streamlit as st
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict


# ============================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="ChatBot ML",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. ESTILOS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
        radial-gradient(circle at top left, #172554 0%, transparent 28%),
        radial-gradient(circle at top right, #164e63 0%, transparent 25%),
        #07111f;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0;
        background: linear-gradient(
            90deg,
            #38bdf8,
            #22d3ee,
            #a5f3fc
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    .status-box {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 13px 16px;
        border-radius: 14px;
        margin-bottom: 20px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 35px;
    }

    div[data-testid="stSidebar"] {
        background: #081321;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. BASE DE CONOCIMIENTO / DATOS DE ENTRENAMIENTO
# ============================================================
#
# Cada intención tiene:
# - patrones: frases con las que se entrena el modelo
# - respuestas: respuestas posibles del chatbot
#
# Puedes agregar todas las categorías y preguntas que quieras.
# ============================================================

INTENTS = {

    "saludo": {
        "patterns": [
            "hola",
            "buenas",
            "buenos dias",
            "buenas tardes",
            "buenas noches",
            "hey",
            "holi",
            "como estas",
            "que tal",
            "hola chatbot",
            "hola asistente",
            "un saludo",
            "saludos",
        ],
        "responses": [
            "¡Hola! 👋 ¿En qué puedo ayudarte?",
            "¡Hola! 🤖 Estoy listo para ayudarte.",
            "¡Qué gusto saludarte! ¿Qué quieres saber?",
        ],
    },

    "despedida": {
        "patterns": [
            "adios",
            "chao",
            "hasta luego",
            "nos vemos",
            "hasta pronto",
            "me voy",
            "bye",
            "terminar",
            "finalizar",
        ],
        "responses": [
            "¡Hasta luego! 👋",
            "Fue un gusto ayudarte. ¡Nos vemos!",
            "¡Hasta pronto! 🤖",
        ],
    },

    "agradecimiento": {
        "patterns": [
            "gracias",
            "muchas gracias",
            "te agradezco",
            "excelente gracias",
            "perfecto gracias",
            "muy amable",
            "me ayudaste",
            "gracias por la ayuda",
        ],
        "responses": [
            "¡Con mucho gusto! 😊",
            "Para eso estoy. 🤖",
            "¡Me alegra haberte ayudado!",
        ],
    },

    "identidad": {
        "patterns": [
            "quien eres",
            "como te llamas",
            "que eres",
            "eres un robot",
            "eres inteligencia artificial",
            "eres un chatbot",
            "cual es tu nombre",
            "presentate",
        ],
        "responses": [
            (
                "Soy un chatbot construido con Python y Streamlit. "
                "Utilizo un modelo de Machine Learning llamado "
                "Multinomial Naive Bayes para identificar la intención "
                "de tus mensajes."
            )
        ],
    },

    "machine_learning": {
        "patterns": [
            "que es machine learning",
            "explicame machine learning",
            "aprendizaje automatico",
            "como funciona machine learning",
            "que significa machine learning",
            "para que sirve machine learning",
            "que es aprendizaje de maquina",
            "modelos de machine learning",
        ],
        "responses": [
            (
                "Machine Learning o aprendizaje automático es una rama "
                "de la inteligencia artificial que permite a un sistema "
                "identificar patrones en datos y utilizarlos para realizar "
                "predicciones o clasificaciones."
            ),
            (
                "El Machine Learning consiste en entrenar algoritmos con "
                "datos para que puedan reconocer patrones y tomar decisiones "
                "sin programar manualmente cada posible situación."
            ),
        ],
    },

    "naive_bayes": {
        "patterns": [
            "que es naive bayes",
            "como funciona naive bayes",
            "explicame naive bayes",
            "modelo naive bayes",
            "clasificador naive bayes",
            "algoritmo bayes",
            "bayes machine learning",
        ],
        "responses": [
            (
                "Naive Bayes es un algoritmo de clasificación basado en "
                "probabilidades. En este chatbot analiza las palabras de "
                "tu mensaje y calcula cuál de las intenciones aprendidas "
                "es la más probable."
            )
        ],
    },

    "inteligencia_artificial": {
        "patterns": [
            "que es inteligencia artificial",
            "explicame inteligencia artificial",
            "que significa ia",
            "para que sirve la inteligencia artificial",
            "como funciona una inteligencia artificial",
            "que es ia",
            "inteligencia artificial",
        ],
        "responses": [
            (
                "La inteligencia artificial es un área de la informática "
                "dedicada a crear sistemas capaces de realizar tareas que "
                "normalmente requieren capacidades asociadas a la "
                "inteligencia humana, como clasificar, predecir, reconocer "
                "patrones o procesar lenguaje."
            )
        ],
    },

    "python": {
        "patterns": [
            "que es python",
            "para que sirve python",
            "lenguaje python",
            "programacion en python",
            "python programacion",
            "aprender python",
            "que puedo hacer con python",
        ],
        "responses": [
            (
                "Python es un lenguaje de programación muy utilizado en "
                "desarrollo web, automatización, análisis de datos, "
                "inteligencia artificial y Machine Learning."
            )
        ],
    },

    "streamlit": {
        "patterns": [
            "que es streamlit",
            "para que sirve streamlit",
            "como funciona streamlit",
            "aplicacion streamlit",
            "crear app con streamlit",
            "streamlit python",
            "desplegar streamlit",
        ],
        "responses": [
            (
                "Streamlit es una herramienta de Python que permite crear "
                "aplicaciones web interactivas directamente desde código "
                "Python. Es especialmente útil para proyectos de datos, "
                "Machine Learning y prototipos."
            )
        ],
    },

    "funcionamiento_bot": {
        "patterns": [
            "como funcionas",
            "como funciona este chatbot",
            "como respondes",
            "como entiendes mis preguntas",
            "como sabes que responder",
            "como detectas lo que escribo",
            "como fuiste entrenado",
        ],
        "responses": [
            (
                "Primero limpio y separo las palabras de tu mensaje. "
                "Después un clasificador Multinomial Naive Bayes calcula "
                "la probabilidad de cada intención. Finalmente selecciono "
                "la intención más probable y genero una respuesta asociada."
            )
        ],
    },

    "capacidades": {
        "patterns": [
            "que puedes hacer",
            "en que puedes ayudarme",
            "que sabes hacer",
            "cuales son tus funciones",
            "que puedo preguntarte",
            "ayudame",
            "necesito ayuda",
            "opciones",
        ],
        "responses": [
            (
                "Puedo conversar contigo y reconocer diferentes tipos de "
                "preguntas utilizando Machine Learning. Por ejemplo, "
                "pregúntame sobre Python, Streamlit, Machine Learning, "
                "inteligencia artificial o cómo funciona este chatbot."
            )
        ],
    },

    "gratis": {
        "patterns": [
            "eres gratis",
            "esto cuesta dinero",
            "necesito pagar",
            "usa api de pago",
            "cuanto cuesta",
            "hay que pagar",
            "es gratuito",
            "tiene costo",
            "necesito una api",
            "necesito api key",
        ],
        "responses": [
            (
                "Este chatbot no necesita una API de inteligencia artificial "
                "de pago. El modelo de clasificación se ejecuta directamente "
                "con Python dentro de la aplicación."
            )
        ],
    },

    "entrenamiento": {
        "patterns": [
            "como entrenar el chatbot",
            "como agregar preguntas",
            "quiero agregar respuestas",
            "como agregar conocimiento",
            "entrenar modelo",
            "agregar datos de entrenamiento",
            "personalizar chatbot",
        ],
        "responses": [
            (
                "Puedes entrenarme con más ejemplos agregando nuevas frases "
                "en la variable INTENTS del archivo app.py. Mientras más "
                "ejemplos representativos tenga cada intención, mejor podrá "
                "clasificar preguntas similares."
            )
        ],
    },

    "contacto": {
        "patterns": [
            "quiero hablar con alguien",
            "contacto",
            "contactar",
            "soporte",
            "asesor",
            "necesito un asesor",
            "hablar con una persona",
        ],
        "responses": [
            (
                "Esta versión es una demostración. Puedes reemplazar esta "
                "respuesta con el teléfono, WhatsApp, correo o formulario "
                "de contacto de tu empresa."
            )
        ],
    },
}


# ============================================================
# 4. FUNCIONES PARA PROCESAR TEXTO
# ============================================================

def remove_accents(text):
    """
    Elimina acentos:
    'programación' -> 'programacion'
    """
    text = unicodedata.normalize("NFD", text)

    return "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )


def tokenize(text):
    """
    Convierte texto a una lista de palabras normalizadas.
    """

    text = text.lower()
    text = remove_accents(text)

    # Conservamos letras, números y algunos caracteres útiles.
    words = re.findall(r"[a-z0-9]+", text)

    return words


# ============================================================
# 5. MODELO MULTINOMIAL NAIVE BAYES DESDE CERO
# ============================================================

class NaiveBayesChatbot:

    def __init__(self, alpha=1.0):

        # Suavizado de Laplace.
        self.alpha = alpha

        # Vocabulario conocido.
        self.vocabulary = set()

        # Número de documentos de cada clase.
        self.class_document_counts = Counter()

        # Cantidad de cada palabra por clase.
        self.word_counts = defaultdict(Counter)

        # Total de palabras de cada clase.
        self.total_words = Counter()

        # Cantidad total de documentos.
        self.total_documents = 0

        # Clases disponibles.
        self.classes = []

    def fit(self, intents):
        """
        Entrena el modelo utilizando los patterns definidos
        dentro de INTENTS.
        """

        self.classes = list(intents.keys())

        for intent_name, intent_data in intents.items():

            for sentence in intent_data["patterns"]:

                self.total_documents += 1

                self.class_document_counts[intent_name] += 1

                tokens = tokenize(sentence)

                for token in tokens:

                    self.vocabulary.add(token)

                    self.word_counts[intent_name][token] += 1

                    self.total_words[intent_name] += 1

        return self

    def predict(self, text):
        """
        Predice la intención y devuelve:

        intención
        confianza
        palabras reconocidas
        """

        tokens = tokenize(text)

        if not tokens:
            return None, 0.0, 0

        known_tokens = [
            token
            for token in tokens
            if token in self.vocabulary
        ]

        # Si no reconoce absolutamente ninguna palabra,
        # evitamos inventar una intención.
        if not known_tokens:
            return None, 0.0, 0

        vocabulary_size = len(self.vocabulary)

        log_scores = {}

        # ====================================================
        # Probabilidad para cada intención
        # ====================================================

        for intent_name in self.classes:

            # P(clase)
            class_probability = (
                self.class_document_counts[intent_name]
                / self.total_documents
            )

            score = math.log(class_probability)

            # P(palabra | clase)
            for token in known_tokens:

                token_count = self.word_counts[intent_name][token]

                probability = (
                    token_count + self.alpha
                ) / (
                    self.total_words[intent_name]
                    + self.alpha * vocabulary_size
                )

                score += math.log(probability)

            log_scores[intent_name] = score

        # ====================================================
        # Encontramos la clase con mayor probabilidad
        # ====================================================

        best_intent = max(
            log_scores,
            key=log_scores.get
        )

        # ====================================================
        # Convertimos log-scores en probabilidades aproximadas
        # usando Softmax de manera numéricamente estable.
        # ====================================================

        maximum_score = max(log_scores.values())

        exponentials = {
            intent: math.exp(score - maximum_score)
            for intent, score in log_scores.items()
        }

        denominator = sum(exponentials.values())

        probabilities = {
            intent: value / denominator
            for intent, value in exponentials.items()
        }

        confidence = probabilities[best_intent]

        return (
            best_intent,
            confidence,
            len(known_tokens)
        )


# ============================================================
# 6. ENTRENAMIENTO DEL MODELO
# ============================================================

@st.cache_resource
def train_model():
    """
    Entrena una única vez el modelo mientras la aplicación
    esté funcionando.
    """

    chatbot_model = NaiveBayesChatbot(alpha=1.0)

    chatbot_model.fit(INTENTS)

    return chatbot_model


model = train_model()


# ============================================================
# 7. RESPUESTA DEL CHATBOT
# ============================================================

def generate_response(user_message):
    """
    Procesa el mensaje del usuario, predice la intención
    y devuelve la respuesta.
    """

    intent, confidence, known_words = model.predict(
        user_message
    )

    # ========================================================
    # CASO 1: mensaje completamente desconocido
    # ========================================================

    if intent is None:

        return {
            "response": (
                "🤔 Todavía no tengo suficiente conocimiento para "
                "responder esa pregunta.\n\n"
                "Puedes agregar nuevos ejemplos y respuestas en la "
                "variable `INTENTS` para enseñarme nuevos temas."
            ),
            "intent": "desconocido",
            "confidence": 0.0,
        }

    # ========================================================
    # CASO 2: confianza demasiado baja
    # ========================================================

    if confidence < 0.25:

        return {
            "response": (
                "No estoy completamente seguro de haber entendido "
                "tu pregunta. ¿Puedes escribirla de otra manera?"
            ),
            "intent": intent,
            "confidence": confidence,
        }

    # ========================================================
    # CASO 3: intención reconocida
    # ========================================================

    response = random.choice(
        INTENTS[intent]["responses"]
    )

    return {
        "response": response,
        "intent": intent,
        "confidence": confidence,
    }


# ============================================================
# 8. ESTADO DE LA CONVERSACIÓN
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = [

        {
            "role": "assistant",
            "content": (
                "¡Hola! 👋 Soy un chatbot gratuito construido con "
                "Streamlit y Machine Learning.\n\n"
                "¿En qué puedo ayudarte?"
            ),
            "intent": "inicio",
            "confidence": 1.0,
        }

    ]


# ============================================================
# 9. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🤖 ChatBot ML")

    st.markdown("---")

    st.subheader("Modelo")

    st.success(
        "Multinomial Naive Bayes"
    )

    st.write(
        "El modelo se ejecuta directamente dentro "
        "de esta aplicación."
    )

    st.markdown("---")

    st.subheader("📊 Información")

    st.metric(
        "Intenciones",
        len(INTENTS)
    )

    total_examples = sum(
        len(data["patterns"])
        for data in INTENTS.values()
    )

    st.metric(
        "Ejemplos de entrenamiento",
        total_examples
    )

    st.metric(
        "Palabras aprendidas",
        len(model.vocabulary)
    )

    st.markdown("---")

    show_diagnostics = st.toggle(
        "Mostrar diagnóstico ML",
        value=False
    )

    st.caption(
        "Actívalo para ver la intención y "
        "confianza estimada por el modelo."
    )

    st.markdown("---")

    if st.button(
        "🗑️ Limpiar conversación",
        use_container_width=True
    ):

        st.session_state.messages = [

            {
                "role": "assistant",
                "content": (
                    "Conversación reiniciada. 👋 "
                    "¿En qué puedo ayudarte?"
                ),
                "intent": "inicio",
                "confidence": 1.0,
            }

        ]

        st.rerun()


# ============================================================
# 10. ENCABEZADO PRINCIPAL
# ============================================================

st.markdown(
    '<div class="main-title">🤖 ChatBot ML</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Chatbot gratuito desarrollado con Python,
    Streamlit y Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="status-box">
    🟢 <strong>Modelo activo</strong><br>
    Sin API Key · Sin OpenAI · Sin pago por mensajes
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 11. MOSTRAR HISTORIAL
# ============================================================

for message in st.session_state.messages:

    avatar = "🤖"

    if message["role"] == "user":
        avatar = "👤"

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):

        st.markdown(
            message["content"]
        )

        # Mostramos diagnóstico únicamente para
        # mensajes del asistente.
        if (
            show_diagnostics
            and message["role"] == "assistant"
            and "intent" in message
            and message["intent"] != "inicio"
        ):

            confidence_percentage = (
                message["confidence"] * 100
            )

            st.caption(
                f"🧠 Intención: `{message['intent']}` "
                f"· Confianza: "
                f"`{confidence_percentage:.1f}%`"
            )


# ============================================================
# 12. ENTRADA DEL USUARIO
# ============================================================

user_prompt = st.chat_input(
    "Escribe tu mensaje..."
)


# ============================================================
# 13. PROCESAR NUEVO MENSAJE
# ============================================================

if user_prompt:

    # Guardamos mensaje del usuario.
    user_message = {
        "role": "user",
        "content": user_prompt,
    }

    st.session_state.messages.append(
        user_message
    )

    # Mostramos mensaje inmediatamente.
    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.markdown(user_prompt)

    # Ejecutamos modelo ML.
    result = generate_response(
        user_prompt
    )

    # Construimos mensaje del bot.
    assistant_message = {
        "role": "assistant",
        "content": result["response"],
        "intent": result["intent"],
        "confidence": result["confidence"],
    }

    # Lo guardamos.
    st.session_state.messages.append(
        assistant_message
    )

    # Mostramos respuesta.
    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):

        st.markdown(
            result["response"]
        )

        if show_diagnostics:

            confidence_percentage = (
                result["confidence"] * 100
            )

            st.caption(
                f"🧠 Intención detectada: "
                f"`{result['intent']}` "
                f"· Confianza: "
                f"`{confidence_percentage:.1f}%`"
            )


# ============================================================
# 14. PIE DE PÁGINA
# ============================================================

st.markdown(
    """
    <div class="footer">
    ChatBot ML · Python + Streamlit ·
    Modelo gratuito ejecutado localmente
    </div>
    """,
    unsafe_allow_html=True
)
