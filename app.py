import time
import platform
from datetime import datetime

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
APP_NAME = "Nova Chat"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_INPUT_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = """
Eres Nova, un asistente de inteligencia artificial útil, claro, amable y profesional.
Reglas:
- Responde principalmente en español, salvo que el usuario escriba claramente en otro idioma.
- Sé conciso cuando la pregunta sea sencilla y amplía cuando el tema lo necesite.
- Si no sabes algo o no tienes información suficiente, dilo claramente y no inventes datos.
- Usa listas, títulos y código Markdown cuando mejoren la comprensión.
- Cuando muestres código, entrégalo completo y bien formateado cuando sea razonable.
- No digas que tienes acceso a Internet, archivos, cámaras, correo u otras herramientas si no las tienes.
""".strip()

st.set_page_config(
    page_title=f"{APP_NAME} · IA local",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# ESTILOS UI / UX
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --bg: #07111f;
        --panel: rgba(15, 23, 42, 0.78);
        --panel-2: rgba(30, 41, 59, 0.72);
        --border: rgba(148, 163, 184, 0.16);
        --primary: #22d3ee;
        --primary-2: #60a5fa;
        --text: #f8fafc;
        --muted: #94a3b8;
        --success: #34d399;
    }

    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 0%, rgba(37, 99, 235, .18), transparent 28%),
            radial-gradient(circle at 88% 4%, rgba(6, 182, 212, .14), transparent 25%),
            linear-gradient(180deg, #07111f 0%, #0a1220 50%, #07101c 100%);
        color: var(--text);
    }

    .block-container {
        max-width: 1000px;
        padding-top: 2.2rem;
        padding-bottom: 7rem;
    }

    [data-testid="stSidebar"] {
        background: rgba(5, 12, 24, .96);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 1.35rem 1.45rem;
        margin-bottom: 1.1rem;
        border: 1px solid var(--border);
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(15, 23, 42, .88), rgba(8, 47, 73, .62));
        box-shadow: 0 20px 70px rgba(0, 0, 0, .22);
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 220px;
        height: 220px;
        border-radius: 999px;
        right: -90px;
        top: -110px;
        background: rgba(34, 211, 238, .13);
        filter: blur(2px);
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: .9rem;
    }

    .brand-icon {
        width: 50px;
        height: 50px;
        flex: 0 0 50px;
        display: grid;
        place-items: center;
        border-radius: 16px;
        font-size: 1.45rem;
        font-weight: 800;
        color: #04111f;
        background: linear-gradient(135deg, #67e8f9, #60a5fa);
        box-shadow: 0 10px 30px rgba(34, 211, 238, .18);
    }

    .hero h1 {
        padding: 0;
        margin: 0;
        font-size: clamp(1.65rem, 4vw, 2.35rem);
        line-height: 1.05;
        letter-spacing: -.04em;
    }

    .hero p {
        margin: .38rem 0 0 0;
        color: #b6c4d7;
        font-size: .98rem;
    }

    .badges {
        display: flex;
        flex-wrap: wrap;
        gap: .5rem;
        margin-top: 1rem;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: .4rem;
        padding: .38rem .66rem;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(15, 23, 42, .62);
        color: #cbd5e1;
        font-size: .78rem;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--success);
        box-shadow: 0 0 10px rgba(52, 211, 153, .75);
    }

    [data-testid="stChatMessage"] {
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: .35rem .55rem;
        margin-bottom: .8rem;
        background: rgba(15, 23, 42, .52);
        box-shadow: 0 10px 35px rgba(0, 0, 0, .10);
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        line-height: 1.62;
    }

    [data-testid="stChatInput"] {
        border-radius: 18px;
    }

    [data-testid="stChatInput"] textarea {
        font-size: 1rem;
    }

    .empty-state {
        padding: 1.1rem 1.2rem;
        border-radius: 18px;
        border: 1px dashed rgba(148, 163, 184, .25);
        background: rgba(15, 23, 42, .36);
        color: var(--muted);
        margin: .5rem 0 1rem;
    }

    .tiny-note {
        color: #64748b;
        font-size: .78rem;
        text-align: center;
        margin-top: 1.4rem;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 12px;
        border: 1px solid var(--border);
        transition: transform .15s ease, border-color .15s ease;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(34, 211, 238, .42);
    }

    @media (max-width: 640px) {
        .block-container {
            padding-top: 1rem;
            padding-left: .8rem;
            padding-right: .8rem;
        }
        .hero {
            border-radius: 20px;
            padding: 1.1rem;
        }
        .brand-icon {
            width: 44px;
            height: 44px;
            flex-basis: 44px;
            border-radius: 14px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# ESTADO DE SESIÓN
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "last_latency" not in st.session_state:
    st.session_state.last_latency = None

# ============================================================
# MODELO
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """Descarga el modelo desde Hugging Face la primera vez y lo reutiliza en caché."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.truncation_side = "left"

    # torch_dtype="auto" respeta el dtype recomendado en la configuración del modelo
    # y normalmente reduce memoria frente a forzar float32.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


def build_model_messages(history, turns_to_keep):
    """Construye el contexto del LLM conservando solamente los últimos turnos."""
    clean_history = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m.get("role") in {"user", "assistant"}
    ]

    # Un turno suele tener user + assistant. Dejamos margen para el último user.
    max_messages = max(2, turns_to_keep * 2 + 1)
    clean_history = clean_history[-max_messages:]

    return [{"role": "system", "content": SYSTEM_PROMPT}] + clean_history


def generate_response(tokenizer, model, device, history, temperature, top_p,
                      max_new_tokens, turns_to_keep):
    """Genera una respuesta con Qwen usando el chat template oficial del tokenizer."""
    messages = build_model_messages(history, turns_to_keep)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "repetition_penalty": 1.08,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }

    if temperature <= 0.05:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs.update({
            "do_sample": True,
            "temperature": float(temperature),
            "top_p": float(top_p),
        })

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    input_length = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][input_length:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if not answer:
        answer = "No pude generar una respuesta útil. Intenta reformular tu pregunta."

    return answer


def conversation_as_text(messages):
    lines = [f"Conversación con {APP_NAME}", "=" * 45, ""]
    for msg in messages:
        speaker = "Tú" if msg["role"] == "user" else APP_NAME
        lines.append(f"{speaker}:\n{msg['content']}\n")
    return "\n".join(lines)


def clear_chat():
    st.session_state.messages = []
    st.session_state.pending_prompt = None
    st.session_state.last_latency = None

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ✦ Nova Chat")
    st.caption("LLM gratuito ejecutado en el servidor de tu app")

    st.divider()
    st.markdown("**Modelo activo**")
    st.code(MODEL_NAME, language=None)

    temperature = st.slider(
        "Creatividad",
        min_value=0.0,
        max_value=1.2,
        value=0.65,
        step=0.05,
        help="Valores bajos = más predecible. Valores altos = más creativo.",
    )

    top_p = st.slider(
        "Diversidad (top-p)",
        min_value=0.1,
        max_value=1.0,
        value=0.90,
        step=0.05,
    )

    max_new_tokens = st.slider(
        "Longitud máxima",
        min_value=64,
        max_value=512,
        value=DEFAULT_MAX_NEW_TOKENS,
        step=32,
        help="Más tokens permiten respuestas más largas, pero tardan más.",
    )

    turns_to_keep = st.slider(
        "Memoria de conversación",
        min_value=1,
        max_value=8,
        value=4,
        step=1,
        help="Número aproximado de intercambios recientes que se envían de nuevo al modelo.",
    )

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Limpiar", use_container_width=True):
            clear_chat()
            st.rerun()

    with col_b:
        st.download_button(
            "⬇️ Exportar",
            data=conversation_as_text(st.session_state.messages),
            file_name=f"nova_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
            disabled=not bool(st.session_state.messages),
        )

    with st.expander("Información técnica"):
        st.write(f"**Python:** {platform.python_version()}")
        st.write(f"**PyTorch:** {torch.__version__}")
        st.write(f"**Hardware:** {'GPU' if torch.cuda.is_available() else 'CPU'}")
        st.write(f"**Contexto usado:** hasta {MAX_INPUT_TOKENS} tokens")
        if st.session_state.last_latency is not None:
            st.write(f"**Última generación:** {st.session_state.last_latency:.1f} s")

# ============================================================
# CABECERA
# ============================================================
st.markdown(
    f"""
    <div class="hero">
        <div class="brand-row">
            <div class="brand-icon">✦</div>
            <div>
                <h1>{APP_NAME}</h1>
                <p>Tu asistente con IA generativa · ejecución local del modelo · sin pagar por mensaje</p>
            </div>
        </div>
        <div class="badges">
            <span class="badge"><span class="status-dot"></span> Modelo local activo</span>
            <span class="badge">🤗 Hugging Face</span>
            <span class="badge">💬 Memoria conversacional</span>
            <span class="badge">🔑 Sin API key</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CARGA DEL MODELO
# ============================================================
try:
    with st.spinner("Preparando el modelo de IA… La primera ejecución descarga Qwen desde Hugging Face."):
        tokenizer, model, device = load_model()
except Exception as exc:
    st.error("No fue posible cargar el modelo de Hugging Face.")
    st.code(str(exc), language=None)
    st.info(
        "Verifica que estén instalados `transformers>=4.37`, `torch`, `accelerate` y `streamlit`, "
        "y que el servidor tenga conexión a Internet durante la primera descarga."
    )
    st.stop()

# ============================================================
# MENSAJE INICIAL + SUGERENCIAS
# ============================================================
if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <strong>¿Qué quieres hacer hoy?</strong><br>
            Puedes pedirme explicaciones, ideas, resúmenes, ayuda con programación o redacción.
        </div>
        """,
        unsafe_allow_html=True,
    )

    s1, s2, s3 = st.columns(3)
    if s1.button("💡 Explícame IA", use_container_width=True):
        st.session_state.pending_prompt = "Explícame qué es la inteligencia artificial con un ejemplo sencillo."
        st.rerun()
    if s2.button("🐍 Ayúdame con Python", use_container_width=True):
        st.session_state.pending_prompt = "Dame un ejemplo sencillo y útil para aprender Python."
        st.rerun()
    if s3.button("🧠 ¿Cómo funcionas?", use_container_width=True):
        st.session_state.pending_prompt = "Explícame de forma sencilla cómo funciona este chatbot y qué modelo utiliza."
        st.rerun()

# ============================================================
# HISTORIAL
# ============================================================
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "✦"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# ============================================================
# ENTRADA + RESPUESTA
# ============================================================
prompt = st.session_state.pending_prompt
if prompt:
    st.session_state.pending_prompt = None
else:
    prompt = st.chat_input("Escribe un mensaje…", max_chars=4000)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="✦"):
        placeholder = st.empty()
        placeholder.markdown("_Pensando…_")
        start = time.perf_counter()

        try:
            answer = generate_response(
                tokenizer=tokenizer,
                model=model,
                device=device,
                history=st.session_state.messages,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                turns_to_keep=turns_to_keep,
            )
        except Exception as exc:
            answer = (
                "Ocurrió un problema al generar la respuesta. "
                "Prueba reduciendo la longitud máxima o limpiando la conversación.\n\n"
                f"`{type(exc).__name__}: {exc}`"
            )

        st.session_state.last_latency = time.perf_counter() - start
        placeholder.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown(
    "<div class='tiny-note'>Qwen puede equivocarse. Verifica la información importante antes de tomar decisiones.</div>",
    unsafe_allow_html=True,
)
