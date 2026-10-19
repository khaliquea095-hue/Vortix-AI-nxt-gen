"""
Vortix AI: a Streamlit chat app with streaming replies, web search with sources,
image creation, and model switching.

Run locally:  streamlit run app.py
Deploy free:  Streamlit Community Cloud, from a GitHub repo (see README.md)
"""

import base64
import hmac
import os
import re
import urllib.request

import anthropic
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Vortix AI", page_icon="⚡", layout="centered")

APP_NAME = "Vortix AI"
MODELS = ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"]
IMAGE_MODEL = "gpt-image-1"  # change if your OpenAI account has a newer image model
MAX_TOKENS = 4096
MEMORY_WINDOW = 40  # recent messages sent to the model with each request

# Anthropic's web search tool. See the README for the newest version.
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}

DEFAULT_SYSTEM_PROMPT = """You are Vortix AI, a fast, honest, and capable assistant.
- Give direct, accurate answers. Say clearly when you are not sure.
- Use short lists and code blocks when they make an answer easier to use.
- Ask one brief clarifying question only when a request is truly ambiguous."""

# Catches requests like "create an image of a fox in snow".
IMAGE_REQUEST = re.compile(
    r"^\s*(?:please\s+)?(?:generate|create|make|draw)\s+(?:me\s+)?(?:an?\s+)?"
    r"(?:image|picture|photo|illustration)s?\s+of\s+(.+)$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Settings, access, and clients
# ---------------------------------------------------------------------------

def get_secret(name, default=None):
    """Read a setting from Streamlit Secrets first, then from environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass  # no secrets file on this machine
    return os.getenv(name, default)


def require_password():
    """If APP_PASSWORD is set, stop the app until the visitor types it."""
    expected = get_secret("APP_PASSWORD")
    if not expected or st.session_state.get("unlocked"):
        return
    typed = st.text_input("Password", type="password")
    if typed:
        if hmac.compare_digest(typed, str(expected)):
            st.session_state.unlocked = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()


def make_claude_client():
    key = get_secret("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=key) if key else None


def system_prompt():
    # Keep your real prompt in Secrets so it stays out of a public GitHub repo.
    return get_secret("SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Chat helpers
# ---------------------------------------------------------------------------

def api_history():
    """Recent chat in the text-only format the Claude API expects."""
    recent = st.session_state.messages[-MEMORY_WINDOW:]
    while recent and recent[0]["role"] != "user":  # a conversation must open with a user turn
        recent = recent[1:]
    return [{"role": m["role"], "content": m["content"]} for m in recent]


def extract_sources(response):
    """Unique (title, url) pairs from the web search results in a response."""
    sources, seen = [], set()
    for block in response.content:
        if block.type != "web_search_tool_result" or not isinstance(block.content, list):
            continue
        for item in block.content:
            url = getattr(item, "url", None)
            if url and url not in seen:
                seen.add(url)
                sources.append((getattr(item, "title", None) or url, url))
    return sources


def stream_answer(client, model):
    """Normal chat: tokens appear as they are generated."""
    try:
        with client.messages.stream(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system_prompt(),
            messages=api_history(),
        ) as stream:
            text = st.write_stream(stream.text_stream)
    except anthropic.APIError as err:
        st.error(f"API error: {err}")
        return None
    if not text.strip():
        return None
    return {"role": "assistant", "content": text, "image": None}


def search_answer(client, model):
    """Web search mode: answer from live results and list the sources."""
    system = system_prompt() + "\nYou can search the web. Use it for anything recent or uncertain."
    messages = api_history()
    try:
        with st.spinner("Searching the web..."):
            while True:
                response = client.messages.create(
                    model=model,
                    max_tokens=MAX_TOKENS,
                    system=system,
                    tools=[WEB_SEARCH_TOOL],
                    messages=messages,
                )
                if response.stop_reason != "pause_turn":
                    break
                # A long search can pause mid-turn. Send the partial turn back so it continues.
                messages = messages + [{"role": "assistant", "content": response.content}]
    except anthropic.APIError as err:
        st.error(f"API error: {err}")
        return None

    answer = "".join(block.text for block in response.content if block.type == "text")
    sources = extract_sources(response)
    if sources:
        links = []
        for title, url in sources:
            safe_title = title.replace("[", "(").replace("]", ")")
            links.append(f"- [{safe_title}]({url})")
        answer += "\n\n**Sources**\n" + "\n".join(links)
    if not answer.strip():
        return None
    st.markdown(answer)
    return {"role": "assistant", "content": answer, "image": None}


def make_image(prompt):
    key = get_secret("OPENAI_API_KEY")
    if not key:
        st.warning("Image creation needs OPENAI_API_KEY in Secrets.")
        return None
    try:
        with st.spinner("Creating image..."):
            result = OpenAI(api_key=key).images.generate(
                model=IMAGE_MODEL, prompt=prompt, size="1024x1024"
            )
    except Exception as err:  # the image provider raises its own error types
        st.error(f"Image error: {err}")
        return None

    item = result.data[0]
    if getattr(item, "b64_json", None):
        image_bytes = base64.b64decode(item.b64_json)
    elif getattr(item, "url", None):
        with urllib.request.urlopen(item.url) as resp:
            image_bytes = resp.read()
    else:
        st.error("No image data came back.")
        return None

    st.image(image_bytes, caption=prompt)
    st.download_button(
        "Download image",
        data=image_bytes,
        file_name="vortix_image.png",
        mime="image/png",
        key=f"new_download_{len(st.session_state.messages)}",
    )
    return {"role": "assistant", "content": f"Image created for: {prompt}", "image": image_bytes}


def handle_prompt(client, prompt, model, use_search):
    st.session_state.messages.append({"role": "user", "content": prompt, "image": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        image_request = IMAGE_REQUEST.match(prompt)
        if image_request:
            reply = make_image(image_request.group(1))
        elif use_search:
            reply = search_answer(client, model)
        else:
            reply = stream_answer(client, model)

    if reply:
        st.session_state.messages.append(reply)
    else:
        st.session_state.messages.pop()  # a failed request leaves no trace in the history


def show_history():
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("image"):
                st.image(msg["image"])
                st.download_button(
                    "Download image",
                    data=msg["image"],
                    file_name="vortix_image.png",
                    mime="image/png",
                    key=f"history_download_{i}",
                )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def main():
    require_password()

    client = make_claude_client()
    if client is None:
        st.error("Add ANTHROPIC_API_KEY in Streamlit Secrets (see the README) to start chatting.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header(APP_NAME)
        model = st.selectbox("Model", MODELS)
        use_search = st.toggle("Web search", value=False, help="Answer from live web results, with sources.")
        if st.button("New chat"):
            st.session_state.messages = []
        st.caption('For images, type "create an image of ...". Chat resets when you refresh the page.')

    st.title(f"⚡ {APP_NAME}")
    show_history()

    prompt = st.chat_input("Ask Vortix anything...")
    if prompt:
        handle_prompt(client, prompt, model, use_search)


main()
