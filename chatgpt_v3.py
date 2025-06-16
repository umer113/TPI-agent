import os
import json
import shutil
import subprocess
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from openai import AsyncOpenAI
from groq import Groq  # your Groq client import

load_dotenv()


API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.error("Please set the OPENAI_API_KEY in your .env file.")
    st.stop()

OPENAI_API_KEY = API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CHAT_DIR = "chat_history"
DATA_DIR = "data"
os.makedirs(CHAT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ——— Chat Persistence ———
def _clean_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_ " else "_" for c in s).strip().replace(" ", "_")

def save_chat(history, chat_id=None, title=None):
    if title:
        new_id = _clean_filename(title)[:50] or datetime.now().strftime("%Y%m%d_%H%M%S")
    elif chat_id:
        new_id = chat_id
    else:
        new_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    new_path = os.path.join(CHAT_DIR, f"{new_id}.json")
    if chat_id and chat_id != new_id:
        old_path = os.path.join(CHAT_DIR, f"{chat_id}.json")
        if os.path.exists(old_path):
            os.remove(old_path)

    with open(new_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return new_id

def load_chats():
    chats = []
    for fname in sorted(os.listdir(CHAT_DIR), reverse=True):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(CHAT_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            msgs = json.load(f)
        chats.append({"id": fname[:-5], "messages": msgs})
    return chats


# ——— Scraper Runner ———
def list_scrapers(scraper_dir="scrapers"):
    return [f[:-3] for f in os.listdir(scraper_dir)
            if f.endswith(".py") and not f.startswith("__")]

def run_scraper(module_name, scraper_dir="scrapers"):
    before = set(os.listdir(DATA_DIR))
    path = os.path.join(scraper_dir, f"{module_name}.py")
    try:
        subprocess.run(["python", path], check=True)
    except Exception as e:
        st.sidebar.error(f"Scraper error: {e}")
        return
    new_files = set(os.listdir(DATA_DIR)) - before
    if new_files:
        for fn in new_files:
            st.sidebar.success(f"New CSV: {fn}")
    else:
        st.sidebar.info("No CSV produced.")


def estimate_tokens(text):
    return len(text.split()) * 1.3  # approximate token count

async def ask_agent(csv_text: str, question: str, model: str):
    system_prompt = """
You are ChatGPT, a large language model trained by OpenAI.
You respond in a friendly, conversational style—clear, concise, and helpful—just like the ChatGPT interface.

Your task is to use the provided CSV data to comprehensively address the user's query. You may summarize, explain, or generate insightful content based entirely on the CSV data. 
• Do not include any facts that aren't explicitly or implicitly supported by the CSV.
• If the data does not contain relevant information to answer the query, explicitly state this.
"""
    MAX_TOKENS = 14000
    client = None
    is_groq = model.startswith("meta-llama/")
    combined_response = ""

    # Estimate total tokens
    total_tokens = estimate_tokens(csv_text + question + system_prompt)

    if total_tokens <= MAX_TOKENS:
        user_prompt = f"### CSV Data:\n{csv_text}\n\n### User Question:\n{question}"
        if is_groq:
            client = Groq(api_key=GROQ_API_KEY)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return resp.choices[0].message.content.strip()
        else:
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return resp.choices[0].message.content.strip()

    # If CSV too long, chunk it
    lines = csv_text.strip().split("\n")
    header = lines[0]
    rows = lines[1:]
    chunk_size = max(10, len(rows) // ceil(total_tokens / MAX_TOKENS))
    chunks = [rows[i:i+chunk_size] for i in range(0, len(rows), chunk_size)]

    for i, chunk_rows in enumerate(chunks):
        chunk_csv = header + "\n" + "\n".join(chunk_rows)
        user_prompt = f"### CSV Data:\n{chunk_csv}\n\n### User Question:\n{question}"
        if is_groq:
            if client is None:
                client = Groq(api_key=GROQ_API_KEY)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            part = resp.choices[0].message.content.strip()
        else:
            if client is None:
                client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            part = resp.choices[0].message.content.strip()

        combined_response += f"\n\n--- Part {i+1} ---\n\n{part}"

    return combined_response.strip()


# ——— Streamlit App ———
def main():
    # ─── 0. Inject CSS ───
    st.markdown("""
    <style>
      .user-message {
        border: 1px solid #6B7280;       /* gray-500 */
        color: #F9FAFB;                  /* gray-50 */
        padding: 0.75rem 1rem;
        border-radius: 0.375rem;
        margin: 0.5rem 0;
        max-width: 80%;
      }
    </style>
    """, unsafe_allow_html=True)

    st.title("🕸️ TPI")

    if "query" not in st.session_state:
        st.session_state["query"] = ""

    st.sidebar.image("logo.png", width=200)
    st.sidebar.header("🔁 Chat History")

    chats = load_chats()
    options = ["🆕 New Chat"] + [
        c.get("title", c['id']) for c in chats
    ]
    default_idx = 0
    if st.session_state.get("chat_id"):
        default_idx = next(
            (i+1 for i, c in enumerate(chats) if c["id"] == st.session_state.chat_id),
            0
        )
    sel = st.sidebar.selectbox(
        "Select chat",
        options,
        index=default_idx,
        key="chat_select"
    )
    if sel != "🆕 New Chat":
        idx = options.index(sel) - 1
        st.session_state.chat_id = chats[idx]["id"]
        st.session_state.chat_history = chats[idx]["messages"].copy()
    else:
        # ─── Generate unique timestamp-based title ───
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        existing_titles = {c.get("title", "") for c in chats}
        unique_ts = ts
        counter = 1
        while unique_ts in existing_titles:
            unique_ts = f"{ts}_{counter}"
            counter += 1

        st.session_state.chat_id = None
        st.session_state.chat_history = []
        st.session_state["new_chat_title"] = unique_ts

    # ─── 2. Scraper & Model ───
    st.sidebar.header("🕷️ Run Scraper")
    raw_model = st.sidebar.selectbox(
        "Model",
        ["gpt-3.5-turbo-16k", "gpt-4", "Groq"],
        key="model_select"
    )
    # Map the "Groq" label to the actual Groq model name
    model = raw_model if raw_model != "Groq" else "meta-llama/llama-4-scout-17b-16e-instruct"

    scrapers = list_scrapers()
    choice   = st.sidebar.selectbox("Select scraper", scrapers)

    if st.sidebar.button("Run Scraper"):
        df = run_scraper(choice)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Case 1: scraper returned a DataFrame
        if isinstance(df, pd.DataFrame):
            fn = f"{choice}_{ts}.csv"
            os.makedirs("data", exist_ok=True)
            df.to_csv(os.path.join("data", fn), index=False)
            st.sidebar.success(f"Saved {fn}")

        # Case 2: scraper wrote its own CSV named "<choice>.csv"
        elif df is None:
            src = f"{choice}.csv"
            if os.path.exists(src):
                fn = f"{choice}_{ts}.csv"
                os.makedirs("data", exist_ok=True)
                shutil.move(src, os.path.join("data", fn))
                st.sidebar.success(f"Scraping done {src} → data/{fn}")
            else:
                st.sidebar.error(f"No DataFrame returned and `{src}` not found.")

        # Case 3: neither DataFrame nor fallback file
        else:
            st.sidebar.error("Scraper did not return a DataFrame.")

    # ─── Sidebar: Browse Past Data ───
    files = sorted(
        [f for f in os.listdir("data") if f.lower().endswith(".csv")],
        key=lambda f: os.path.getmtime(os.path.join("data", f)),
        reverse=True
    )
    if not files:
        st.sidebar.info("No data yet. Run a scraper.")
        return

    labels = [
        f"{f} — {datetime.fromtimestamp(os.path.getmtime(os.path.join('data', f))).strftime('%Y-%m-%d %H:%M:%S')}"
        for f in files
    ]
    sel = st.sidebar.selectbox("Browse past data", labels)
    sel_file = files[labels.index(sel)]
    df = pd.read_csv(os.path.join("data", sel_file))

    # ─── Main UI: Show Dataset & Ask Agent ───

    # ─── 4. Predefined Prompts ───
    st.sidebar.header("Predefined Prompts")
    predef_prompts = [
        "Summarize the key insights from this dataset.",
        "Write a comprehensive article based on this dataset, weaving in key insights, context, and potential implications.",
        "Give me a narrative overview of what this data represents."
    ]
    def _set_query(p):
        st.session_state["query"] = p

    for i, p in enumerate(predef_prompts):
        st.sidebar.button(
            p,
            key=f"predef_{i}",
            on_click=_set_query,
            args=(p,)
        )

    # ─── 5. Show Dataset ───
    df = pd.read_csv(os.path.join(DATA_DIR, sel_file))
    st.subheader(f"Dataset: {sel_file}")
    st.dataframe(df)        # <-- let Streamlit fill the column
    st.markdown("---")

    # ─── 6. Chat Form ───
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.form("chat_form", clear_on_submit=False):
        query = st.text_input("Ask anything—article, summary, insight…", key="query")
        submitted = st.form_submit_button("Ask Agent")
        if submitted and query:
            st.session_state.chat_history.append({"role": "user", "content": query})
            csv_text = df.to_csv(index=False)
            with st.spinner("🤖 Agent is thinking..."):
                answer = asyncio.run(ask_agent(csv_text, query, model))
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            csv_basename = os.path.splitext(sel_file)[0]
            chat_title = st.session_state.get("new_chat_title", "untitled")

            new_id = save_chat(
                st.session_state.chat_history,
                chat_id=st.session_state.get("chat_id"),
                title=chat_title
            )
            st.session_state.chat_id = new_id
            st.session_state.chat_id = new_id
            try:
                st.experimental_rerun()
            except AttributeError:
                st.rerun()

    # ─── 7. Display Chat ───
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"**🤖** {msg['content']}")

    st.markdown("---")


if __name__ == "__main__":
    main()
