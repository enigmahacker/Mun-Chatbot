import streamlit as st

import toml
import os

# Load API key from TOML
config = toml.load("config.toml")
HUGGINGFACE_API_KEY = config["api_keys"]["huggingface"]
os.environ["HUGGINGFACE_API_KEY"] = "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"


import os
import requests
import json
from datetime import datetime
import time
from typing import Optional

# -----------------------------
# MUN AI Assistant - Streamlit
# Improved and corrected version
# -----------------------------

# HuggingFace Spaces compatibility
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Page config
st.set_page_config(
    page_title="MUN AI Assistant",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS (kept simple and accessible)
st.markdown(
    """
    <style>
        .main-header { font-size: 2.2rem; color: #1e88e5; text-align: center; margin-bottom: 1rem; }
        .country-badge { background-color: #e3f2fd; padding: 0.35rem 0.8rem; border-radius: 16px; display:inline-block; margin:0.15rem; font-weight:600; }
        .resolution-format { background-color: #fff8e1; padding: 1rem; border-left: 4px solid #ff9800; font-family: monospace; white-space: pre-wrap; }
        .diplomatic-response { background-color: #e8f5e8; padding: 1rem; border-left: 4px solid #4caf50; font-style: italic; white-space: pre-wrap; }
        .user-msg { background:#f1f8ff; padding:0.6rem; border-radius:8px; }
        .assistant-msg { background:#fff; padding:0.6rem; border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helper utilities
# -----------------------------

def validate_api_key(key: Optional[str]) -> bool:
    if not key:
        return False
    return isinstance(key, str) and key.startswith("hf_") and len(key) > 10


def hf_post_with_retries(url: str, headers: dict, payload: dict, max_attempts: int = 4, backoff_base: float = 1.0):
    """Simple retry with exponential backoff for transient errors (503, 429, network issues)."""
    attempt = 0
    while attempt < max_attempts:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            # Success
            if resp.status_code == 200:
                return resp
            # Retryable statuses
            if resp.status_code in (429, 503, 500):
                attempt += 1
                wait = backoff_base * (2 ** (attempt - 1)) + (0.2 * attempt)
                time.sleep(wait)
                continue
            # Non-retryable
            return resp
        except requests.RequestException:
            attempt += 1
            wait = backoff_base * (2 ** (attempt - 1))
            time.sleep(wait)
    # If exhausted attempts, raise last exception-like response
    raise RuntimeError("Failed to reach the HuggingFace Inference API after retries")

# -----------------------------
# Core MUN Assistant class
# -----------------------------
class MUNAssistant:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.base_url = "https://api-inference.huggingface.co/models/"

        # Light config - move large static data here so it isn't recreated repeatedly
        self.committees = {
            "UNSC": "United Nations Security Council",
            "UNGA": "United Nations General Assembly",
            "ECOSOC": "Economic and Social Council",
            "HRC": "Human Rights Council",
            "UNEP": "United Nations Environment Programme",
            "WHO": "World Health Organization",
            "UNESCO": "United Nations Educational, Scientific and Cultural Organization",
        }

        self.country_positions = {
            "USA": {"stance": "Western democratic", "allies": ["UK", "France", "Canada"], "priorities": ["Security", "Democracy", "Free Trade"]},
            "Russia": {"stance": "Eastern power", "allies": ["China", "Belarus"], "priorities": ["Sovereignty", "Security", "Energy"]},
            "China": {"stance": "Rising power", "allies": ["Russia", "Pakistan"], "priorities": ["Development", "Sovereignty", "Trade"]},
            "UK": {"stance": "Western ally", "allies": ["USA", "France"], "priorities": ["Democracy", "Human Rights", "Trade"]},
            "France": {"stance": "European leader", "allies": ["Germany", "UK"], "priorities": ["Multilateralism", "Climate", "Culture"]},
            "Germany": {"stance": "European power", "allies": ["France", "EU"], "priorities": ["Environment", "Economy", "Peace"]},
            "India": {"stance": "Non-aligned", "allies": ["Various"], "priorities": ["Development", "South-South Cooperation"]},
            "Brazil": {"stance": "Regional power", "allies": ["BRICS"], "priorities": ["Environment", "Development", "Regional Stability"]},
            "South Africa": {"stance": "African leader", "allies": ["AU"], "priorities": ["Human Rights", "Development", "Peace"]},
            "Japan": {"stance": "Asian democracy", "allies": ["USA"], "priorities": ["Security", "Technology", "Regional Stability"]},
        }

    def set_api_key(self, key: str):
        self.api_key = key

    def query_huggingface(self, model: str, prompt: str, max_tokens: int = 800, temperature: float = 0.7) -> str:
        """Query HuggingFace Inference API with retries and helpful error messages."""
        if not validate_api_key(self.api_key):
            return "Error: Invalid or missing HuggingFace API key. It should start with 'hf_'."

        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": max_tokens, "temperature": temperature, "do_sample": True, "return_full_text": False},
        }

        url = f"{self.base_url}{model}"
        try:
            resp = hf_post_with_retries(url, headers, payload)
        except RuntimeError as e:
            return f"Error: {str(e)}"

        if resp.status_code == 200:
            try:
                result = resp.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "(no text returned)")
                # some models return dicts
                if isinstance(result, dict):
                    # try common keys
                    return result.get("generated_text") or json.dumps(result)
                return str(result)
            except Exception:
                return "Error: Unable to parse response JSON from model."

        # Handle common error codes gracefully
        if resp.status_code in (401, 403):
            return "Error: Authentication failed (401/403). Please check your HuggingFace API key and permissions."
        if resp.status_code == 429:
            return "Error: Rate limited (429). Try again later or use a smaller model."
        if resp.status_code == 503:
            return "Error: Model is currently loading (503). Try again in a few moments."

        return f"API Error: {resp.status_code} - {resp.text}"

    def create_mun_prompt(self, mode: str, country: str, committee: str, topic: str, question: str) -> str:
        country_info = self.country_positions.get(country, {"stance": "Neutral", "priorities": ["Peace", "Development"]})
        committee_full = self.committees.get(committee, committee)

        prompts = {
            "Position Paper": f"""
You are representing {country} in the {committee_full}.
Country stance: {country_info['stance']}
Key priorities: {', '.join(country_info['priorities'])}

Write a concise position paper for {country} on the topic: "{topic}"
Include:
1) Official position (1 short paragraph)
2) Historical context & national interest (2-3 short bullet points)
3) Proposed solutions (3 concrete, implementable points)
4) Areas for international cooperation (2 bullets)

Position Paper:
""",

            "Resolution Drafting": f"""
You are drafting a formal UN resolution for the {committee_full} on "{topic}".
Leading sponsor: {country}

Draft a UN-style resolution with:
- 5 preambulatory clauses
- 6 operative clauses (clear actions, responsible parties, and timelines where possible)
- Use formal diplomatic language

Resolution:
""",

            "Debate Speech": f"""
You are the delegate of {country} speaking in the {committee_full}.
Topic: {topic}
Country priorities: {', '.join(country_info['priorities'])}

Deliver a 2-minute speech addressing: {question}
Use formal diplomatic tone, reference the country's position, and propose 2-3 concrete policy steps.
Speech:
""",

            "Crisis Response": f"""
CRISIS UPDATE: {question}
You are {country}'s delegate in {committee_full}.
Respond considering national interests, regional security, international law, and diplomatic relations.
Provide an immediate statement (2 short paragraphs) and 3 recommended actions.

Crisis Response:
""",

            "Negotiation Strategy": f"""
You are strategizing for {country} in {committee_full}.
Topic: {topic}
Allies: {', '.join(country_info.get('allies', ['Various']))}

Provide a negotiation plan: key talking points, potential allies/opponents, compromise positions, and clear red lines.

Strategy:
""",

            "Research Brief": f"""
Research briefing for {country}'s delegation to {committee_full} on: {topic}
Provide: 1) succinct background, 2) key facts & (where to check them), 3) stakeholders, 4) previous UN actions, 5) suggested next steps.

Brief:
""",
        }

        return prompts.get(mode, f"MUN Question: {question}\n\nAnswer:")


# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    # Initialize bot in session state
    if "bot" not in st.session_state:
        st.session_state.bot = MUNAssistant()

    # Messages history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Header
    st.markdown('<h1 class="main-header">🌍 MUN AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#666;">Your AI partner for Model United Nations preparation</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("🏛️ Configuration")

        api_key_input = st.text_input("🔑 HuggingFace API Key:", type="password", value=st.session_state.bot.api_key or "", placeholder="hf_...", help="Get your token from https://huggingface.co/settings/tokens")
        if api_key_input:
            st.session_state.bot.set_api_key(api_key_input.strip())
            if validate_api_key(api_key_input.strip()):
                st.success("API key configured")
            else:
                st.error("API key format doesn't look valid (should start with 'hf_')")

        st.subheader("🎭 Delegation Setup")
        countries = sorted(list(st.session_state.bot.country_positions.keys()) + [
            "Netherlands", "Sweden", "Norway", "Australia", "Canada", "Mexico", "Argentina", "Nigeria", "Egypt", "Iran", "Turkey", "Indonesia", "Thailand", "Philippines", "Kenya", "Morocco",
        ])
        selected_country = st.selectbox("Your Country:", countries, index=countries.index("India") if "India" in countries else 0)

        committee = st.selectbox("Committee:", list(st.session_state.bot.committees.keys()))
        topic = st.text_input("Current Topic:", placeholder="e.g., Climate Change and Security")

        st.subheader("🎯 Mode")
        mun_modes = ["Position Paper", "Resolution Drafting", "Debate Speech", "Crisis Response", "Negotiation Strategy", "Research Brief"]
        selected_mode = st.selectbox("Mode:", mun_modes)

        st.subheader("🤖 AI Model")
        models = {
            "Llama 2 Chat (Recommended)": "meta-llama/Llama-2-7b-chat-hf",
            "Flan-T5 Large (Fast)": "google/flan-t5-large",
            "GPT-2 (Lightweight)": "gpt2",
            "Mistral 7B (Alternative)": "mistralai/Mistral-7B-Instruct-v0.1",
        }
        selected_model_name = st.selectbox("Model:", list(models.keys()))
        selected_model = models[selected_model_name]
        st.caption(f"ℹ️ Model note: choose a model based on quality vs speed")

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.experimental_rerun()

    # Chat input
    user_input = st.chat_input("What MUN assistance do you need?")

    # Left column: chat
    col1, col2 = st.columns([2, 1])
    with col1:
        # show history
        for msg in st.session_state.messages:
            role = msg.get("role")
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                with st.chat_message("assistant"):
                    content = msg.get("content", "")
                    mode = msg.get("mode", "General")
                    if msg.get("mode") == "Resolution Drafting":
                        st.markdown('<div class="resolution-format">', unsafe_allow_html=True)
                        st.markdown(content)
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif msg.get("mode") == "Debate Speech":
                        st.markdown('<div class="diplomatic-response">', unsafe_allow_html=True)
                        st.markdown(content)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='assistant-msg'>{content}</div>", unsafe_allow_html=True)
                    st.caption(f"Mode: {mode} | Country: {msg.get('country','N/A')}")

        # Handle new input
        if user_input:
            if not st.session_state.bot.api_key:
                st.error("🔑 Please enter your HuggingFace API key in the sidebar!")
                st.stop()
            if not topic:
                st.warning("📝 Please enter a topic in the sidebar!")
                st.stop()

            # append user message
            st.session_state.messages.append({"role": "user", "content": user_input})

            # generate response
            with st.chat_message("assistant"):
                with st.spinner(f"Generating {selected_mode}..."):
                    prompt = st.session_state.bot.create_mun_prompt(selected_mode, selected_country, committee, topic, user_input)
                    result = st.session_state.bot.query_huggingface(selected_model, prompt, max_tokens=800, temperature=0.8)

                    # handle errors returned as strings
                    if isinstance(result, str) and result.startswith("Error:") or (isinstance(result, str) and result.startswith("API Error")):
                        st.error(result)
                    else:
                        # success
                        if selected_mode == "Resolution Drafting":
                            st.markdown('<div class="resolution-format">', unsafe_allow_html=True)
                            st.markdown(result)
                            st.markdown('</div>', unsafe_allow_html=True)
                        elif selected_mode == "Debate Speech":
                            st.markdown('<div class="diplomatic-response">', unsafe_allow_html=True)
                            st.markdown(result)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(result)

                        st.caption(f"Mode: {selected_mode} | Country: {selected_country}")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result,
                            "mode": selected_mode,
                            "country": selected_country,
                            "committee": committee,
                            "timestamp": datetime.utcnow().isoformat(),
                        })

    # Right column: dashboard and utilities
    with col2:
        st.subheader("📊 Session")
        total = len(st.session_state.messages)
        st.metric("Total Interactions", total)
        if total:
            user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
            bot_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])
            st.metric("Requests", user_msgs)
            st.metric("AI Responses", bot_msgs)

        st.subheader("💡 Quick Tips")
        tips = [
            "Research your country's historical positions",
            "Use formal diplomatic language",
            "Build alliances before committee sessions",
            "Prepare clear red-lines and compromises",
        ]
        for t in tips:
            st.write(f"• {t}")

        st.subheader("📥 Export")
        if st.button("Export chat as TXT"):
            txt = []
            for m in st.session_state.messages:
                ts = m.get("timestamp", "")
                role = m.get("role")
                content = m.get("content")
                txt.append(f"[{role}] {ts}\n{content}\n\n")
            txt_blob = "".join(txt)
            st.download_button("Download TXT", data=txt_blob, file_name="mun_chat_history.txt", mime="text/plain")

        st.subheader("ℹ️ Committee Info")
        st.write(st.session_state.bot.committees.get(committee, committee))

    # Footer
    st.markdown("---")
    cols = st.columns(3)
    cols[0].markdown("🌍 **MUN AI Assistant**")
    cols[1].markdown("📚 **Resources**")
    cols[1].markdown("• https://www.un.org")
    cols[2].markdown("🔗 **Links**")
    cols[2].markdown("• https://huggingface.co/settings/tokens")


if __name__ == "__main__":
    main()
