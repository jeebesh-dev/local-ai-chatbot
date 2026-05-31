import os
import json
import uuid
from typing import Dict, Generator, List, TYPE_CHECKING, Optional

import streamlit as st  # type: ignore[import]

#  Ollama Import Handling 
if TYPE_CHECKING:
    import ollama  # type: ignore[import]

try:
    import ollama  # type: ignore[import]
except Exception:                                      
    def _missing_ollama_chat(*args, **kwargs):
        raise RuntimeError(
            "The 'ollama' package is not installed or importable. Install it with 'pip install ollama' "
            "or configure OLLAMA_HOST to point to an Ollama server."
        )

    class _OllamaModule:                                
        chat = staticmethod(_missing_ollama_chat)          

    ollama = _OllamaModule()              

#  Constants 
DEFAULT_MODEL = os.getenv("MODEL_NAME", "tinyllama:latest")            
HISTORY_FILE = "chat_history.json"

#  History Management Functions 
def load_chat_history() -> Dict[str, List[Dict[str, str]]]:         
    if os.path.exists(HISTORY_FILE):                           
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:            
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_chat_history(history: Dict[str, List[Dict[str, str]]]):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")

#  Session Setup 
def ensure_session() -> None:                      
    # Initialize global settings
    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_MODEL
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.2
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = "You are a helpful assistant."

    # Initialize chat history structure
    if "all_chats" not in st.session_state:                  
        st.session_state.all_chats = load_chat_history()            

    # Ensure there is at least one chat and a current ID selected
    if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.all_chats:
        if not st.session_state.all_chats:
            # Create first chat if empty
            new_id = str(uuid.uuid4())                         
            st.session_state.all_chats[new_id] = []                   
            st.session_state.current_chat_id = new_id              
        else:
            # Select the most recent chat key (optional: sort by something else if needed)
            st.session_state.current_chat_id = list(st.session_state.all_chats.keys())[0]

def stream_chat_response(model: str, messages: List[Dict[str, str]], temperature: float) -> Generator[str, None, None]:             
    response_stream = ollama.chat(
        model=model,
        messages=messages,
        stream=True,
        options={
            "temperature": float(temperature),
        },
    )                                            
    for chunk in response_stream:                           
        part = chunk.get("message", {}).get("content", "")           
        if part:
            yield part

def main() -> None:
    st.set_page_config(page_title="Ollama Chat (tinyllama)", page_icon="💬")
    ensure_session()

    #  Sidebar: Settings & History 
    with st.sidebar:
        st.title("💬 Chat History")
        
        # "New Chat" Button
        if st.button("➕ New Chat", use_container_width=True):
            new_id = str(uuid.uuid4())
            st.session_state.all_chats[new_id] = []
            st.session_state.current_chat_id = new_id
            st.rerun()                

        # List existing chats
        st.markdown("---")
        st.write("Your Conversations:")
        
        # Iterate through chats to create buttons
        # We use reversed to show newest created (if dict preserves order) or just list them
        for chat_id, messages in list(st.session_state.all_chats.items()):
            # Generate a label based on the first user message, or "Empty Chat"
            label = "Empty Chat"
            for m in messages:
                if m["role"] == "user":
                    label = m["content"][:25] + "..." if len(m["content"]) > 25 else m["content"]
                    break
            
            # Highlight the current chat
            if chat_id == st.session_state.current_chat_id:
                st.markdown(f"👉 {label}")
            else:
                if st.button(label, key=chat_id):                      
                    st.session_state.current_chat_id = chat_id
                    st.rerun()

        st.markdown("---")                
        st.header("Settings")
        st.session_state.model = st.text_input("Model", value=st.session_state.model, help="Ollama model name, e.g. tinyllama:latest")
        st.session_state.temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=float(st.session_state.temperature), step=0.05)
        st.session_state.system_prompt = st.text_area("System prompt", value=st.session_state.system_prompt, height=120)
        st.caption("Backend: Ollama. Data saved to chat_history.json")

    #  Main Chat Area 
    st.title("💬 Chat with tinyllama")
    
    # Get the messages for the CURRENT active chat ID
    current_chat_id = st.session_state.current_chat_id
    current_messages = st.session_state.all_chats[current_chat_id]

    # Display Chat Messages
    for m in current_messages:                      
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # User Input Processing
    user_input = st.chat_input("Type your message…")        
    if user_input:
        # 1. Append user message to current chat history
        current_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2. Prepare messages for Ollama
        sys_prompt = (st.session_state.system_prompt or "").strip()
        model_msgs: List[Dict[str, str]] = []
        if sys_prompt:
            model_msgs.append({"role": "system", "content": sys_prompt})
        model_msgs.extend(current_messages)

        # 3. Generate Assistant Response
        with st.chat_message("assistant"):                      
            response_placeholder = st.empty()               
            full_response = ""
            try:
                for token in stream_chat_response(
                    model=st.session_state.model,
                    messages=model_msgs,
                    temperature=float(st.session_state.temperature),
                ):
                    full_response += token
                    response_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"Error while contacting Ollama: {e}")
                return

        # 4. Append assistant response to history
        current_messages.append({"role": "assistant", "content": full_response})
        
        # 5. SAVE history to file
        save_chat_history(st.session_state.all_chats)


if __name__ == "__main__":
    main()