import streamlit as st
import pandas as pd # Added for graphing
from core.generator import LLMClient
from core.detector import EnsembleJudge, LocalDetector, APIDetector
from core.loop_manager import HumanizerLoop

st.set_page_config(page_title="AI Text Humanizer", layout="wide")

st.set_page_config(page_title="AI Text Humanizer", layout="wide")

st.title("🤖 AI Text Humanizer")
st.markdown("Transform robotic AI text into human-like writing through an iterative feedback loop.")

# Sidebar Configuration
with st.sidebar:
    st.header("Settings")
    llm_model = st.text_input("Local LLM Model (LM Studio)", value="lmstudio-community/gemma-4-31b-it-uncensored-heretic")
    threshold = st.slider("Humanity Threshold (%)", 0, 100, 80)
    max_iters = st.slider("Max Iterations", 1, 10, 5)
    temp_start = st.slider("Starting Temperature", 0.0, 2.0, 0.7)
    
    st.divider()
    st.subheader("Persona Settings")
    persona = st.selectbox("Writing Persona", ["Graduate Student", "Hyper Casual"], index=0)
    persona_map = {"Graduate Student": "professional", "Hyper Casual": "casual"}
    selected_persona = persona_map[persona]

    st.divider()
    st.subheader("Optional API Detectors")
    api_key = st.text_input("API Key", type="password")
    provider = st.selectbox("Provider", ["GPTZero", "Originality.ai", "Other"])

# Initialize components
llm_client = LLMClient(model=llm_model)
local_det = LocalDetector()
api_dets = []
if api_key:
    api_dets.append(APIDetector(api_key=api_key, provider=provider))

judge = EnsembleJudge(local_detector=local_det, api_detectors=api_dets)
humanizer = HumanizerLoop(llm_client, judge)

# Main UI
col1, col2 = st.columns(2)

with col1:
    input_text = st.text_area("Input AI Text", height=300, placeholder="Paste your AI-generated text here...")

if st.button("Humanize Text"):
    if not input_text:
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Iterating to improve humanity score..."):
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # We manually run the loop here to update UI per iteration
            current_text = input_text
            final_score = 0
            iteration = 0
            scores_history = [] # Track scores for graphing
            
            while iteration < max_iters:
                score, _ = judge.analyze(current_text)
                status_text.text(f"Iteration {iteration}: Humanity Score = {score:.2f}%")
                progress_bar.progress((iteration / max_iters))
                
                # Store score for the graph
                scores_history.append(score)
                
                if score >= threshold:
                    final_score = score
                    break
                
                current_text = llm_client.humanize_pipeline(current_text, persona=selected_persona, temperature=temp_start + (iteration * 0.1))
                iteration += 1
                final_score = score

            progress_bar.empty()
            status_text.empty()
            
            with col2:
                st.subheader("Humanized Result")
                st.write(current_text)
                st.metric("Final Humanity Score", f"{final_score:.2f}%")
                if final_score >= threshold:
                    st.success("Threshold reached!")
                else:
                    st.info("Max iterations reached without hitting threshold.")

                # Display the progress graph
                if len(scores_history) > 0:
                    st.subheader("Humanization Progress")
                    chart_data = pd.DataFrame({
                        "Iteration": [f"Iter {i}" for i in range(len(scores_history))],
                        "Score": scores_history
                    })
                    st.bar_chart(data=chart_data, x="Iteration", y="Score")

                st.subheader("Original vs Humanized")
                st.text_area("Original", value=input_text, height=150, disabled=True)
                st.text_area("Humanized", value=current_text, height=150, disabled=True)
