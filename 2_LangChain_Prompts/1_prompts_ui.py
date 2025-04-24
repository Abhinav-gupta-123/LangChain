from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt

# Initialize Ollama LLaMA 3.2 model
llm = ChatOllama(model="llama3.2")

template = load_prompt(r"C:\Users\abhin\Desktop\langchain\2_LangChain_Prompts\3_template.json")

# Streamlit implement
st.header('Research Summary Tool')

# selector
paper_input = st.selectbox(
    "Select Research Paper",
    ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers",
     "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"]
)

style_input = st.selectbox(
    "Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# submit button to submit the input to get the output
if st.button("Submit"):
    chain = template | llm 

    with st.spinner("Thinking"):
        try:
            result = chain.invoke({
                "paper_input": paper_input,
                "style_input": style_input,
                "length_input": length_input
            })
            st.markdown(result)

        except Exception as e:
            st.error(f"Error:{e}")