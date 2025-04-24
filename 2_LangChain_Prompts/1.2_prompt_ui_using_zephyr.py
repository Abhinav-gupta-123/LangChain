"""In this we will use different model for just testing the quality of content we can use any small
 model also for leanring purpose """



from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt

# Initialize Ollama LLaMA 3.2 model
llms =HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100,
    )
)
model=ChatHuggingFace(llm=llms)

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
    chain = template | model 

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