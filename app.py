import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama


import os 
from dotenv import load_dotenv
load_dotenv()


# langsmith tracking 

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = "Q&A-project chatbot with openai "


# prompt template

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based on the context provided."),
    ("user", "Question: {question}")
])

# here the below function : interact with the openai with the user query 
# here engine/llm: which model we are using . 
#temperature is the value which lies between 0 and 1 : higher the value , more creative the llm should be . 

def generate_answer(question,engine,temperature,max_tokens):
    llm = Ollama(
    model=engine,
    temperature=temperature,
    num_predict=max_tokens
    )
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question': question})
    return answer 

# title of the app 
st.title("Enhanced Q&A Chatbot with the openai")


llm = st.sidebar.selectbox("select an open ai model",["qwen2.5:32b-instruct","qwen3.5:27b","qwen3.5:9b"])

temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

# main interface for user 

st.write("go ahead and ask any question")
user_input=st.text_input("you:")

if user_input:
    response = generate_answer(user_input,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("please mention the query ")