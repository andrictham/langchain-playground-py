import os
import streamlit as st

from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.title('LangChain Sequential Chain Playground')

user_question = st.text_input('What is your question?')

llm = OpenAI(streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0.0, openai_api_key=OPENAI_API_KEY)

if st.button("Tell me about it", type="primary"):
    # Chain 1
    template = """{question}\n\n"""
    prompt_template = PromptTemplate(input_variables=["question"], template=template)
    question_chain = LLMChain(llm=llm, prompt=prompt_template)
    st.subheader("Chain 1")
    st.info(question_chain.run(user_question))
    # # Chain 2
    # template = """Here is a statement: 
    # {statement}
    # Make a bulleted list of the assumptions you made when producing the above statement.\n\n
    # """
    # prompt_template = PromptTemplate(input_variables=["statement"], template=template)