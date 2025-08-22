import os 
import streamlit as sl 
from langchain_openai import ChatOpenAI

from langchain_google_genai import ChatGoogleGenerativeAI
class ModelSelection:
    def __init__(self,user_contols_input):
        self.user_controls_input=user_contols_input
    def setup_llm_model(self):
        selected_model=self.user_controls_input["selected_llm"]
        if not selected_model:
            st.error("Please select a Gen AI provider")
        elif not self.user_controls_input["GENAI_API_KEY"]:
            st.error("Please Include a GenAI API Key")

        gen_api_key=self.user_controls_input["GENAI_API_KEY"]
        if  selected_model=='OpenAI':      
            try: 
                llm = ChatOpenAI(
                temperature=0.1,
                model_name="gpt-4o-mini",
                openai_api_key=gen_api_key
            )
                return llm
            except Exception as e:
                raise ValueError(f"Error Ocuured With Exception : {e}")
        else: 
            llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash',google_api_key=gen_api_key,temperature=0.1)
            return llm


        