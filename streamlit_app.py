import configparser
import altair as alt
import streamlit as st
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.messages import AnyMessage, AIMessage,SystemMessage, HumanMessage,AIMessageChunk

from streamlitui.constants import unsdg_countries
from llm.llm_setup import ModelSelection
import pandas as pd
from state.state import StateVector
from graph.state_vector_nodes import question_model,research_model
from graph.graph_builder import BuildGraphOptions
import re
import os
import torch
device=torch.get_default_device()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
class StreamlitConfigUI:

    """
    A Streamlit UI class that uses ConfigParser to load settings from an INI file.
    """

    def __init__(self, config_file: str = "streamlitui/uiconfigfile.ini"):
        """
        Initialize the UI class with configuration file.

        Args:
            config_file (str): Path to the INI configuration file
        """
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
    def get_llm_options(self):
           return self.config["DEFAULT"].get("LLM_OPTIONS").split(",")
    def get_page_title(self):
           return self.config["DEFAULT"].get("PAGE_TITLE")
    def get_usecase_options(self):
        return self.config["DEFAULT"].get("USE_CASE_OPTIONS").split(",")


class LoadStreamlitUI:
    def __init__(self):
        self.config=StreamlitConfigUI()
        self.user_controls={}
        self.unsdg_countries=unsdg_countries
    def filter_countries(self, query: str) -> List[str]:
        """
        Filter countries based on the query string.

        Args:
            query (str): The search query

        Returns:
            List[str]: Filtered list of countries
        """
        if not query:
            return self.unsdg_countries

        query_lower = query.lower()
        return [
            country for country in self.unsdg_countries
            if query_lower in country.lower()
        ]

    def load_streamlit_ui(self):
        st.set_page_config(page_title= "🇺🇳 " + self.config.get_page_title(), layout="wide")
        st.header("🇺🇳 " + self.config.get_page_title())

        with st.sidebar:
            # Get options from config
            llm_options = self.config.get_llm_options()
            usecase_options = self.config.get_usecase_options()

            # LLM selection
            self.user_controls["selected_llm"] = st.selectbox("Select LLM", llm_options)
            self.user_controls["GENAI_API_KEY"] = st.session_state["GENAI_API_KEY"]=st.text_input("API Key",type="password")
            if not self.user_controls["GENAI_API_KEY"]:
                st.warning("⚠️ Please enter a Gemini or Open AI ChatGPT API key to proceed. Don't have? refer : https://platform.openai.com/api-keys or https://aistudio.google.com/welcome?gclsrc=aw.ds&gad_source=1&gad_campaignid=21521909442 ")
            self.user_controls["selected_usecase"]=st.selectbox("Select Usecases",usecase_options)
            self.user_controls['UN SDG Country']= st.selectbox("Choose a country (start typing to search):",options=[""] + self.unsdg_countries)
            if self.user_controls['selected_usecase']=='DeepRishSearch':
                self.user_controls["TAVILY_API_KEY"] = st.session_state["TAVILY_API_KEY"]=st.text_input("Tavily API Key",type="password")
        return self.user_controls
def display_result_on_ui( state_graph,mode="Question Refining Mode:"):
	if mode =="Question Refining Mode:":
			for event in graph.stream({'messages':("user",user_message)}):
				print(event.values())
				for value in event.values():
					print(value['messages'])
					with st.chat_message("user"):
						st.write(user_message)
					with st.chat_message("SDG AI Assistant"):
						st.write(value["messages"].content)

if __name__=='__main__':
    ui=LoadStreamlitUI()
    user_input=ui.load_streamlit_ui()
    LLM_Selection=ModelSelection(user_input)
    if user_input["GENAI_API_KEY"]:llm=LLM_Selection.setup_llm_model()
    loaded_tokenizer = AutoTokenizer.from_pretrained('train_bert/topic_classifier_model')
    loaded_model = AutoModelForSequenceClassification.from_pretrained('train_bert/topic_classifier_model',device_map=device)
    df_keys=pd.read_csv('train_bert/training_data/Keyword_Patterns.csv')

    if not user_input:
        st.error("Error: Failed to load user input from the UI.")
    #Input prompt
    user_message = st.chat_input("Ask me a question or give me keywords to kick off a query about a UN SDG Goal:")

    if user_message and user_input['UN SDG Country']:
        state=StateVector(country=user_input['UN SDG Country'], seed_question=user_message, messages=[])
        if user_input['selected_usecase']=='AskSmart SDG Assistant':
            SmartQuestions=question_model(#StateVector=state,
                                    loaded_tokenizer=loaded_tokenizer,
                                    loaded_model=loaded_model, llm=llm, df_keys=df_keys)
            builder=BuildGraphOptions(SmartQuestions)
            graph=builder.build_question_graph()
            with st.chat_message("assistant"):
                intro="Hello, I am an assistant designed to help you learn about the 17 UN SDG goals listed here: https://sdgs.un.org/goals.\
                            You can ask me about any of the goals or specific topics, and I will provide you with information and resources related to them.\
                            I can also help you create a question related to the SDGs based on your input.\
                            Please provide me with a topic or question related to the SDGs and select a country, and I will do my best to assist you." #,
                st.write(intro)
            initial_input = {'country': user_input['UN SDG Country'], 'seed_question': user_message}
            with st.chat_message("user"):st.write(user_message)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                accumulated_content = ""
                for chunk in graph.stream(initial_input, stream_mode='messages'):
                    message, meta=chunk
                    if isinstance(message, AIMessage):
                            accumulated_content += message.content
                            message_placeholder.write(accumulated_content)
        elif user_input['selected_usecase']=='DeepRishSearch':
            print("Deep Rishsearch")
            SmartQuestions=question_model(#StateVector=state,
                                    loaded_tokenizer=loaded_tokenizer,
                                    loaded_model=loaded_model, llm=llm, df_keys=df_keys)
            builder=BuildGraphOptions(SmartQuestions)
            ResearchModel=research_model(llm=SmartQuestions.genai_model, tavily_api_key=user_input['TAVILY_API_KEY'])
            graph=builder.build_research_graph(ResearchModel)
            with st.chat_message("assistant"):
                intro="Hello, I am an assistant designed to help you learn about the 17 UN SDG goals listed here: https://sdgs.un.org/goals.\
                            You can ask me about any of the goals or specific topics, and I will provide you with information and resources related to them.\
                            I can also help you create a question related to the SDGs based on your input.\
                            Please provide me with a topic or question related to the SDGs and select a country, and I will do my best to assist you." #,
                st.write(intro)
            with st.chat_message("user"):st.write(user_message)
            initial_input = StateVector({'country': user_input['UN SDG Country'], 'seed_question': user_message})
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                accumulated_content = ""
                for chunk in graph.stream(initial_input, stream_mode='messages'):
                    message, meta=chunk
                    if isinstance(message, AIMessage):
                            accumulated_content += message.content
                            message_placeholder.write(accumulated_content)
