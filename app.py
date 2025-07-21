import configparser
import streamlit as st
from typing import List, Optional

from streamlitui.constants import unsdg_countries
class StreamlitConfigUI:

    """
    A Streamlit UI class that uses ConfigParser to load settings from an INI file.
    """
    
    def __init__(self, config_file: str = "./streamlitui/uiconfigfile.ini"):
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
    def autocomplete_selectbox_only(self) -> Optional[str]:
        """
        Method 2: Simple selectbox with searchable options.
        Streamlit's selectbox has built-in search functionality.
        """
        st.subheader("Searchable Box for Country")
        
        selected_country = st.selectbox(
            "Choose a country (start typing to search):",
            options=[""] + self.unsdg_countries,
            key="select_method2"
        )
        
        if selected_country:
            st.success(f"Selected: **{selected_country}**")
            return selected_country
            
        return None
    def load_streamlit_ui(self):
        st.set_page_config(page_title= "🤖 " + self.config.get_page_title(), layout="wide")
        st.header("🤖 " + self.config.get_page_title())

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
        return self.user_controls

if __name__=='__main__':
    ui=LoadStreamlitUI()
    user_input=ui.load_streamlit_ui()
    if not user_input:
        st.error("Error: Failed to load user input from the UI.")
    user_message = st.chat_input("Enter your message:")
