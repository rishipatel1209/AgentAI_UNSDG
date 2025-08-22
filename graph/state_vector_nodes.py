
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage,ToolMessage
from langchain_core.output_parsers import NumberedListOutputParser,JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from state.state import StateVector
from streamlitui.constants import *
import torch
import torch.nn.functional as F
#import tensorflow as tf
import re
from langchain_openai import ChatOpenAI
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper
from langchain_tavily import TavilySearch
import pandas as pd
import torch.nn.functional as F
import os
torch.classes.__path__ = []
class question_model:
    def __init__(self,loaded_tokenizer,loaded_model, llm, df_keys):
        #self.state=StateVector
        self.tokenizer=loaded_tokenizer
        self.distilbert_model=loaded_model
        self.genai_model=llm
        self.df_keys=df_keys
    def create_question_prompt_template(self, state:StateVector) -> StateVector:
        """
        Creates a prompt template based on the state vector.
        """
        state['messages'].extend([SystemMessage(
                    content="You are an AI assistant that helps users find information about the Sustainable Development Goals (SDGs)."
                )
            ])
        for topic, keywords in state['topic_kw'].items():
            state['messages'].append(SystemMessage(content=f"For the UN SDG Goal: {topic}\n. \
                                                Use the following keywords : {', '.join(keywords)}. Generate questions related to the topic in the country of {state['country']} using these keywords.\n"))
        state['messages'].append(AIMessage(content="Based on the provided information, here is an enhanced list of the question: \n"))

        return state

    #Check input raw prompt and extract topics and keywords
    def check_inputs(self,state:StateVector) -> StateVector:
        """Check if topic and keywords are set"""
        #print(state)
        if not state.get('seed_question') or len(state.get('seed_question').strip())<3:
            raise ValueError("Seed question is not set in the state vector.")
        #print(state.get('seed_question').lower())
        predict_input = self.tokenizer(
            text=state.get('seed_question').lower(),
            truncation=True,
            padding=True,
            return_tensors="pt")
        #print(predict_input)
        with torch.no_grad():
            logits = self.distilbert_model(**predict_input).logits
            prob_value=F.softmax(logits, dim=1).cpu().numpy()[0]
            Topic_Bool=prob_value>0.4
            Topics=[]
            Keywords={}
            for index, key in enumerate(sdg_goals):
                if not Topic_Bool[index]:continue
                #print(sdg_goals[key])
                Topics.append((index+1,sdg_goals[key]))
            #print(Topics)
            for i,t in Topics:
                kw_patterns=self.df_keys[self.df_keys['topic_num']==i]['keywords'].values[0].split(',')
                Keywords[t] = re.findall(r'%s' %("|".join(kw_patterns)),state['seed_question'])
                if not Keywords[t]:
                    Keywords[t] = kw_patterns
                    state['messages'].append(AIMessage(content="Will add keywords for the topic: %s \n" % t ))
            state['topic'] = Topics
            state['topic_kw'] = Keywords
            if not state.get('country'):
                state['messages'].append(AIMessage(content="Country is not set. Please provide a country. \n"))
                return state
            elif not state.get('topic'):
                state['messages'].append(AIMessage(content="Missing topic please ask a question about the 17 Sustainable Development Goals. Graph will terminate. \n"))
            state['messages'].append(AIMessage(content="Topics are: %s and keywords found: %s.\n Proceeding to prompt creation. \n" \
                                            %(", ".join(Keywords.keys()), ", ".join([kw for kws in Keywords.values() for kw in kws]))))
        return state    

    def should_continue(self, state:StateVector) -> str:
        """Determine whether to continue to prompt creation or terminate"""
        if not state.get('topic') or not state.get('topic_kw'):
            return "terminate"
        return "create_question_prompt_template"
    def generate_questions(self, state:StateVector) -> StateVector:
        """
        Generates questions based on the provided topics and keywords.
        This is a placeholder function that can be extended to include more complex question generation logic.
        """
        parser=NumberedListOutputParser()
        runner= self.genai_model | parser
        #template= ChatPromptTemplate.from_messages(state['messages'][-2])
        result = runner.invoke(state['messages'])

        #print("Generated Question: %s" %result)
        
        state['questions'] = result
        #ai_response="\n".join(state['questions'])
        #state['messages'].append(AIMessage(content="Generated questions: "+ai_response))
        return state

class research_model:
    def __init__(self,llm,tavily_api_key):
        self.llm=llm
        self.local_analysis_file='src/graph/data_analyst_prompts.csv'
        self.tool_names=["direct_semantic_scholar_query", "direct_tavily_search" ]
        semantic_scholar_tool = SemanticScholarQueryRun(
            api_wrapper=SemanticScholarAPIWrapper()
        )
        self.tools=[semantic_scholar_tool,self.direct_tavily_search]
        # Bind the tool to the LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        os.environ['TAVILY_API_KEY']=tavily_api_key

    def direct_semantic_scholar_query(self,query: str):

        """Direct invocation of SemanticScholarQueryRun without agent"""
        
        # Create the tool directly
        tool = SemanticScholarQueryRun(
            api_wrapper=SemanticScholarAPIWrapper()
        )
        
        # Invoke the tool directly
        result = tool.invoke(query, k=10, output_parser=JsonOutputParser(), fields=["paperId","title","authors", "url","abstract","year","paperId"],sort="year")

        return result

    def direct_tavily_search(self,query: str):
        """Direct invocation of TavilySearchResults without agent"""
        # Create the tool directly
        tavily = TavilySearch(max_results=5, include_answer=True, include_snippet=True, include_source=True)
        result = tavily.invoke(query)
        answer=result['answer']
        response=f"Summary Answer for all webpages: {answer} \n"
        for r in result['results']:
            response +="Found a webpage: %s at %s \n" %(r['title'], r['url'])
            response +="Summary of the page: %s \n" %r['content']
            response +="Relevance score: %s\n" %r['score']
        return response
    def data_analysis(self,state:StateVector):
        df_analyst=pd.read_csv(self.local_analysis_file)
        analysis_prompt=[]
        topics=state['topic']
        for t in topics:
            Goal_Number=t[0]
            df_analyst=df_analyst[df_analyst['country']==state['country']]
            df_analyst['goal_number']=df_analyst['goal_number'].astype(int)
            df_analyst=df_analyst[df_analyst['goal_number']==Goal_Number]
            #print(df_analyst.head())

            if df_analyst.shape[0]>0:
                analysis_prompt.extend(df_analyst['analysis_prompt'].to_list())
        return "\n".join(analysis_prompt)
    
    def create_prompt_template(self,state:StateVector) -> ChatPromptTemplate:
            """
            Creates a prompt template based on the provided questions.
            """
            topic_string = ", ".join(f"{name}" for num, name in state['topic'])
            keywords=[]
            kw_string=''
            for i,v in state['topic_kw'].items():
                keywords.append(",".join(v))
            kw_string += f" with keywords: {', '.join(keywords)}"
            questions=state["questions"]
            country=state['country']
            messages = [
                    SystemMessage(content= f"You are an AI assistant that helps users find information about the Sustainable Development Goal: {topic_string}.\
                                Your task is to answer questions related to this goal using the provided tools with toolNames: {self.tool_names}\
                                    You will be provided with a list of questions to answer below: \
                                    questions = {questions} "),

                    SystemMessage(content=f"Search for recent papers on {kw_string} in {country}."),
                    SystemMessage(content=f"Search the internet for webpages or news on {kw_string} in {country}."),
                ]
            state['messages'] = messages
            return state

    def tool_calling_agent(self):
        """Show how to bind the tool to LLM using tool calling"""
        
        # Initialize LLM
        '''
        llm = ChatOpenAI(
            temperature=0.1,
            model_name="gpt-4o-mini",
            openai_api_key=openai_api_key
        )
        '''
        # Create the tool
        semantic_scholar_tool = SemanticScholarQueryRun(
            api_wrapper=SemanticScholarAPIWrapper()
        )
        self.tools=[semantic_scholar_tool,self.direct_tavily_search]
        # Bind the tool to the LLM
        llm_with_tools = self.llm.bind_tools(self.tools)
        
        return llm_with_tools,self.tools


    def tool_calling_llm(self,state:StateVector):
        return {"messages":[self.llm_with_tools.invoke(state["messages"])]}

    def summary_answer(self,state:StateVector)->StateVector:
        """
        Function to summarize the answer from the LLM.
        This is a placeholder function that can be extended to include more complex summarization.
        """
        
        initial_system_message= state["messages"][0] # This is the system message that sets the context for the LLM with the listed questions
        initial_system_message.content += "Please provide a comprehensive answer to the questions. \n"
        
        tool_messages = [msg for msg in state["messages"] if isinstance(msg, ToolMessage)]
        augmented_data=""
        if tool_messages:
            initial_system_message.content += "Use the following information gathered from the tools as reference information: \n"
            
            for tool_msg in tool_messages:
                print(tool_msg.content, type(tool_msg.content))
                Label_Source=""
                if 'semanticscholar' in tool_msg.name.lower():
                    Label_Source="(Source: Scholarly Publication Abstracts from Semantic Scholar)"
                    augmented_data+= f"{tool_msg.content}\n"
                elif 'tavily' in tool_msg.name.lower():
                    Label_Source="(Source: News Search Results)"
                    augmented_data += f"{tool_msg.content}\n"
                else:
                    print("Unknown Tool Call")

                initial_system_message.content += f"{Label_Source} \n {augmented_data}\n"
        analysis_prompt=self.data_analysis(state)
        initial_system_message.content+=analysis_prompt
        initial_system_message.content+="\n Assess if the resources indicate a general positive or negative trend and grade progress\
            from 0-10 where 0 is very negative and 10 is very positive.\n"
        initial_system_message.content+="\n Provide detailed answers to the questions and a list of references used."
        print(initial_system_message.content)
        state["messages"].append(initial_system_message)
        airesponse = self.llm.invoke(state["messages"][-1].content)
        # For simplicity, we just return the messages as they are
        return {"messages": [airesponse]}