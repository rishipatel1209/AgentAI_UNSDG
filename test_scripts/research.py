#React Agent for Research and Development
from curses import wrapper
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.messages import AnyMessage,SystemMessage,AIMessage,HumanMessage,ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from newsapi import NewsApiClient
from owid import catalog
from typing import Annotated,List, Dict
from typing_extensions import TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults
from datetime import datetime, timedelta
import pandas as pd
import ast
from IPython.display import Image, display
tavily_api_key=os.environ.get('TAVILY_API_KEY')
openai_api_key=os.environ['OPENAI_API_KEY']
google_api_key=os.environ['GOOGLE_GEMINI_API']


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error

#from langchain.agents import create_tool_calling_agent, AgentExecutor
#from langchain.agents import create_react_agent
questionsfrommessages = "1.How is Moldova implementing the Dakar Framework to ensure sustainable education for all its citizens?\n \
2. What initiatives are in place in Moldova to promote access to quality education, particularly in rural areas?\n \
3. How does the HESI partnership contribute to improving higher education in Moldova?\n \
4. In what ways is Moldova addressing the challenges of education financing to enhance school infrastructure?\n \
5.    What role does the campus network play in promoting global education initiatives within Moldovan universities?\n \
6.    How can Moldova leverage sustainability initiatives to improve primary education outcomes?\n \
7.    What strategies are being employed in Moldova to increase school completion rates among disadvantaged populations?\n \
8.    How is the education summit influencing policy changes related to quality education in Moldova?\n \
9.    What measures are being taken to ensure that education in Moldova is not only accessible but also sustainable for future generations?\n \
10.    How does the slower pace of educational reform in Moldova impact the overall quality of education provided to students?\n \
"
tool_names = ["direct_semantic_scholar_query", "direct_tavily_search" ]



class State(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]
    country: str
    topic: List[tuple[int, str]]  # List of tuples (topic_num, topic_name)
    topic_kw: Dict[str, List[str]] #  # Dictionary mapping topic names to lists of keywords
    questions: List[str]
def direct_semantic_scholar_query(query: str):
    """Direct invocation of SemanticScholarQueryRun without agent"""
    
    # Create the tool directly
    tool = SemanticScholarQueryRun(
        api_wrapper=SemanticScholarAPIWrapper()
    )
    
    # Invoke the tool directly
    result = tool.invoke(query, k=10, output_parser=JsonOutputParser(), fields=["paperId","title","authors", "url","abstract","year","paperId"],sort="year")

    return result
def direct_tavily_search(query: str):
    """Direct invocation of TavilySearchResults without agent"""
    # Create the tool directly
    tavily = TavilySearchResults()
    result = tavily.invoke(query, max_results=5, include_answer=True, include_snippet=True, include_source=True)
    response=""
    for r in result:
        response +="Found a webpage: %s at %s" %(r['title'], r['url'])
        response +="Summary of the page: %s" %r['content']
        response +="Relevance score: %s" %r['score']
    return response
def direct_newsapi_search(query: str):
    """Direct invocation of newsapi without agent"""
    # Create the tool directly
    today = datetime.now().date()
    thirty_days_ago = today - timedelta(days=29)
    newsapi = NewsApiClient(api_key=os.environ['NEWSAPI_KEY'])
    all_articles = newsapi.get_everything(q=query,from_param=thirty_days_ago)
    article_string= ""
    for article in all_articles['articles']:
        article_string += f"In the article titled: {article['title']} "
        article_string += f"Published on {article['publishedAt']} "
        article_string += f"Sourced from {article['source']['name']} at {article['url']}"
        article_string += f"by Authors {article['author']}\n"
        article_string += f"The news reads: {article['description']}\n"
    # Invoke the tool directly
    return article_string
#def create_prompt_template(questions: str, topic='End poverty in all its forms everywhere') -> ChatPromptTemplate:
def data_analysis(state:State):
    df_analyst=pd.read_csv('data_analysis/data_analyst_prompts.csv')
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


def create_prompt_template(state:State) -> ChatPromptTemplate:

    """
    Creates a prompt template based on the provided questions.
    """
    topic_string = ", ".join(f"{name}" for num, name in state['topic'])
    keywords=[]
    kw_string=''
    for i,v in state['topic_kw'].items():
        keywords.append(",".join(v))
    kw_string += f" with keywords: {', '.join(keywords)}"
    messages = [
            SystemMessage(content= f"You are an AI assistant that helps users find information about the Sustainable Development Goal: {topic_string}.\
                          Your task is to answer questions related to this goal using the provided tools with toolNames: {tool_names}\
                              You will be provided with a list of questions to answer below: \
                              questions = {state["questions"]} "),

            #AIMessage(content="Using publications on Semantic Scholar and my own reference data, I will answer the questions related to the Sustainable Development Goal: %s." % topic),
            SystemMessage(content=f"Search for recent papers on {kw_string} in {state['country']}."),
            SystemMessage(content=f"Search for recent news on {kw_string} in {state['country']}."),
            SystemMessage(content=f"Search the internet for webpages on {kw_string} in {state['country']}."),
            #HumanMessage(content="Please provide a comprehensive answer to the questions based on the information gathered from the tools.")
        ]
    state['messages'] = messages
    return state

def tool_calling_agent():
    """Show how to bind the tool to LLM using tool calling"""
    
    # Initialize LLM
    llm = ChatOpenAI(
        temperature=0.1,
        model_name="gpt-4o-mini",
        openai_api_key=openai_api_key
    )
    # Create the tool
    semantic_scholar_tool = SemanticScholarQueryRun(
        api_wrapper=SemanticScholarAPIWrapper()
    )
    tools=[semantic_scholar_tool,direct_tavily_search]
    # Bind the tool to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    return llm_with_tools,tools


def tool_calling_llm(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

def summary_answer(state:State)->State:
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
    analysis_prompt=data_analysis(state)
    print(analysis_prompt)
    initial_system_message.content+=analysis_prompt
    initial_system_message.content+="\n Assess if the resources indicate a general positive or negative trend and grade progress\
          from 0-10 where 0 is very negative and 10 is very positive.\n"
    initial_system_message.content+="\n Provide detailed answers to the questions and a list of references used."
    state["messages"].append(initial_system_message)
    '''
    llm = ChatOpenAI(
        temperature=0.4,
        model_name="gpt-4o",
        openai_api_key=openai_api_key
    )
    '''
    llm=ChatGoogleGenerativeAI(model='gemini-2.5-pro',google_api_key=google_api_key,temperature=0.3)
    
    #print(state["messages"][-1].content)
    airesponse = llm.invoke(state["messages"][-1].content)
    # For simplicity, we just return the messages as they are
    return {"messages": [airesponse]}
#def main():

    #print(r['title'], r['description'], r['url'], r['publishedAt']) #This is the content of the message, which is
    #m.pretty_print() #This information needs to be collected from URL's, factchecked and summarized
#    print(r)
    #print("Found a webpage: %s at %s" %(r['title'], r['url']))
    #print("Summary of the page: %s" %r['content'])
    #print("Relevance score: %s" %r['score'])

    #print(r['title'], r['content'], r['score'], r['url'])
#print(response, type(response))

openai_api_key = os.environ.get('OPENAI_API_KEY')# With binded tools this can be an open source small LLM (miniLLM, Llama3, etc.)
if not openai_api_key:
    raise ValueError("OpenAI API key is not set in the environment variables.")

llm_with_tools,tools=tool_calling_agent()#Runnable LLM with tools bound to it and these need to be global variables
#print("LLM with tools: %s" %tools)
    #print(type(llm_with_tools))
    #These would then be the stages in the graph:
    #This is then the call function
state = State({'messages': [],
    'country': 'Moldova',
    'topic': [(4, 'Quality Education')],
    'topic_kw': {'Quality Education': ['education sustainable', 'education life', 'access education']},
    'questions': questionsfrommessages.split('\n')})
state = create_prompt_template(state)
#print(state)#Used to init the state vector

builder = StateGraph(State)
builder.add_node(
    tool_calling_llm,
    name="tool_calling_llm",
    description="Invoke the LLM with curated questions to answer.",
)
builder.add_node("tools", ToolNode(tools))

builder.add_node(
    summary_answer,
    name="summary_answer",
    description="Summarize the answer from the LLM using the information gathered from the tools.",
)

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to Summary answer with no retrieved docs
    tools_condition,
)
#builder.add_edge("tools",END)
builder.add_edge("tools", "summary_answer")
builder.add_edge("summary_answer", END)
graph = builder.compile()
#display(Image(graph.get_graph().draw_mermaid_png(output_file_path="static/research_collection.png")))


#state = State({'messages':prompt})
#print(state)

messages= graph.invoke(state)

#analysis_prompt=data_analysis(state)
#print(analysis_prompt)
for m in messages['messages']:
    print(m.content)
    #print(m.content) #This is the content of the message, which is
    #m.pretty_print() #This information needs to be collected from URL's, factchecked and summarized


'''
results=llm_with_tools.invoke([SystemMessage(content=f"Search for recent papers on poverty eradication in India.")]) #Should hit semantic scholar tool
print(results)
results=llm_with_tools.invoke([SystemMessage(content=f"Search for recent news on poverty eradication in India.")]) #Should hit News API tool
print(results)
results=llm_with_tools.invoke([SystemMessage(content=f"Search the internet for webpages on poverty eradication in India.")]) #Should hit DuckDuckGo Search tool
print(results)
'''
    #Search for data analytics and projections for poverty eradication in India
#if __name__ == "__main__":
#    main()