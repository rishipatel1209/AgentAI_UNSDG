from argparse import ArgumentParser

import sys
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from constants import *
import os
import pandas as pd 
import re
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START, END

from typing import Annotated, List,Dict
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from langchain_core.output_parsers import NumberedListOutputParser
from langgraph.graph.state import CompiledStateGraph

from IPython.display import Image, display

parser = ArgumentParser()
parser.add_argument("--prompt", type=str, help="Raw prompt for graph input")
parser.add_argument("--country", type=str, default="India", help="Country for the analysis")

openai_api_key=os.environ['OPENAI_API_KEY']
llm = ChatOpenAI(
            temperature=0.1,
            model_name="gpt-4",
            openai_api_key=openai_api_key
        )
class StateVector(TypedDict):
    seed_question: str
    country: str
    messages: Annotated[List[AnyMessage], add_messages] 
    topic: List[tuple[int, str]]  # List of tuples (topic_num, topic_name)
    topic_kw: Dict[str, List[str]] #  # Dictionary mapping topic names to lists of keywords
    questions: List[str] 

model_directory='train_bert/model'
loaded_tokenizer = DistilBertTokenizerFast.from_pretrained(model_directory)
loaded_model = TFDistilBertForSequenceClassification.from_pretrained(model_directory)
df_keys=pd.read_csv('train_bert/training_data/Keyword_Patterns.csv')

def create_prompt_template(state: StateVector) -> StateVector:
    """
    Creates a prompt template based on the state vector.
    """
    state['messages'].extend([
            SystemMessage(
                content="You are an AI assistant that helps users find information about the Sustainable Development Goals (SDGs)."
            ),
            AIMessage(
                content="Hello, I am an assistant designed to help you learn about the 17 UN SDG goals listed here: https://sdgs.un.org/goals.\
                    You can ask me about any of the goals or specific topics, and I will provide you with information and resources related to them.\
                    I can also help you create a question related to the SDGs based on your input.\
                    Please provide me with a topic or question related to the SDGs and select a country, and I will do my best to assist you."
            )
        ])
    for topic, keywords in state['topic_kw'].items():
        state['messages'].append(SystemMessage(content=f"For the UN SDG Goal: {topic}\n. \
                                               Use the following keywords : {', '.join(keywords)}. Generate questions related to the topic in the country of {state['country']} using these keywords."))
    state['messages'].append(AIMessage(content="Based on the provided information, here is an enhanced version of the question:"))

    return state


#Check input raw prompt and extract topics and keywords
def check_inputs(state: StateVector) -> StateVector:
    """Check if topic and keywords are set"""
    #print(state)
    if not state.get('seed_question') or len(state.get('seed_question').strip())<3:
        raise ValueError("Seed question is not set in the state vector.")
    predict_input = loaded_tokenizer.encode(
        text=state.get('seed_question').lower(),
        truncation=True,
        padding=True,
        return_tensors="tf")
    output = loaded_model(predict_input)[0]
    prediction_value = tf.argmax(output, axis=1).numpy()#All answers
    prob_value=tf.nn.softmax(output).numpy()[0]#Probability of TF output
    Topic_Bool=prob_value>0.4
    Topics=[]
    Keywords={}
    for index, key in enumerate(sdg_goals):
        if not Topic_Bool[index]:continue
        #print(sdg_goals[key])
        Topics.append((index+1,sdg_goals[key]))
    for i,t in Topics:
        kw_patterns=df_keys[df_keys['topic_num']==i]['keywords'].values[0].split(',')
        Keywords[t] = re.findall(r'%s' %("|".join(kw_patterns)),raw_prompt)
        if not Keywords[t]:
            Keywords[t] = kw_patterns
            state['messages'].append(AIMessage(content="Will add keywords for the topic: %s" % t ))
        #print( Keywords[t])

    state['topic'] = Topics
    state['topic_kw'] = Keywords
    if not state.get('country'):
        state['messages'].append(AIMessage(content="Country is not set. Please provide a country."))
        return state
    elif not state.get('topic'):
        state['messages'].append(AIMessage(content="Missing topic please ask a question about the 17 Sustainable Development Goals. Graph will terminate."))
    state['messages'].append(AIMessage(content="Topics are: %s and keywords found: %s. Proceeding to prompt creation." \
                                    %(", ".join(Keywords.keys()), ", ".join([kw for kws in Keywords.values() for kw in kws]))))
    return state    

def should_continue(state: StateVector) -> str:
    """Determine whether to continue to prompt creation or terminate"""
    if not state.get('topic') or not state.get('topic_kw'):
        return "terminate"
    return "create_prompt_template"

def generate_questions(state: StateVector) -> StateVector:
    """
    Generates questions based on the provided topics and keywords.
    This is a placeholder function that can be extended to include more complex question generation logic.
    """
    parser=NumberedListOutputParser()
    runner= llm | parser
    #template= ChatPromptTemplate.from_messages(state['messages'][-2])
    result = runner.invoke(state['messages'])

    print("Generated Question: %s" %result)
    state['questions'] = result
    state['messages'].append(AIMessage(content="Generated questions: %s" % "\n".join(state['questions'])))
    return state

def create_graph() -> CompiledStateGraph:

    """
    Creates a state graph for the SDG analysis.
    """
    workflow = StateGraph(StateVector)
    # Add nodes
    workflow.add_node("check_inputs", check_inputs)
    workflow.add_node("create_prompt_template", create_prompt_template)
    workflow.add_node("generate_questions", generate_questions)
    # Set entry point
    workflow.set_entry_point("check_inputs")
    # Add conditional edges
    
    workflow.add_conditional_edges(
        "check_inputs",
        should_continue,
        {
            "create_prompt_template": "create_prompt_template",
            "terminate": END
        }
    )
    workflow.add_edge("create_prompt_template", "generate_questions")
    workflow.add_edge("generate_questions", END)

    return workflow.compile()

if __name__ == "__main__":
    # Create the graph
    args = parser.parse_args()

    raw_prompt=sys.argv[1] #replace with user arguments or input
    print("RAW PROMPT: %s" %args.prompt)
    graph = create_graph()
    display(Image(graph.get_graph().draw_mermaid_png(output_file_path="static/question_refining.png")))
    result=graph.invoke({'country': args.country, 'seed_question': raw_prompt })
    print(result['questions'])

