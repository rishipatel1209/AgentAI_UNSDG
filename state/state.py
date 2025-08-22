from typing import Annotated, List,Dict
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class StateVector(TypedDict):
    seed_question: str
    country: str
    messages: Annotated[List[AnyMessage], add_messages] 
    topic: List[tuple[int, str]]  # List of tuples (topic_num, topic_name)
    topic_kw: Dict[str, List[str]] #  # Dictionary mapping topic names to lists of keywords
    questions: List[str] 