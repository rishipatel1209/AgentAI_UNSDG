from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph,START, END
from state.state import StateVector
from graph.state_vector_nodes import question_model,research_model
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode

class BuildGraphOptions():
    def __init__(self,question_model):
        self.graph_builder=StateGraph(StateVector) #For invocation
        self.questionmodel=question_model
    def load_research_model(self,research_model):
        self.research_model=research_model
    def build_question_graph(self):
        #workflow = self.graph_builder

        self.graph_builder.add_node("check_inputs", self.questionmodel.check_inputs)
        self.graph_builder.add_node("create_question_prompt_template", self.questionmodel.create_question_prompt_template)
        self.graph_builder.add_node("generate_questions", self.questionmodel.generate_questions)
        # Set entry point
        self.graph_builder.set_entry_point("check_inputs")
        # Add conditional edges
    
        self.graph_builder.add_conditional_edges(
            "check_inputs",
            self.questionmodel.should_continue,
            {
                "create_question_prompt_template": "create_question_prompt_template",
                "terminate": END
            }
        )
        self.graph_builder.add_edge("create_question_prompt_template", "generate_questions")
        self.graph_builder.add_edge("generate_questions", END)
        return self.graph_builder.compile()
    def build_research_graph(self,research_model:research_model):
        self.graph_builder.add_node("check_inputs", self.questionmodel.check_inputs)
        self.graph_builder.add_node("create_question_prompt_template", self.questionmodel.create_question_prompt_template)
        self.graph_builder.add_node("generate_questions", self.questionmodel.generate_questions)
        self.graph_builder.add_node("create_prompt_template", research_model.create_prompt_template)

        # Set entry point
        self.graph_builder.set_entry_point("check_inputs")
        # Add conditional edges
    
        self.graph_builder.add_conditional_edges(
            "check_inputs",
            self.questionmodel.should_continue,
            {
                "create_question_prompt_template": "create_question_prompt_template",
                "terminate": END
            }
        )
        self.graph_builder.add_edge("create_question_prompt_template", "generate_questions")
        self.graph_builder.add_node(research_model.tool_calling_llm,
            name="tool_calling_llm",
            description="Invoke the LLM with curated questions to answer.",)
        #llm_with_tools,tools=research_model.tool_calling_agent()
        self.graph_builder.add_node("tools", ToolNode(research_model.tools))
        self.graph_builder.add_node(research_model.summary_answer,name="summary_answer",
            description="Summarize the answer from the LLM using the information gathered from the tools.",)
        self.graph_builder.add_edge("generate_questions","create_prompt_template")
        self.graph_builder.add_edge("create_prompt_template","tool_calling_llm")

        self.graph_builder.add_conditional_edges(
                "tool_calling_llm",
                # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
                # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to Summary answer with no retrieved docs
                    tools_condition,
                {
                    "tools": "tools",
                    "summary_answer": "summary_answer",
                    # You might also want to handle other potential returns
                },
            )
        self.graph_builder.add_edge("tools", "summary_answer")
        self.graph_builder.add_edge("summary_answer", END)
        return self.graph_builder.compile()
