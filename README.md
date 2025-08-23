# AgentAI UNSDG

This project creates an application that allows to do Deep Research on the UN SDG goals.
It allows to use a free text of keywords to create a full report with a progress grade.
The application has two use cases:
1) Deep Research (DeepRishSearch)
2). AskSmart UN SDG Assistant.
The 2nd use case mainly generates questions,
but if you use a Gemini API key Gemini Pro 2.5 will answer the questions.

The application requires a Google or OpenAI API key,
and in the Deep Research mode also a key with Tavily Search.

For a detailed deep dive on the algorithm and testing check out this article on Medium:
[Minding the Context Gap in the development of AI Research Agents](https://generativeai.pub/minding-the-context-gap-in-the-development-of-ai-research-agents-78804ea016ce).

# Train DistilBert Model

This is done in the folder ``train_bert`` that has the main code and will store the model in a folder ``topic_classifier_model``. The model makes use of the distilbert_model and related tokenizer loaded from ``transformers``.

## Collect Webpage Training Data

``python PreProcess_TrainingData.py`` pulls down each individual goals page and related topic pages and creates document chunks and labels to train the model.
The code also creates a column with a list of keywords using KeyBert. This is useful for generating the curated file ``Keyword_Patterns.csv`` from the raw data in JSON ``KeywordPayload.json``. This is used to enhance questions via the selected keywords (passed in the prompt). All results from the script are stored in
``training_data``.

## Train and test the model

``python train_classifier.py `` then performs the training loop, produces a confusion matrix, and runs a unit test on an example test to produce a list of probabilities. The model output is stored in a folder called ``topic_classifier_model``. This model is called in the Agent after the initial user prompt to identify probabilistically, which topics the user would like questions for.

# Agent Code: Nodes and Edges
The main applications and functions are defined in the folders:
```
state
llm
graph
streamlitui
```
To test the two cases of functionality, use the ``test_scripts`` folder: ``PreprocessPrompt.py`` (For the question generation) and ``research.py``.

The folder also consists of the diagram for the Agent:

<img src="test_scripts/FullResearchAgent.png" title="Agent Workflow">

The state folder defines the state vector as a typed dictionary of fields. This state vector is transformed in each individual node, and is used to create the prompt for the Generative AI provider from a prompt template.

The llm folder consists of a simple LLM setup function that loads either:
* Gemini Pro 2.5 using the user specified Gemini Key
* OpenAI gpt-4o-mini using the user specified OpenAI key

You can consider even modifying the LLM setup code with a GROQ API key allowing to select multiple model types e.g. (Qwen, Llamma, Mistral, Phi etc )

## Question Model
The folder graph contains the node functions for the entire agent defined in two phases. The first model is the question model that generates questions from a user prompt. The listed nodes are in the functions:
1. ``check_inputs`` this is the entry point of the agent that is initiated by a user prompt. The user prompt is input to the DistilBert model to recognize the topics. The identified topics are used to pull keywords to input to the question enhancement prompt. The topics and keywords are stored in the state vector.
2. ``create_question_prompt_template`` creates the list of SystemMessages that define the AI role and the system prompt to enhance questions based on keywords.
3. ``generate_questions`` creates a runner from the llm model and parsing output as a numbered list. The questions are stored in the state vector.

The main edge is defined to continue to the end if there is no legitimate question on the UN SDG goals and this will END and return:
```
Missing topic please ask a question about the 17 Sustainable Development Goals. Graph will terminate.
```

## Research Model
The 2nd phase is the research data tool calls and building out the template that allows to answer the questions in detail.
1. ``create_prompt_template`` Builds the prompt template for answering the questions on the selected topics. Defining a new AI role to give detailed answers, and setting commands to pull from the available API tools.
2. ``tool_calling_llm``: The tool calling LLM performs the API requests for pulling news articles and semantic scholar publications. If no toll calls are performed the questions are passed forward without enhancing the prompt with additional information.
3. ``summary_answer``: This pulls in the questions from the state vector, the analysis data from the local stored file ``data_analyst_prompts.csv`` and all the available tool data in formatted text blocks. The prompt also includes a request to grade progress based on the available data.


# Run in Streamlit Container

## User Loop with Graph Stream:
```
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
```
Messages accumulate from the AI Message responses from the agent.

Nodes and edges are built in ``graph_builder.py``.

To run the repo locally:

```
streamlit run streamlit_app.py
```

For DockerBuild:
```
docker build -t test_unagent .
docker run test_unagent:latest
```
