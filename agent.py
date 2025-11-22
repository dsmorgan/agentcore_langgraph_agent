import os
import logging
from typing import Annotated
import uuid

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph_checkpoint_aws import AgentCoreMemorySaver
from bedrock_agentcore.memory import MemorySessionManager
from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole

from tools import load_search_tools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
langgraph_logger = logging.getLogger("langgraph")
langgraph_logger.setLevel(logging.DEBUG)

# Log critical library versions on initialization
from importlib.metadata import version, PackageNotFoundError

packages_to_log = [
    "langgraph",
    "langchain",
    "bedrock-agentcore",
    "langgraph-checkpoint-aws"
]

for pkg in packages_to_log:
    try:
        pkg_version = version(pkg)
        logger.info(f"{pkg} version: {pkg_version}")
    except PackageNotFoundError:
        logger.warning(f"Could not determine {pkg} version: package not found")

logger.info("Starting up...")

# Initialize LLM
llm = init_chat_model(
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    model_provider="bedrock_converse",
)

# Load search tools
tools = load_search_tools()

# Initialize AgentCore Memory and Checkpointer
MEMORY_ID = os.environ.get("AGENTCORE_MEMORY_ID", "langgraph_agent_web_search_mem-U27RdV377G")
REGION = os.environ.get("AWS_REGION", "us-east-1")

try:
    checkpointer = AgentCoreMemorySaver(
        memory_id=MEMORY_ID,
        region_name=REGION
    )
    memory_manager = MemorySessionManager(
        memory_id=MEMORY_ID,
        region_name=REGION
    )
    logger.info(f"Initialized AgentCoreMemorySaver with memory_id={MEMORY_ID}")
except Exception as e:
    logger.warning(f"Failed to initialize AgentCore Memory: {e}")
    checkpointer = None
    memory_manager = None

llm_with_tools = llm.bind_tools(tools)

logger.info("Defining state...")

# Define state
class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


logger.info("Configuring graph...")
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile with checkpointer if available
if checkpointer:
    graph = graph_builder.compile(checkpointer=checkpointer)
    logger.info("Graph compiled with AgentCoreMemorySaver checkpointer")
else:
    graph = graph_builder.compile()
    logger.warning("Graph compiled without checkpointer - state will not persist")

graph_configured = True

from bedrock_agentcore.runtime import BedrockAgentCoreApp
app = BedrockAgentCoreApp()

@app.entrypoint
def agent_invocation(payload, context):
    """
    Agent invocation handler with session and memory persistence.

    Expected payload:
    {
        "prompt": "User question",
        "session_id": "optional-session-id",
        "actor_id": "optional-user-id"
    }
    """
    logger.info("Received payload")
    logger.debug(f"Payload: {payload}")

    # Extract session and user identifiers
    session_id = payload.get("session_id") or str(uuid.uuid4())
    actor_id = payload.get("actor_id", "default-user")
    prompt = payload.get("prompt", "No prompt found in input, please guide customer as to what tools can be used")

    logger.info(f"Session: {session_id}, Actor: {actor_id}")

    # Prepare messages for graph invocation
    messages = [{"role": "user", "content": prompt}]

    # Build config for checkpointer (enables state persistence)
    config = {
        "configurable": {
            "thread_id": session_id,
            "actor_id": actor_id
        }
    } if checkpointer else None

    # Retrieve long-term memories if memory manager is available
    if memory_manager:
        try:
            session = memory_manager.create_memory_session(
                actor_id=actor_id,
                session_id=session_id
            )

            # Retrieve relevant long-term memories
            memories = session.search_long_term_memories(
                query=prompt,
                namespace_prefix=f"/users/{actor_id}",
                top_k=5
            )

            if memories:
                logger.info(f"Retrieved {len(memories)} long-term memories")
                memory_context = "\n".join([
                    m.get("content", {}).get("text", "")
                    for m in memories if m.get("content", {}).get("text")
                ])
                if memory_context:
                    # Prepend memory context as system message
                    messages.insert(0, {
                        "role": "system",
                        "content": f"Relevant context from memory:\n{memory_context}"
                    })

        except Exception as e:
            logger.warning(f"Failed to retrieve long-term memories: {e}")

    # Invoke graph with session config and messages
    try:
        if config:
            tmp_output = graph.invoke({"messages": messages}, config=config)
        else:
            tmp_output = graph.invoke({"messages": messages})

        logger.info("Graph invocation completed")
        logger.debug(f"Output: {tmp_output}")

        # Extract the assistant response
        assistant_response = tmp_output['messages'][-1].content

        # Save conversation turn to memory if memory manager is available
        if memory_manager:
            try:
                session = memory_manager.create_memory_session(
                    actor_id=actor_id,
                    session_id=session_id
                )
                session.add_turns([
                    ConversationalMessage(prompt, MessageRole.USER),
                    ConversationalMessage(assistant_response, MessageRole.ASSISTANT)
                ])
                logger.info("Saved conversation turn to AgentCore Memory")
            except Exception as e:
                logger.warning(f"Failed to save conversation turn to memory: {e}")

        return {
            "result": assistant_response,
            "session_id": session_id,
            "actor_id": actor_id
        }

    except Exception as e:
        logger.error(f"Graph invocation failed: {e}")
        raise


if __name__ == "__main__":
    app.run()
