"""
Test script for the LangGraph agent with state persistence.
Tests single-turn and multi-turn conversations with session management.
Supports both streaming and non-streaming modes.
"""

import json
import uuid
import asyncio
from agent import agent_invocation, graph, memory_manager


class MockContext:
    """Mock BedrockAgentCoreContext for local testing."""
    def __init__(self, session_id=None):
        self.session_id = session_id


async def test_single_turn():
    """Test a single-turn conversation (non-streaming)."""
    print("\n" + "=" * 80)
    print("TEST 1: Single-Turn Conversation (Non-Streaming)")
    print("=" * 80)

    payload = {
        "prompt": "What is the capital of France?",
        "session_id": str(uuid.uuid4()),
        "actor_id": "test-user-1",
        "stream": False
    }

    context = MockContext()

    print(f"\nInput Payload:")
    print(json.dumps(payload, indent=2))

    try:
        # Non-streaming mode now yields a single result
        result = None
        async for item in agent_invocation(payload, context):
            result = item
        print(f"\nResult:")
        print(json.dumps(result, indent=2))
        print("\n✓ Single-turn test passed")
        return result
    except Exception as e:
        print(f"\n✗ Single-turn test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_multi_turn():
    """Test multi-turn conversation with session persistence (non-streaming)."""
    print("\n" + "=" * 80)
    print("TEST 2: Multi-Turn Conversation with Session Persistence (Non-Streaming)")
    print("=" * 80)

    session_id = str(uuid.uuid4())
    actor_id = "test-user-2"
    context = MockContext(session_id)

    # Turn 1: Ask about France
    print("\n--- Turn 1 ---")
    payload1 = {
        "prompt": "Tell me about the Eiffel Tower.",
        "session_id": session_id,
        "actor_id": actor_id,
        "stream": False
    }

    print(f"\nInput Payload:")
    print(json.dumps(payload1, indent=2))

    try:
        # Non-streaming mode now yields a single result
        result1 = None
        async for item in agent_invocation(payload1, context):
            result1 = item
        print(f"\nResult:")
        print(json.dumps(result1, indent=2))
        print("\n✓ Turn 1 passed")
    except Exception as e:
        print(f"\n✗ Turn 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Turn 2: Follow-up question (should have access to previous context)
    print("\n--- Turn 2 ---")
    payload2 = {
        "prompt": "How tall is it?",
        "session_id": session_id,
        "actor_id": actor_id,
        "stream": False
    }

    print(f"\nInput Payload:")
    print(json.dumps(payload2, indent=2))

    try:
        # Non-streaming mode now yields a single result
        result2 = None
        async for item in agent_invocation(payload2, context):
            result2 = item
        print(f"\nResult:")
        print(json.dumps(result2, indent=2))
        print("\n✓ Turn 2 passed")
    except Exception as e:
        print(f"\n✗ Turn 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n✓ Multi-turn test passed")


async def test_memory_integration():
    """Test AgentCore Memory integration (non-streaming)."""
    print("\n" + "=" * 80)
    print("TEST 3: AgentCore Memory Integration (Non-Streaming)")
    print("=" * 80)

    if not memory_manager:
        print("✗ Memory manager not initialized - skipping test")
        return

    session_id = str(uuid.uuid4())
    actor_id = "test-user-3"
    context = MockContext(session_id)

    # Turn 1: Establish a preference
    print("\n--- Turn 1: Establish Preference ---")
    payload1 = {
        "prompt": "I really like Italian food and coffee.",
        "session_id": session_id,
        "actor_id": actor_id,
        "stream": False
    }

    print(f"\nInput Payload:")
    print(json.dumps(payload1, indent=2))

    try:
        # Non-streaming mode now yields a single result
        result1 = None
        async for item in agent_invocation(payload1, context):
            result1 = item
        print(f"\nResult:")
        print(json.dumps(result1, indent=2))
        print("\n✓ Turn 1 passed")
    except Exception as e:
        print(f"\n✗ Turn 1 failed: {e}")
        return

    print("\n(Waiting for memory extraction...)")
    await asyncio.sleep(2)

    # Turn 2: Ask about preferences (should retrieve from memory)
    print("\n--- Turn 2: Retrieve from Long-Term Memory ---")
    payload2 = {
        "prompt": "What are my favorite cuisines?",
        "session_id": session_id,
        "actor_id": actor_id,
        "stream": False
    }

    print(f"\nInput Payload:")
    print(json.dumps(payload2, indent=2))

    try:
        # Non-streaming mode now yields a single result
        result2 = None
        async for item in agent_invocation(payload2, context):
            result2 = item
        print(f"\nResult:")
        print(json.dumps(result2, indent=2))
        print("\n✓ Turn 2 passed")
    except Exception as e:
        print(f"\n✗ Turn 2 failed: {e}")
        return

    print("\n✓ Memory integration test passed")


async def test_streaming():
    """Test streaming mode."""
    print("\n" + "=" * 80)
    print("TEST 4: Streaming Response")
    print("=" * 80)

    payload = {
        "prompt": "Tell me a short story about a robot.",
        "session_id": str(uuid.uuid4()),
        "actor_id": "test-user-4",
        "stream": True
    }

    context = MockContext()

    print(f"\nInput Payload:")
    print(json.dumps(payload, indent=2))
    print("\nStreaming response:")
    print("-" * 80)

    try:
        full_response = ""
        async for event in agent_invocation(payload, context):
            event_type = event.get("type")

            if event_type == "content_chunk":
                content = event.get("content", "")
                print(content, end="", flush=True)
                full_response += content
            elif event_type == "tool_start":
                tool = event.get("tool", "unknown")
                print(f"\n[Tool started: {tool}]", flush=True)
            elif event_type == "tool_end":
                tool = event.get("tool", "unknown")
                print(f"\n[Tool completed: {tool}]", flush=True)
            elif event_type == "done":
                print("\n" + "-" * 80)
                print(f"\n✓ Streaming test passed")
                print(f"Total characters streamed: {len(full_response)}")
            elif event_type == "error":
                print(f"\n✗ Streaming error: {event.get('error')}")
                return

    except Exception as e:
        print(f"\n✗ Streaming test failed: {e}")
        import traceback
        traceback.print_exc()
        return


async def main():
    """Run all tests."""
    print("\nLangGraph Agent with State Persistence - Test Suite")
    print("=" * 80)

    # Run tests
    await test_single_turn()
    await test_multi_turn()
    await test_memory_integration()
    await test_streaming()

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
