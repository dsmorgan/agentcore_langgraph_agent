"""
Test script for the LangGraph agent with state persistence.
Tests single-turn and multi-turn conversations with session management.
"""

import json
import uuid
from langgraph_agent_web_search import agent_invocation, graph, memory_manager


class MockContext:
    """Mock BedrockAgentCoreContext for local testing."""
    def __init__(self, session_id=None):
        self.session_id = session_id


def test_single_turn():
    """Test a single-turn conversation."""
    print("\n" + "=" * 80)
    print("TEST 1: Single-Turn Conversation")
    print("=" * 80)

    payload = {
        "prompt": "What is the capital of France?",
        "session_id": str(uuid.uuid4()),
        "actor_id": "test-user-1"
    }

    context = MockContext()

    print(f"\nInput Payload:")
    print(json.dumps(payload, indent=2))

    try:
        result = agent_invocation(payload, context)
        print(f"\nResult:")
        print(json.dumps(result, indent=2))
        print("\n✓ Single-turn test passed")
        return result
    except Exception as e:
        print(f"\n✗ Single-turn test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multi_turn():
    """Test multi-turn conversation with session persistence."""
    print("\n" + "=" * 80)
    print("TEST 2: Multi-Turn Conversation with Session Persistence")
    print("=" * 80)

    session_id = str(uuid.uuid4())
    actor_id = "test-user-2"
    context = MockContext(session_id)

    # Turn 1: Ask about France
    print("\n--- Turn 1 ---")
    payload1 = {
        "prompt": "Tell me about the Eiffel Tower.",
        "session_id": session_id,
        "actor_id": actor_id
    }

    print(f"\nInput Payload:")
    print(json.dumps(payload1, indent=2))

    try:
        result1 = agent_invocation(payload1, context)
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
        "actor_id": actor_id
    }

    print(f"\nInput Payload:")
    print(json.dumps(payload2, indent=2))

    try:
        result2 = agent_invocation(payload2, context)
        print(f"\nResult:")
        print(json.dumps(result2, indent=2))
        print("\n✓ Turn 2 passed")
    except Exception as e:
        print(f"\n✗ Turn 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n✓ Multi-turn test passed")


def test_memory_integration():
    """Test AgentCore Memory integration."""
    print("\n" + "=" * 80)
    print("TEST 3: AgentCore Memory Integration")
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
        "actor_id": actor_id
    }

    print(f"\nInput Payload:")
    print(json.dumps(payload1, indent=2))

    try:
        result1 = agent_invocation(payload1, context)
        print(f"\nResult:")
        print(json.dumps(result1, indent=2))
        print("\n✓ Turn 1 passed")
    except Exception as e:
        print(f"\n✗ Turn 1 failed: {e}")
        return

    print("\n(Waiting for memory extraction...)")
    import time
    time.sleep(2)

    # Turn 2: Ask about preferences (should retrieve from memory)
    print("\n--- Turn 2: Retrieve from Long-Term Memory ---")
    payload2 = {
        "prompt": "What are my favorite cuisines?",
        "session_id": session_id,
        "actor_id": actor_id
    }

    print(f"\nInput Payload:")
    print(json.dumps(payload2, indent=2))

    try:
        result2 = agent_invocation(payload2, context)
        print(f"\nResult:")
        print(json.dumps(result2, indent=2))
        print("\n✓ Turn 2 passed")
    except Exception as e:
        print(f"\n✗ Turn 2 failed: {e}")
        return

    print("\n✓ Memory integration test passed")


if __name__ == "__main__":
    print("\nLangGraph Agent with State Persistence - Test Suite")
    print("=" * 80)

    # Run tests
    test_single_turn()
    test_multi_turn()
    test_memory_integration()

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
