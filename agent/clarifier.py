"""Clarification node for asking user for more details."""

from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage
from agent.state import AgentState


def create_clarification_node():
    """Node that asks user for clarification."""
    
    def clarification_node(state: AgentState) -> AgentState:
        """Ask user for clarification and get their response."""
        verbose = state.get("verbose", False)
        question = state.get("clarification_question", "Could you provide more details?")
        
        if verbose:
            print("\n" + "=" * 60)
            print("CLARIFIER: Asking for Clarification")
            print("=" * 60)
            print(f"Question: {question}")
            print("\n[CLARIFIER] Waiting for user response...")
        
        # Prompt user for input
        print(f"\n{question}")
        user_response = input("Your response: ").strip()
        
        if not user_response:
            # If user just presses enter, use original query
            user_response = state["user_query"]
            if verbose:
                print("[CLARIFIER] No response provided, using original query")
        else:
            if verbose:
                print(f"[CLARIFIER] User provided: {user_response}")
        
        # Update state with new query and continue to planner
        return {
            **state,
            "user_query": user_response,
            "messages": state["messages"] + [
                AIMessage(content=question),
                HumanMessage(content=user_response)
            ],
            "needs_clarification": False,  # Clear the flag so planner processes the new query
            "clarification_question": None
        }
    
    return clarification_node


def should_ask_clarification(state: AgentState) -> Literal["clarify", "execute"]:
    """Router: decide whether to ask for clarification or execute."""
    verbose = state.get("verbose", False)
    needs_clarification = state.get("needs_clarification", False)
    
    if verbose:
        print("\n" + "=" * 60)
        print("ROUTING DECISION")
        print("=" * 60)
        print(f"Needs clarification: {needs_clarification}")
        if needs_clarification:
            print(f"Clarification question: {state.get('clarification_question', 'N/A')}")
            print("[ROUTING] → Going to CLARIFIER")
        else:
            print(f"Confidence: {state.get('confidence', 0):.2f}")
            print(f"Time ranges found: {len(state.get('time_ranges', []))}")
            print("[ROUTING] → Going to EXECUTOR")
    
    if needs_clarification:
        return "clarify"
    return "execute"

