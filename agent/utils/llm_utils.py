"""LLM utility functions for consistent invocation and JSON extraction."""

import json
import re
import time
from functools import wraps
from typing import Dict, Any, Optional, List, Callable, TypeVar
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

T = TypeVar('T')


def extract_json_from_response(response_text: str) -> str:
    """
    Extract JSON from LLM response text.
    Handles multiple formats: ```json blocks, ``` blocks, or inline JSON.
    
    Args:
        response_text: Raw response text from LLM
        
    Returns:
        Extracted JSON string
    """
    text = response_text.strip()
    
    # Try ```json block first
    if "```json" in text:
        return text.split("```json")[1].split("```")[0].strip()
    
    # Try ``` block
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    
    # Try to find JSON object with regex (handles multiline)
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group()
    
    # Return as-is if no JSON found
    return text.strip()


def parse_json_response(response_text: str, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Parse JSON from LLM response with fallback handling.
    
    Args:
        response_text: Raw response text from LLM
        fallback: Optional fallback dict if parsing fails
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        ValueError: If parsing fails and no fallback provided
    """
    try:
        json_text = extract_json_from_response(response_text)
        return json.loads(json_text)
    except (json.JSONDecodeError, ValueError) as e:
        if fallback is not None:
            return fallback
        raise ValueError(f"Failed to parse JSON from response: {e}")


def retry_llm_call(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (Exception,)
):
    """
    Decorator to retry LLM calls with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 1.0)
        backoff_factor: Multiplier for delay between retries (default: 2.0)
        retryable_exceptions: Tuple of exceptions that should trigger retry
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            delay = initial_delay
            verbose = kwargs.get('verbose', False)
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        if verbose:
                            print(f"[RETRY] Attempt {attempt}/{max_attempts} failed: {e}")
                            print(f"[RETRY] Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        if verbose:
                            print(f"[RETRY] All {max_attempts} attempts failed")
                        raise
                except Exception as e:
                    # Non-retryable exception, raise immediately
                    raise
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Retry logic failed unexpectedly")
        
        return wrapper
    return decorator


def invoke_llm_with_json(
    llm,
    system_prompt: str,
    user_message: str,
    fallback: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Invoke LLM and parse JSON response in one call.
    Includes automatic retry logic for transient failures.
    
    Args:
        llm: LLM instance (ChatOpenAI, ChatAnthropic, etc.)
        system_prompt: System prompt content
        user_message: User message content
        fallback: Optional fallback dict if parsing fails
        verbose: Whether to print debug info
        
    Returns:
        Parsed JSON dictionary
    """
    # CRITICAL FIX: Retry logic with exponential backoff
    max_attempts = 3
    initial_delay = 1.0
    backoff_factor = 2.0
    delay = initial_delay
    last_exception = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            response = llm.invoke(messages)
            response_text = response.content.strip()
            
            if verbose:
                print(f"[LLM RESPONSE] {response_text[:300]}...")
            
            return parse_json_response(response_text, fallback=fallback)
        except Exception as e:
            last_exception = e
            if attempt < max_attempts:
                if verbose:
                    print(f"[RETRY] Attempt {attempt}/{max_attempts} failed: {e}")
                    print(f"[RETRY] Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= backoff_factor
            else:
                if verbose:
                    print(f"[RETRY] All {max_attempts} attempts failed")
                # Re-raise the last exception
                raise
    
    # Should never reach here, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic failed unexpectedly")


def create_llm(model_name: str = "gpt-4o-mini"):
    """
    Create LLM instance with fallback handling.
    Tries OpenAI first, falls back to Anthropic if needed.
    
    Args:
        model_name: Model name for OpenAI (ignored for Anthropic fallback)
        
    Returns:
        LLM instance
        
    Raises:
        ValueError: If neither OpenAI nor Anthropic is available
    """
    # Try OpenAI first
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=0)
    except (ImportError, Exception):
        pass
    
    # Fallback to Anthropic
    try:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
    except ImportError:
        raise ValueError("Need either langchain-openai or langchain-anthropic installed")
    except Exception as e:
        raise ValueError(f"Failed to create LLM: {e}")

