"""Planner agent for analyzing queries and retrieving relevant moments."""

import json
import re
import time
from typing import Optional, Set
from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import AgentState
from agent.feature_extractor import PerSecondFeatureExtractor
from agent.query_analysis import (
    analyze_query_semantics,
    plan_search_strategy,
    validate_and_adjust_intent,
    configure_weights
)
from agent.scoring import score_seconds, group_contiguous_seconds
from agent.refinement import decide_refine_or_research, refine_existing_results, validate_search_results, validate_activity_evidence
from agent.utils import extract_json, merge_time_ranges
from agent.prompts import (
    PLANNER_SYSTEM_PROMPT,
    SEGMENT_TREE_INSPECTION_PROMPT,
    QUERY_REASONING_PROMPT,
    SEARCH_QUERY_GENERATION_PROMPT,
    VIDEO_NARRATION_PROMPT
)

# Import LLM classes
try:
    from langchain_openai import ChatOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from langchain_anthropic import ChatAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


def create_video_narrative(content_inspection: dict, query: str, llm, verbose: bool = False, logger=None) -> Optional[dict]:
    """
    Create a coherent narrative understanding of the video content.
    This helps generate better search queries by understanding context, not just keywords.
    
    Args:
        content_inspection: Dictionary from segment_tree.inspect_content()
        query: User query string
        llm: Language model instance
        verbose: Whether to print verbose output
        logger: Optional logger instance
        
    Returns:
        Dictionary with video narrative understanding or None if failed
    """
    if not content_inspection:
        return None
    
    # Helper function to log or print
    def log_info(msg):
        if logger:
            logger.info(msg)
        elif verbose:
            print(msg)
    
    # Build context for narration - emphasize actual keywords
    all_keywords = content_inspection['all_keywords'][:100]  # Get more keywords
    context = f"""Video Content Inspection:

AVAILABLE KEYWORDS (use ONLY these keywords - do not invent new ones):
{', '.join(all_keywords)}

Object classes detected: {', '.join(sorted(content_inspection['object_classes'].keys())[:20])}

Sample descriptions from video:
"""
    for i, desc in enumerate(content_inspection['sample_descriptions'][:10], 1):
        context += f"{i}. [{desc['second']}s] {desc['description'][:150]}\n"
    
    context += f"\nUser Query: {query}\n"
    context += "\nCRITICAL: Create a narrative structure (intro, body, ending) using ONLY keywords from the list above. "
    context += "Each narrative part should have keywords that actually exist in the hierarchical tree. "
    context += "What would be considered highlights using these actual keywords?"
    
    try:
        start_time = time.time()
        response = llm.invoke([
            SystemMessage(content=VIDEO_NARRATION_PROMPT),
            HumanMessage(content=context)
        ])
        elapsed = time.time() - start_time
        
        response_text = response.content.strip()
        json_text = extract_json(response_text)
        narrative = json.loads(json_text)
        
        log_info(f"  Video theme: {narrative.get('video_theme', 'N/A')}")
        log_info(f"  Narrative summary: {narrative.get('narrative_summary', 'N/A')[:100]}...")
        
        # Log narrative structure if available
        narrative_structure = narrative.get('narrative_structure', {})
        if narrative_structure:
            log_info(f"  Narrative structure:")
            for part_name in ['intro', 'body', 'ending']:
                part = narrative_structure.get(part_name, {})
                if part:
                    keywords = part.get('keywords', [])
                    log_info(f"    {part_name.upper()}: {', '.join(keywords[:5]) if keywords else 'N/A'}")
        
        highlight_criteria = narrative.get('highlight_criteria', {})
        if isinstance(highlight_criteria, dict):
            highlight_keywords = highlight_criteria.get('keywords', [])
            log_info(f"  Highlight keywords: {', '.join(highlight_keywords[:5]) if highlight_keywords else 'N/A'}")
        else:
            log_info(f"  Highlight criteria: {highlight_criteria}")
        
        log_info(f"  Narrative creation completed in {elapsed:.2f}s")
        
        return narrative
    except Exception as e:
        log_info(f"  [WARNING] Narrative creation failed: {e}")
        return None


def create_planner_agent(model_name: str = "gpt-4o-mini"):
    """Create the planner agent that analyzes queries and retrieves relevant moments."""
    
    # Try OpenAI first, fallback to Anthropic if needed
    if HAS_OPENAI:
        try:
            llm = ChatOpenAI(model=model_name, temperature=0)
        except:
            if HAS_ANTHROPIC:
                llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
            else:
                raise ValueError("Need either OpenAI or Anthropic API key configured")
    elif HAS_ANTHROPIC:
        llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
    else:
        raise ValueError("Need either langchain-openai or langchain-anthropic installed")
    
    def planner_node(state: AgentState) -> AgentState:
        """Planner agent: Analyzes user query and retrieves relevant moments using ALL search types."""
        query = state["user_query"]
        segment_tree = state["segment_tree"]
        verbose = state.get("verbose", False)
        logger = state.get("logger")
        
        # Helper function to log or print
        def log_info(msg):
            if logger:
                logger.info(msg)
            elif verbose:
                print(msg)
        
        # AGENTIC DECISION: Check if we should refine existing results or do new search
        if state.get("previous_time_ranges") and state.get("previous_query"):
            log_info("\n" + "=" * 60)
            log_info("PLANNER AGENT: Context-Aware Decision")
            log_info("=" * 60)
            log_info(f"Previous query: {state.get('previous_query')}")
            log_info(f"Previous results: {len(state.get('previous_time_ranges', []))} time ranges")
            log_info(f"Current query: {query}")
            log_info("\n[DECISION] Analyzing user intent...")
            
            decision = decide_refine_or_research(state, llm)
            
            log_info(f"  Decision: {decision.get('action')}")
            log_info(f"  Reason: {decision.get('reason')}")
            if decision.get('extract_number'):
                log_info(f"  Extract number: {decision.get('extract_number')}")
            
            if decision.get("action") == "REFINE":
                # Refine existing results
                return refine_existing_results(state, decision, verbose=verbose)
            # Otherwise, continue with new search (fall through)
        
        # Initialize variables
        search_results = []
        time_ranges = []
        confidence = 0.5
        needs_clarification = False
        clarification_question = None
        filtered_seconds = []  # For preserving scored seconds
        
        log_info("\n" + "=" * 60)
        log_info("PLANNER AGENT: Multi-Modal Search Strategy")
        log_info("=" * 60)
        log_info(f"Query: {query}")
        
        # Check for temporal constraints
        temporal_constraint = state.get("temporal_constraint")
        temporal_type = state.get("temporal_type")
        if temporal_constraint:
            log_info(f"Temporal constraint: '{temporal_constraint}' (type: {temporal_type})")
            log_info("  → Query includes temporal/conditional constraint - will search with full context")
        
        # STEP 0: Decide if we need to inspect segment tree content
        # Inspect for abstract queries, highlights, vague queries, etc.
        query_lower = query.lower()
        abstract_indicators = [
            "highlight", "highlights", "best", "important", "key", "significant",
            "interesting", "exciting", "memorable", "notable", "noteworthy",
            "moments", "scenes", "parts", "events", "action"
        ]
        needs_inspection = any(indicator in query_lower for indicator in abstract_indicators)
        
        # Also inspect if query is very vague or general
        vague_queries = ["what", "show me", "find", "get", "give me"]
        if any(vq in query_lower for vq in vague_queries) and len(query.split()) <= 5:
            needs_inspection = True
        
        content_inspection = None
        if needs_inspection and segment_tree:
            log_info("\n[INSPECTION] Inspecting segment tree to understand video content...")
            try:
                start_time = time.time()
                content_inspection = segment_tree.inspect_content(max_keywords=100, max_sample_descriptions=20)
                elapsed = time.time() - start_time
                log_info(f"  Found {content_inspection['keyword_count']} unique keywords")
                log_info(f"  Found {content_inspection['object_class_count']} object classes")
                log_info(f"  Sample keywords: {', '.join(content_inspection['all_keywords'][:15])}...")
                log_info(f"  Inspection completed in {elapsed:.2f}s")
            except Exception as e:
                log_info(f"  [WARNING] Inspection failed: {e}")
                content_inspection = None
        
        # NEW STEP: Create video narrative understanding
        video_narrative = None
        if content_inspection and needs_inspection:
            log_info("\n[NARRATION] Creating video narrative understanding...")
            video_narrative = create_video_narrative(content_inspection, query, llm, verbose=verbose, logger=logger)
            if video_narrative:
                # Logging is handled inside create_video_narrative
                pass
            else:
                log_info("  [WARNING] Narrative creation failed, proceeding with raw inspection data")
        
        log_info("\n[STEP 1] Generating search queries for all modalities...")
        
        # STEP 1: Generate search queries/keywords for ALL search types
        # Use prompts from prompts.py
        system_prompt_step1 = PLANNER_SYSTEM_PROMPT + "\n\n" + SEARCH_QUERY_GENERATION_PROMPT
        
        # Build user message with inspection data if available
        user_message_content = f"User query: {query}\n\n"
        if content_inspection:
            user_message_content += f"""I've inspected the video content. Here's what's available:

{content_inspection['summary']}

Sample keywords from video: {', '.join(content_inspection['all_keywords'][:30])}
Object classes: {', '.join(sorted(content_inspection['object_classes'].keys())[:20])}

Sample descriptions:
"""
            for i, desc in enumerate(content_inspection['sample_descriptions'][:5], 1):
                user_message_content += f"{i}. [{desc['second']}s] {desc['description'][:100]}\n"
            
            # Add video narrative understanding if available
            if video_narrative:
                narrative_structure = video_narrative.get('narrative_structure', {})
                highlight_criteria = video_narrative.get('highlight_criteria', {})
                
                user_message_content += f"""

Video Narrative Understanding:
- Theme: {video_narrative.get('video_theme', 'N/A')}
- Summary: {video_narrative.get('narrative_summary', 'N/A')}

Narrative Structure (use keywords from each part):
"""
                if narrative_structure:
                    for part_name in ['intro', 'body', 'ending']:
                        part = narrative_structure.get(part_name, {})
                        if part:
                            keywords = part.get('keywords', [])
                            description = part.get('description', 'N/A')
                            user_message_content += f"- {part_name.upper()}: {description}\n"
                            user_message_content += f"  Keywords: {', '.join(keywords) if keywords else 'N/A'}\n"
                
                if highlight_criteria:
                    highlight_keywords = highlight_criteria.get('keywords', [])
                    highlight_desc = highlight_criteria.get('description', 'N/A')
                    user_message_content += f"\nHighlight Criteria: {highlight_desc}\n"
                    user_message_content += f"Highlight Keywords: {', '.join(highlight_keywords) if highlight_keywords else 'N/A'}\n"
                
                user_message_content += f"\nKey Objects: {', '.join(video_narrative.get('key_objects', []))}\n"
                
                user_message_content += """
CRITICAL INSTRUCTIONS:
1. Use ONLY the keywords provided in the narrative structure above - these are actual keywords from the hierarchical tree
2. Generate search queries that use these actual keywords, not abstract concepts
3. Create queries for each narrative part (intro, body, ending) using their specific keywords
4. For highlights, use the highlight_criteria keywords which are searchable in the tree
5. Do NOT invent keywords that aren't in the narrative structure - use only what's provided
"""
            else:
                user_message_content += "\nBased on this actual video content, generate search queries that target concrete elements, not abstract concepts.\n"
                user_message_content += "For example, if the query is 'highlights' and the video contains 'camping', 'fishing', 'cooking', generate queries like 'people cooking together', 'fishing success', 'group activities'.\n\n"
        
        user_message_content += "Generate search queries and keywords for ALL search types. Return JSON only."
        
        messages_step1 = [
            SystemMessage(content=system_prompt_step1),
            HumanMessage(content=user_message_content)
        ]
        
        log_info("[LLM] Calling LLM to generate search queries...")
        start_time = time.time()
        
        response_step1 = llm.invoke(messages_step1)
        elapsed = time.time() - start_time
        
        log_info(f"[LLM] Query generation completed in {elapsed:.2f}s")
        response_text_step1 = response_step1.content.strip()
        
        if logger:
            logger.debug(f"[LLM RESPONSE] {response_text_step1[:300]}...")
        elif verbose:
            print(f"[LLM RESPONSE] {response_text_step1[:300]}...")
        
        # Extract JSON from response
        if "```json" in response_text_step1:
            response_text_step1 = response_text_step1.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text_step1:
            response_text_step1 = response_text_step1.split("```")[1].split("```")[0].strip()
        
        try:
            search_plan = json.loads(response_text_step1)
        except:
            json_match = re.search(r'\{[^}]+\}', response_text_step1)
            if json_match:
                search_plan = json.loads(json_match.group())
            else:
                # Fallback: use original query for semantic, extract keywords for hierarchical tree
                # Extract meaningful keywords (nouns/verbs, at least 3 chars)
                words = re.findall(r'\b\w+\b', query.lower())
                meaningful_keywords = [w for w in words if len(w) >= 3]
                search_plan = {
                    "semantic_queries": [query],
                    "hierarchical_keywords": meaningful_keywords[:5],  # Top 5 keywords
                    "object_classes": [],
                    "activity_name": "",
                    "activity_keywords": [],
                    "evidence_keywords": [],
                    "is_general_highlight_query": "highlight" in query.lower(),
                    "needs_clarification": False
                }
        
        log_info("\n[SEARCH PLAN] Generated search queries:")
        if content_inspection:
            log_info(f"  [INSPECTION] Used segment tree inspection: Yes")
        log_info(f"  Semantic queries: {search_plan.get('semantic_queries', [])}")
        log_info(f"  Hierarchical keywords: {search_plan.get('hierarchical_keywords', [])}")
        log_info(f"  Object classes: {search_plan.get('object_classes', [])}")
        log_info(f"  Activity: {search_plan.get('activity_name', 'N/A')}")
        log_info(f"  Is general highlight query: {search_plan.get('is_general_highlight_query', False)}")
        
        # AGENTIC PHASE 1: Semantic Query Analysis
        log_info("\n[AGENTIC] Phase 1: Semantic Query Analysis...")
        start_time = time.time()
        semantic_analysis = analyze_query_semantics(query, llm)
        elapsed = time.time() - start_time
        log_info(f"  Query type: {semantic_analysis.get('query_type', 'POSITIVE')}")
        log_info(f"  Search intent: {semantic_analysis.get('search_intent', 'hybrid')}")
        log_info(f"  Target entities: {semantic_analysis.get('target_entities', {})}")
        log_info(f"  Special handling: {semantic_analysis.get('special_handling', {})}")
        log_info(f"  Reasoning: {semantic_analysis.get('reasoning', 'N/A')}")
        log_info(f"  Analysis completed in {elapsed:.2f}s")
        
        # AGENTIC PHASE 2: Dynamic Strategy Planning
        log_info("\n[AGENTIC] Phase 2: Dynamic Strategy Planning...")
        start_time = time.time()
        strategy = plan_search_strategy(semantic_analysis, llm, verbose=verbose)
        elapsed = time.time() - start_time
        log_info(f"  Strategy operations: {len(strategy.get('search_operations', []))}")
        log_info(f"  Post-processing: {strategy.get('post_processing', [])}")
        log_info(f"  Reasoning: {strategy.get('reasoning', 'N/A')}")
        log_info(f"  Strategy planning completed in {elapsed:.2f}s")
        
        # Convert strategy to query_intent format for backward compatibility
        query_intent = {
            "is_object_centric": semantic_analysis.get("search_intent") == "object",
            "mentioned_classes": semantic_analysis.get("target_entities", {}).get("objects", []),
            "primary_intent": semantic_analysis.get("search_intent", "hybrid"),
            "object_priority": semantic_analysis.get("object_priority", {}),
            "needs_semantic": semantic_analysis.get("modalities", {}).get("semantic_search", True),
            "needs_activity": semantic_analysis.get("modalities", {}).get("activity_detection", False),
            "needs_hierarchical": semantic_analysis.get("modalities", {}).get("hierarchical_search", True),
            "confidence": semantic_analysis.get("confidence", 0.7),
            "_semantic_analysis": semantic_analysis,
            "_strategy": strategy
        }
        
        # Cross-validate with search plan (keep for compatibility)
        log_info("\n[VALIDATION] Cross-validating with search plan...")
        query_intent = validate_and_adjust_intent(query_intent, search_plan, verbose=verbose)
        
        # NEW: Initialize feature extractor
        feature_extractor = PerSecondFeatureExtractor(segment_tree)
        
        # NEW: Get all object classes and configure weights (use strategy scoring if available)
        all_object_classes = set(segment_tree.get_all_classes().keys())
        if strategy.get("scoring"):
            # Use strategy-based weights
            strategy_scoring = strategy["scoring"]
            weights = {
                "semantic_weight": strategy_scoring.get("weights", {}).get("semantic", 0.4),
                "activity_weight": strategy_scoring.get("weights", {}).get("activity", 0.3),
                "hierarchical_weight": strategy_scoring.get("weights", {}).get("hierarchical", 0.1),
                "object_weights": strategy_scoring.get("object_weights", {}),
                "threshold": strategy_scoring.get("threshold", 0.3)
            }
            # Initialize all object classes with default low weight
            for class_name in all_object_classes:
                if class_name not in weights["object_weights"]:
                    weights["object_weights"][class_name] = 0.1
            
            # FIX: Ensure semantic weight is set if semantic queries were generated
            has_semantic_queries = bool(search_plan.get("semantic_queries"))
            if has_semantic_queries and weights["semantic_weight"] == 0.0:
                # Override: if semantic queries exist, semantic weight should be > 0
                if weights["hierarchical_weight"] > 0:
                    # Reduce hierarchical to make room for semantic
                    weights["hierarchical_weight"] = max(0.05, weights["hierarchical_weight"] * 0.5)
                weights["semantic_weight"] = 0.4  # Set reasonable default
        else:
            # Fallback to old weight configuration
            weights = configure_weights(query_intent, all_object_classes)
            
            # FIX: Ensure semantic weight is set if semantic queries were generated
            has_semantic_queries = bool(search_plan.get("semantic_queries"))
            if has_semantic_queries and weights["semantic_weight"] == 0.0:
                # Override: if semantic queries exist, semantic weight should be > 0
                if weights["hierarchical_weight"] > 0:
                    # Reduce hierarchical to make room for semantic
                    weights["hierarchical_weight"] = max(0.05, weights["hierarchical_weight"] * 0.5)
                weights["semantic_weight"] = 0.4  # Set reasonable default
        log_info(f"\n[WEIGHT CONFIGURATION]")
        log_info(f"  Semantic weight: {weights['semantic_weight']:.2f}")
        log_info(f"  Activity weight: {weights['activity_weight']:.2f}")
        log_info(f"  Hierarchical weight: {weights['hierarchical_weight']:.2f}")
        log_info(f"  Threshold: {weights['threshold']:.2f}")
        high_priority_objects = {k: v for k, v in weights['object_weights'].items() if v > 0.3}
        if high_priority_objects:
            log_info(f"  High priority objects: {high_priority_objects}")
        
        needs_clarification = search_plan.get("needs_clarification", False)
        clarification_question = search_plan.get("clarification_question")
        
        # Check for audio query ambiguity
        query_lower = query.lower()
        audio_indicators = ["say", "said", "talk", "speak", "mention", "discuss", "conversation", "audio", "transcription"]
        is_audio_query = any(indicator in query_lower for indicator in audio_indicators)
        ambiguous_pronouns = ["they", "he", "she", "it", "that", "this", "those", "these"]
        has_ambiguous_refs = any(pronoun in query_lower.split() for pronoun in ambiguous_pronouns)
        
        if is_audio_query and has_ambiguous_refs and not needs_clarification:
            needs_clarification = True
            clarification_question = "I notice your query mentions audio/transcription but uses ambiguous references. Could you clarify who or what you're looking for?"
            log_info("[WARNING] Ambiguous audio query detected - will ask for clarification")
        
        if needs_clarification and not is_audio_query:
            log_info("[INFO] LLM suggested clarification, but query seems clear enough. Will try searching first.")
            needs_clarification = False
        
        if not needs_clarification:
            log_info("\n[STEP 2] Executing search with hierarchical tree + semantic search...")
            
            # STEP 2: Execute search types and collect results
            all_search_results = []  # Store all results with metadata about search type
            
            # Check if this is a general highlight query
            is_general_highlight = search_plan.get("is_general_highlight_query", False)
            if "highlight" in query_lower and ("all" in query_lower or "find" in query_lower):
                is_general_highlight = True
            
            # 0. General highlight detection (using hierarchical tree)
            if is_general_highlight and segment_tree.hierarchical_tree:
                log_info(f"\n  [SEARCH TYPE: General Highlights (Hierarchical Tree)]")
                # Extract action keywords from query if any
                hierarchical_keywords = search_plan.get("hierarchical_keywords", [])
                scored_leaves = segment_tree.hierarchical_score_leaves_for_highlights(
                    action_keywords=hierarchical_keywords if hierarchical_keywords else None,
                    max_results=20
                )
                for leaf in scored_leaves:
                    tr = leaf.get("time_range", [])
                    if tr and len(tr) >= 2:
                        all_search_results.append({
                            "search_type": "hierarchical_highlight",
                            "time_range": tr,
                            "score": leaf.get("score", 0),
                            "node_id": leaf.get("node_id"),
                            "keyword_count": leaf.get("keyword_count", 0)
                        })
                log_info(f"      Found {len(scored_leaves)} highlight candidates")
            
            # 1. Hierarchical tree keyword search (fast pre-filter)
            hierarchical_keywords = search_plan.get("hierarchical_keywords", [])
            if hierarchical_keywords and segment_tree.hierarchical_tree:
                log_info(f"\n  [SEARCH TYPE: Hierarchical Tree (Fast Keyword Lookup)]")
                for keyword in hierarchical_keywords:
                    log_info(f"    Keyword: '{keyword}'")
                    results = segment_tree.hierarchical_keyword_search(
                        [keyword],
                        match_mode="any",
                        max_results=20
                    )
                    for result in results:
                        # Prefer leaf nodes
                        if result.get("level", -1) == 0:
                            tr = result.get("time_range", [])
                            if tr and len(tr) >= 2:
                                all_search_results.append({
                                    "search_type": "hierarchical",
                                    "time_range": tr,
                                    "score": result.get("match_count", 1),
                                    "node_id": result.get("node_id"),
                                    "matched_keyword": keyword
                                })
                    log_info(f"      Found {len([r for r in results if r.get('level') == 0])} leaf node matches")
            
            # 2. Semantic search (replaces old visual/audio keyword search)
            semantic_queries = search_plan.get("semantic_queries", [query])
            if semantic_queries:
                log_info(f"\n  [SEARCH TYPE: Semantic Search (Visual + Audio)]")
                for sem_query in semantic_queries:
                    log_info(f"    Query: '{sem_query}'")
                    start_time = time.time()
                    threshold = 0.5
                    results = segment_tree.semantic_search(
                        sem_query,
                        top_k=15,
                        threshold=threshold,
                        search_transcriptions=True,
                        search_unified=True,
                        verbose=False
                    )
                    if not results:
                        threshold = 0.4
                        results = segment_tree.semantic_search(
                            sem_query,
                            top_k=15,
                            threshold=threshold,
                            search_transcriptions=True,
                            search_unified=True,
                            verbose=False
                        )
                    if not results:
                        threshold = 0.35
                        results = segment_tree.semantic_search(
                            sem_query,
                            top_k=15,
                            threshold=threshold,
                            search_transcriptions=True,
                            search_unified=True,
                            verbose=False
                        )
                    elapsed = time.time() - start_time
                    
                    for result in results:
                        result["search_type"] = "semantic"
                        result["search_query"] = sem_query
                        all_search_results.append(result)
                    log_info(f"      Found {len(results)} matches (took {elapsed:.2f}s)")
            
            # 3. Object search (with negation handling)
            object_classes = search_plan.get("object_classes", [])
            # Also check semantic analysis for object classes
            semantic_objects = semantic_analysis.get("target_entities", {}).get("objects", [])
            all_object_classes = list(set(object_classes + semantic_objects))
            
            # Check if this is a negative query
            is_negative = semantic_analysis.get("query_type") == "NEGATIVE" or semantic_analysis.get("special_handling", {}).get("negation", False)
            
            if all_object_classes:
                log_info(f"\n  [SEARCH TYPE: Object Detection (YOLO)]")
                if is_negative:
                    log_info(f"    [NEGATION] Inverting search - finding seconds WITHOUT objects")
                
                if is_negative:
                    # NEGATIVE QUERY: Find seconds WITHOUT the objects
                    # Strategy: Get all seconds, then remove those with object detections
                    all_seconds_with_objects = set()
                    total_seconds = len(segment_tree.seconds)
                    
                    # Helper to map time to second index
                    def time_to_second_idx(time_seconds: float) -> Optional[int]:
                        """Map a time in seconds to the corresponding second index."""
                        for idx, second_data in enumerate(segment_tree.seconds):
                            tr = second_data.get("time_range", [])
                            if tr and len(tr) >= 2 and tr[0] <= time_seconds <= tr[1]:
                                return idx
                        # Fallback: approximate by rounding
                        return int(time_seconds) if 0 <= int(time_seconds) < total_seconds else None
                    
                    for obj_class in all_object_classes:
                        log_info(f"    Class: '{obj_class}' (inverted)")
                        results = segment_tree.find_objects_by_class(obj_class)
                        for result in results:
                            tr = result.get("time_range", [])
                            if tr and len(tr) >= 2:
                                # Map time to second index properly
                                start_idx = time_to_second_idx(tr[0])
                                end_idx = time_to_second_idx(tr[1])
                                # Mark all seconds in the range as having the object
                                if start_idx is not None and end_idx is not None:
                                    for idx in range(start_idx, end_idx + 1):
                                        if 0 <= idx < total_seconds:
                                            all_seconds_with_objects.add(idx)
                                elif start_idx is not None:
                                    all_seconds_with_objects.add(start_idx)
                    
                    # Create results for seconds WITHOUT objects
                    for second_idx in range(total_seconds):
                        if second_idx not in all_seconds_with_objects:
                            second_data = segment_tree.get_second_by_index(second_idx)
                            if second_data:
                                tr = second_data.get("time_range", [])
                                if tr and len(tr) >= 2:
                                    all_search_results.append({
                                        "search_type": "object_negated",
                                        "object_class": all_object_classes[0],  # Primary class
                                        "time_range": tr,
                                        "second": second_idx,
                                        "inverted": True
                                    })
                    
                    log_info(f"      Found {total_seconds - len(all_seconds_with_objects)} seconds WITHOUT objects (inverted from {len(all_seconds_with_objects)} with objects)")
                else:
                    # POSITIVE QUERY: Normal object detection
                    for obj_class in all_object_classes:
                        log_info(f"    Class: '{obj_class}'")
                        results = segment_tree.find_objects_by_class(obj_class)
                        for result in results:
                            result["search_type"] = "object"
                            result["object_class"] = obj_class
                            all_search_results.append(result)
                        log_info(f"      Found {len(results)} detections")
            
            # 4. Activity search (keep - pattern matching)
            activity_name = search_plan.get("activity_name", "")
            activity_keywords = search_plan.get("activity_keywords", [])
            evidence_keywords = search_plan.get("evidence_keywords", [])
            if activity_name or activity_keywords:
                log_info(f"\n  [SEARCH TYPE: Activity Pattern Matching]")
                if "fish" in query_lower and ("catch" in query_lower or "caught" in query_lower):
                    log_info(f"    Activity: Fish catching (specialized)")
                    result = segment_tree.check_fish_caught()
                    result["search_type"] = "activity"
                    # Add metadata for validation
                    result["activity_name"] = "fishing"
                    result["evidence_name"] = "fish caught"
                    # Validate evidence descriptions to filter false positives
                    log_info(f"      Evidence scenes before validation: {result.get('fish_holding_count', 0)}")
                    result = validate_activity_evidence(query, result, llm, verbose=verbose)
                    # Update fish-specific fields after validation
                    result["fish_holding_count"] = result.get("evidence_count", 0)
                    result["fish_caught"] = result.get("detected", False)
                    all_search_results.append(result)
                    log_info(f"      Evidence scenes after validation: {result.get('fish_holding_count', 0)}")
                elif activity_keywords:
                    log_info(f"    Activity: {activity_name or 'general'}")
                    result = segment_tree.check_activity(
                        activity_keywords=activity_keywords,
                        evidence_keywords=evidence_keywords or activity_keywords,
                        activity_name=activity_name or "activity"
                    )
                    result["search_type"] = "activity"
                    # Add metadata for validation
                    result["activity_name"] = activity_name or "activity"
                    result["evidence_name"] = activity_name or "activity"
                    # Validate evidence descriptions to filter false positives
                    log_info(f"      Evidence scenes before validation: {result.get('evidence_count', 0)}")
                    result = validate_activity_evidence(query, result, llm, verbose=verbose)
                    all_search_results.append(result)
                    log_info(f"      Evidence scenes after validation: {result.get('evidence_count', 0)}")
            
            log_info(f"\n[AGGREGATION] Collected results from all search types:")
            log_info(f"  Total results: {len(all_search_results)}")
            
            # NEW STEP 3: Score all seconds using weighted features
            log_info(f"\n[STEP 3] Scoring all seconds with weighted features...")
            log_info(f"  Total seconds to score: {len(segment_tree.seconds)}")
            start_time = time.time()
            
            # Separate results by type for scoring
            semantic_results = [r for r in all_search_results if r.get("search_type") == "semantic"]
            activity_results = [r for r in all_search_results if r.get("search_type") == "activity"]
            hierarchical_results = [r for r in all_search_results if r.get("search_type") in ["hierarchical", "hierarchical_highlight"]]
            negated_results = [r for r in all_search_results if r.get("search_type") == "object_negated"]
            
            # CORE FIX: For negative queries, only score the seconds identified by negation logic
            is_negative = semantic_analysis.get("query_type") == "NEGATIVE" or semantic_analysis.get("special_handling", {}).get("negation", False)
            negated_second_indices: Optional[Set[int]] = None
            
            if is_negative and negated_results:
                # Extract the second indices from negated results - these are the ONLY seconds to score
                negated_second_indices = set()
                for result in negated_results:
                    second_idx = result.get("second")
                    if second_idx is not None:
                        negated_second_indices.add(second_idx)
                
                log_info(f"\n[NEGATION] Only scoring {len(negated_second_indices)} seconds identified by negation logic")
            
            # Score seconds (only negated ones if negative query)
            scored_seconds = score_seconds(
                feature_extractor,
                query_intent,
                weights,
                semantic_results,
                activity_results,
                hierarchical_results,
                semantic_analysis=semantic_analysis,
                negated_second_indices=negated_second_indices,  # Pass the restricted set
                verbose=verbose
            )
            elapsed = time.time() - start_time
            log_info(f"[STEP 3] Scoring completed in {elapsed:.2f}s")
            
            # Filter by threshold and sort
            threshold = weights["threshold"]
            filtered_seconds = [s for s in scored_seconds if s["score"] >= threshold]
            filtered_seconds.sort(key=lambda x: x["score"], reverse=True)
            
            log_info(f"\n[FILTERING] Filtered to {len(filtered_seconds)} seconds above threshold {threshold:.2f}")
            if filtered_seconds:
                log_info(f"  Top 10 scored seconds:")
                for i, sec in enumerate(filtered_seconds[:10], 1):
                    log_info(f"    {i}. Second {sec['second']}: score={sec['score']:.3f} "
                              f"(sem={sec['semantic_score']:.2f}, act={sec['activity_score']:.2f}, "
                              f"obj={sec['object_score']:.2f})")
            
            # AGENTIC PHASE 3: Self-Validation
            log_info(f"\n[AGENTIC] Phase 3: Self-Validation...")
            
            validation_result = validate_search_results(
                query=query,
                semantic_analysis=semantic_analysis,
                filtered_seconds=filtered_seconds[:20] if filtered_seconds else [],  # Top 20 for validation
                llm=llm,
                verbose=verbose
            )
            
            if validation_result.get("needs_refinement", False):
                log_info(f"  [VALIDATION] Issues detected: {validation_result.get('issues', [])}")
                log_info(f"  [VALIDATION] Suggestions: {validation_result.get('suggestions', [])}")
            
            # NEW STEP 4: Group contiguous seconds into time ranges
            log_info(f"\n[STEP 4] Grouping contiguous seconds into time ranges...")
            
            time_ranges = group_contiguous_seconds(filtered_seconds, min_duration=1.0, gap_tolerance=1.0)
            
            log_info(f"  Grouped into {len(time_ranges)} time range(s)")
            if time_ranges:
                for i, (start, end) in enumerate(time_ranges, 1):
                    log_info(f"    {i}. {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
            
            # Determine if user wants one or multiple results
            query_lower = query.lower()
            
            # Indicators for multiple results (strong signals)
            multiple_indicators = [
                "all", "every", "each", "multiple", "several",
                "moments", "instances", "scenes", "clips", "times"
            ]
            has_multiple_indicator = any(indicator in query_lower for indicator in multiple_indicators)
            
            # Indicators for single result (strong signals)
            single_indicators = [
                "the moment", "the best", "the first", "the one",
                "a moment", "one moment", "single moment"
            ]
            has_single_indicator = any(indicator in query_lower for indicator in single_indicators)
            
            # Check for plural vs singular "moment"
            has_plural_moment = "moments" in query_lower
            has_singular_moment = "moment" in query_lower and "moments" not in query_lower
            
            # Determine intent
            if has_multiple_indicator:
                user_wants_one = False
                reason = "multiple indicator detected"
            elif has_single_indicator:
                user_wants_one = True
                reason = "single indicator detected"
            elif has_plural_moment:
                user_wants_one = False  # Plural suggests multiple
                reason = "plural 'moments' detected"
            elif has_singular_moment:
                user_wants_one = True  # Singular suggests one
                reason = "singular 'moment' detected"
            else:
                # Default: if we found many results, assume user wants multiple
                user_wants_one = len(time_ranges) <= 3
                reason = f"default (found {len(time_ranges)} ranges)"
            
            log_info(f"\n[SELECTION] User intent detection:")
            log_info(f"  Multiple indicators: {has_multiple_indicator}")
            log_info(f"  Single indicators: {has_single_indicator}")
            log_info(f"  Plural 'moments': {has_plural_moment}")
            log_info(f"  Singular 'moment': {has_singular_moment}")
            log_info(f"  Decision: {'ONE result' if user_wants_one else 'MULTIPLE results'} ({reason})")
            
            if user_wants_one and time_ranges:
                time_ranges = [time_ranges[0]]
                log_info(f"  [ACTION] Selecting top match only")
            elif not user_wants_one:
                log_info(f"  [ACTION] Returning all {len(time_ranges)} time ranges")
            
            # Calculate confidence based on scores
            if filtered_seconds:
                top_score = filtered_seconds[0]["score"]
                confidence = min(0.95, max(0.3, top_score))  # Map score to confidence
            else:
                confidence = 0.3
            
            # Store all search results for reference
            search_results = all_search_results
            
            # AUTOMATIC RERANKING AND FILTERING: Check if user specified "top N" or "all"
            # Extract number from query (e.g., "top 10", "first 5", "best 15", "proceed top 10")
            top_n_match = re.search(r'(?:proceed\s+)?(?:top|first|best|select)\s+(\d+)', query_lower)
            extract_number = None
            if top_n_match:
                extract_number = int(top_n_match.group(1))
                log_info(f"\n[AUTO-RERANK] Detected 'top {extract_number}' in query - will automatically rerank and filter")
            elif "all" in query_lower or "every" in query_lower:
                extract_number = None  # "all" means keep all, but still rerank
                log_info(f"\n[AUTO-RERANK] Detected 'all/every' in query - will automatically rerank and filter false positives")
            
            # If we have results and user specified a number or "all", automatically rerank and filter
            if time_ranges and (extract_number is not None or "all" in query_lower or "every" in query_lower):
                log_info(f"\n[AUTO-RERANK] Automatically reranking and filtering {len(time_ranges)} results...")
                
                # Map time ranges to their descriptions from search_results
                range_descriptions = {}
                for result in search_results:
                    result_tr = result.get("time_range", [])
                    if result_tr and len(result_tr) >= 2:
                        tr_key = (result_tr[0], result_tr[1])
                        # Get description (prefer unified_description, fallback to description or text)
                        desc = result.get("unified_description") or result.get("description") or result.get("text", "")
                        if desc:
                            if tr_key not in range_descriptions:
                                range_descriptions[tr_key] = []
                            range_descriptions[tr_key].append(desc)
                
                # Match time ranges with their descriptions
                ranges_with_descriptions = []
                for tr in time_ranges:
                    tr_key = (tr[0], tr[1])
                    descriptions = range_descriptions.get(tr_key, [])
                    # Combine descriptions or use first one
                    combined_desc = " | ".join(descriptions[:2]) if descriptions else f"Time range {tr[0]:.1f}s-{tr[1]:.1f}s"
                    ranges_with_descriptions.append({
                        "time_range": tr,
                        "description": combined_desc
                    })
                
                # Use LLM to rerank and filter
                if len(ranges_with_descriptions) > 1:  # Only rerank if we have multiple results
                    from agent.refinement import rank_ranges_with_llm
                    
                    # Determine top_n: use extract_number if specified, otherwise use all but filter false positives
                    top_n_for_rerank = extract_number if extract_number else len(ranges_with_descriptions)
                    
                    try:
                        ranked_ranges = rank_ranges_with_llm(
                            ranges_with_descriptions,
                            query,  # Use original query for ranking
                            query,  # No previous query in auto-rerank
                            top_n_for_rerank,
                            logger,
                            verbose=verbose
                        )
                        
                        # Extract time ranges from ranked results
                        reranked_time_ranges = [r["time_range"] for r in ranked_ranges]
                        
                        # Filter out false positives: use LLM to validate each result matches the query
                        log_info(f"  [AUTO-FILTER] Validating {len(reranked_time_ranges)} results to remove false positives...")
                        validated_ranges = []
                        
                        for r in ranked_ranges:
                            tr = r["time_range"]
                            desc = r["description"]
                            
                            # Quick validation prompt
                            validation_prompt = f"""Does this video clip match the query?

Query: "{query}"
Clip description: "{desc[:200]}"

Return JSON only:
{{"is_valid": true/false, "reasoning": "brief explanation"}}"""
                            
                            try:
                                validation_response = llm.invoke([
                                    SystemMessage(content="You are a video search validator. Determine if a clip description actually matches the user's query. Filter out false positives."),
                                    HumanMessage(content=validation_prompt)
                                ])
                                
                                validation_text = validation_response.content.strip()
                                validation_json = extract_json(validation_text)
                                validation_result = json.loads(validation_json)
                                
                                if validation_result.get("is_valid", True):  # Default to valid if validation fails
                                    validated_ranges.append(tr)
                                else:
                                    reasoning = validation_result.get("reasoning", "No reasoning")
                                    log_info(f"    Filtered out {tr[0]:.1f}s-{tr[1]:.1f}s: {reasoning[:60]}")
                            except Exception as e:
                                # On validation error, keep the result (don't filter)
                                validated_ranges.append(tr)
                                if verbose:
                                    log_info(f"    Validation error for {tr[0]:.1f}s-{tr[1]:.1f}s, keeping result")
                        
                        if validated_ranges:
                            time_ranges = validated_ranges
                            log_info(f"  [AUTO-RERANK] Reranked and filtered to {len(time_ranges)} valid results")
                        else:
                            log_info(f"  [AUTO-RERANK] Warning: All results filtered out, keeping original top results")
                            # Fallback: keep top N even if all filtered
                            time_ranges = reranked_time_ranges[:min(extract_number or 10, len(reranked_time_ranges))]
                    except Exception as e:
                        log_info(f"  [AUTO-RERANK] Error during reranking: {e}, using original results")
                        # Fallback: just take top N if specified
                        if extract_number and len(time_ranges) > extract_number:
                            time_ranges = time_ranges[:extract_number]
            
            log_info(f"\n[FINAL SELECTION]")
            log_info(f"  Selected {len(time_ranges)} time range(s)")
            log_info(f"  Confidence: {confidence:.2f}")
            if time_ranges:
                log_info(f"  Time ranges:")
                for i, (start, end) in enumerate(time_ranges, 1):
                    log_info(f"    {i}. {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
            
            if not time_ranges:
                confidence = 0.3
                needs_clarification = True
                clarification_question = "No relevant results found. Could you rephrase your query or provide more details?"
                log_info("  [WARNING] No results found - will ask for clarification")
            elif len(time_ranges) > 15:
                # Use LLM to decide if clarification is needed
                log_info(f"\n[CLARIFICATION DECISION] Found {len(time_ranges)} results - asking LLM if clarification needed...")
                
                from agent.prompts import CLARIFICATION_DECISION_PROMPT
                clarification_prompt = CLARIFICATION_DECISION_PROMPT.format(
                    query=query,
                    result_count=len(time_ranges)
                )
                
                try:
                    start_time = time.time()
                    clarification_response = llm.invoke([
                        SystemMessage(content="You are a helpful assistant that determines if users need clarification for video search queries."),
                        HumanMessage(content=clarification_prompt)
                    ])
                    elapsed = time.time() - start_time
                    
                    clarification_text = clarification_response.content.strip()
                    # Extract JSON
                    if "```json" in clarification_text:
                        clarification_text = clarification_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in clarification_text:
                        clarification_text = clarification_text.split("```")[1].split("```")[0].strip()
                    
                    clarification_decision = json.loads(clarification_text)
                    needs_clarification = clarification_decision.get("needs_clarification", False)
                    clarification_question = clarification_decision.get("clarification_question", f"Found {len(time_ranges)} potential moments. Could you narrow down what you're looking for?")
                    reasoning = clarification_decision.get("reasoning", "")
                    
                    log_info(f"  LLM Decision: {'Needs clarification' if needs_clarification else 'Proceed with all results'} (took {elapsed:.2f}s)")
                    log_info(f"  Reasoning: {reasoning}")
                    
                    if needs_clarification:
                        confidence = 0.6
                        log_info(f"  [WARNING] Too many results ({len(time_ranges)}) - will ask for clarification")
                    else:
                        confidence = min(0.95, max(0.6, confidence))
                        log_info(f"  [SUCCESS] User wants all results - proceeding with {len(time_ranges)} time range(s)")
                            
                except Exception as e:
                    # Fallback to default behavior if LLM call fails
                    log_info(f"  [FALLBACK] LLM clarification decision failed: {e}, using default logic")
                    # Default: ask for clarification if > 15 results
                    confidence = 0.6
                    needs_clarification = True
                    clarification_question = f"Found {len(time_ranges)} potential moments. Could you narrow down what you're looking for?"
                    log_info(f"  [WARNING] Too many results ({len(time_ranges)}) - will ask for clarification")
            else:
                log_info(f"  [SUCCESS] Found {len(time_ranges)} time range(s) - proceeding with extraction")
            
        # Merge overlapping time ranges
        if time_ranges:
            original_count = len(time_ranges)
            log_info(f"\n[MERGING] Merging {original_count} time ranges...")
            log_info(f"  Before merge: {original_count} ranges")
            time_ranges = merge_time_ranges(time_ranges)
            log_info(f"  After merge: {len(time_ranges)} non-overlapping ranges")
            log_info(f"  Merged ranges with 2.0s padding:")
            for i, (start, end) in enumerate(time_ranges, 1):
                log_info(f"    Range {i}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
        else:
            log_info("\n[MERGING] No time ranges to merge")
        
        # Preserve state for potential refinement (preserve even if clarification needed)
        # This allows user to refine results after clarification
        preserved_scored_seconds = filtered_seconds if filtered_seconds else state.get("previous_scored_seconds")
        
        return {
            **state,
            "query_type": "multi_modal",  # Indicate we used all search types
            "search_results": search_results,
            "time_ranges": time_ranges,
            "confidence": confidence,
            "needs_clarification": needs_clarification,
            "clarification_question": clarification_question,
            # Preserve for future refinement (always preserve if we have new results)
            "previous_time_ranges": time_ranges if time_ranges else state.get("previous_time_ranges"),
            "previous_scored_seconds": preserved_scored_seconds,
            "previous_query": query if time_ranges else state.get("previous_query"),
            "previous_search_results": search_results if search_results else state.get("previous_search_results")
        }
    
    return planner_node

