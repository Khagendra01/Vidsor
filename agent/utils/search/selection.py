"""Selection and diversity functions for video segments."""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np


def select_diverse_highlights(
    scored_seconds: List[Dict],
    segment_tree,
    max_seconds: Optional[int] = None,
    min_score: float = 0.5,
    diversity_threshold: float = 0.85,  # Increased: only filter if VERY similar (>0.85)
    verbose: bool = False
) -> List[Dict]:
    """
    Select diverse highlights using semantic similarity to avoid redundancy.
    
    Uses Maximal Marginal Relevance (MMR) algorithm:
    1. Start with highest scoring second
    2. Iteratively add seconds that are both high-scoring AND diverse from already selected
    
    Args:
        scored_seconds: List of scored seconds (sorted by score descending)
        segment_tree: SegmentTreeQuery instance for getting descriptions
        max_seconds: Maximum number of seconds to select (None = no limit)
        min_score: Minimum score to consider
        diversity_threshold: Minimum semantic similarity to consider redundant (0-1)
        verbose: Whether to print verbose output
        
    Returns:
        List of selected seconds (diverse, high-scoring)
    """
    if not scored_seconds:
        return []
    
    # Filter by minimum score
    filtered = [s for s in scored_seconds if s.get("score", 0) >= min_score]
    if not filtered:
        return []
    
    # If no max_seconds specified, use reasonable default (10-15% of video)
    if max_seconds is None:
        total_seconds = len(segment_tree.seconds) if segment_tree else 600
        max_seconds = max(30, int(total_seconds * 0.12))  # ~12% of video
    
    if verbose:
        print(f"[DIVERSITY SELECTION] Selecting up to {max_seconds} diverse seconds from {len(filtered)} candidates")
    
    # Get embeddings for descriptions (for similarity calculation)
    selected = []
    selected_embeddings = []  # Store embeddings for selected seconds
    
    # Try to get embedding model for similarity calculation
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        use_embeddings = True
    except:
        use_embeddings = False
        if verbose:
            print("[DIVERSITY SELECTION] Embedding model not available, using simple distance-based diversity")
    
    for sec in filtered:
        if len(selected) >= max_seconds:
            break
        
        second_idx = sec.get("second")
        if second_idx is None:
            continue
        
        # Get description for this second
        second_data = segment_tree.get_second_by_index(second_idx) if segment_tree else None
        if not second_data:
            continue
        
        # Get description (prefer unified, fallback to transcription)
        description = second_data.get("unified_description", "")
        if not description:
            # Try to get transcription from segment tree
            tr = sec.get("time_range", [])
            if tr and len(tr) >= 2 and segment_tree:
                # Get transcriptions that overlap with this time range
                all_transcriptions = segment_tree.transcriptions if hasattr(segment_tree, 'transcriptions') else []
                for trans in all_transcriptions:
                    trans_tr = trans.get("time_range", [])
                    if trans_tr and len(trans_tr) >= 2:
                        # Check if transcription overlaps with second's time range
                        if trans_tr[0] <= tr[1] and trans_tr[1] >= tr[0]:
                            trans_text = trans.get("transcription", "")
                            if trans_text:
                                description = trans_text
                                break
        
        if not description:
            # No description available, just add it (can't check diversity)
            selected.append(sec)
            continue
        
        # Check diversity against already selected seconds
        # IMPORTANT: Only filter if BOTH conditions are true:
        # 1. Very high semantic similarity (>0.85, not 0.75) AND
        # 2. Close in time (<5 seconds apart)
        # This prevents filtering different narrative moments that happen to have similar descriptions
        is_diverse = True
        current_time = sec.get("time_range", [0])[0]
        
        if use_embeddings and selected_embeddings:
            try:
                # Embed current description
                current_embedding = model.encode([description], convert_to_numpy=True)[0]
                
                # Check similarity AND time distance with all selected
                for i, selected_emb in enumerate(selected_embeddings):
                    # Calculate semantic similarity
                    similarity = np.dot(current_embedding, selected_emb) / (
                        np.linalg.norm(current_embedding) * np.linalg.norm(selected_emb)
                    )
                    
                    # Calculate time distance
                    selected_sec = selected[i]
                    selected_time = selected_sec.get("time_range", [0])[0]
                    time_distance = abs(current_time - selected_time)
                    
                    # Only filter if BOTH very similar (>0.85) AND very close in time (<5s)
                    # This prevents filtering different moments (e.g., two different hospital scenes)
                    if similarity >= 0.85 and time_distance < 5.0:
                        is_diverse = False
                        if verbose:
                            print(f"  Filtering redundant: similarity={similarity:.3f}, time_dist={time_distance:.1f}s")
                        break
                
                if is_diverse:
                    selected_embeddings.append(current_embedding)
            except Exception as e:
                if verbose:
                    print(f"[DIVERSITY SELECTION] Error calculating similarity: {e}, adding anyway")
        else:
            # Fallback: Only filter if VERY close in time (<3 seconds) AND lower score
            # Don't filter just because they're close - they might be different moments
            if selected:
                min_time_distance = min(
                    abs(current_time - s.get("time_range", [0])[0])
                    for s in selected
                )
                # Only filter if VERY close (<3s) AND significantly lower score
                # This is conservative - prefer keeping moments over filtering
                if min_time_distance < 3.0 and sec.get("score", 0) < 0.6:
                    is_diverse = False
                    if verbose:
                        print(f"  Filtering very close moment: time_dist={min_time_distance:.1f}s, score={sec.get('score', 0):.3f}")
        
        if is_diverse:
            selected.append(sec)
            if verbose and len(selected) % 10 == 0:
                print(f"  Selected {len(selected)}/{max_seconds} diverse seconds...")
    
    if verbose:
        print(f"[DIVERSITY SELECTION] Selected {len(selected)} diverse seconds")
        if selected:
            score_range = (selected[-1].get("score", 0), selected[0].get("score", 0))
            print(f"  Score range: {score_range[0]:.3f} - {score_range[1]:.3f}")
    
    return selected


def select_best_of(
    scored_seconds: List[Dict],
    top_k: Optional[int] = None,
    min_score: float = 0.5,
    prioritize_semantic: bool = True,
    verbose: bool = False
) -> List[Dict]:
    """
    Select best seconds with optional semantic prioritization and score normalization.
    
    Args:
        scored_seconds: List of scored seconds (should be sorted by score)
        top_k: Maximum number to select (None = all above threshold)
        min_score: Minimum score threshold
        prioritize_semantic: If True, prioritize seconds with high semantic scores
        verbose: Whether to print verbose output
        
    Returns:
        List of best seconds
    """
    if not scored_seconds:
        return []
    
    # Filter by minimum score
    filtered = [s for s in scored_seconds if s.get("score", 0) >= min_score]
    
    if not filtered:
        return []
    
    # Normalize scores to [0, 1] range for better comparison
    if len(filtered) > 1:
        max_score = max(s.get("score", 0) for s in filtered)
        min_score_val = min(s.get("score", 0) for s in filtered)
        score_range = max_score - min_score_val
        
        if score_range > 0:
            for sec in filtered:
                # Normalize to [0, 1]
                normalized = (sec.get("score", 0) - min_score_val) / score_range
                sec["normalized_score"] = normalized
    
    if prioritize_semantic:
        # Re-sort: prioritize semantic score, then total score
        # Use composite score: 60% semantic, 40% total score (less aggressive)
        # This ensures semantic relevance but doesn't completely ignore other signals
        for sec in filtered:
            sem_score = sec.get("semantic_score", 0)
            total_score = sec.get("normalized_score", sec.get("score", 0))
            # Composite: balance semantic and total score
            sec["composite_score"] = 0.6 * sem_score + 0.4 * total_score
        
        filtered.sort(key=lambda x: -x.get("composite_score", 0))
        if verbose:
            print(f"[BEST-OF] Prioritizing semantic relevance (60% semantic, 40% total score)")
    else:
        # Just use score (should already be sorted)
        filtered.sort(key=lambda x: -x.get("score", 0))
    
    if top_k is not None:
        filtered = filtered[:top_k]
    
    if verbose:
        print(f"[BEST-OF] Selected {len(filtered)} best seconds (min_score={min_score:.2f})")
        if filtered and prioritize_semantic:
            top_sem = filtered[0].get("semantic_score", 0)
            print(f"  Top semantic score: {top_sem:.3f}")
    
    return filtered

