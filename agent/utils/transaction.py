"""Transaction support for timeline operations with rollback capability."""

import copy
from datetime import datetime
from typing import Dict, List, Optional, Any
from agent.timeline_manager import TimelineManager


class TimelineTransaction:
    """
    Manages timeline state for atomic operations with rollback capability.
    
    Usage:
        with TimelineTransaction(timeline_manager) as tx:
            tx.add_operation(operation_result)
            result = execute_operation(state, operation_params)
            if result["success"]:
                tx.commit()
            # Automatic rollback on exception or if not committed
    """
    
    def __init__(self, timeline_manager: TimelineManager, verbose: bool = False):
        """
        Initialize transaction.
        
        Args:
            timeline_manager: TimelineManager instance to manage
            verbose: Whether to print verbose output
        """
        self.timeline_manager = timeline_manager
        self.verbose = verbose
        self.backup = copy.deepcopy(timeline_manager.chunks)
        self.backup_timeline_data = copy.deepcopy(timeline_manager.timeline_data) if timeline_manager.timeline_data else None
        self.operations: List[Dict[str, Any]] = []
        self.committed = False
        self.rollback_called = False
    
    def add_operation(self, operation: Dict[str, Any]):
        """
        Track operation for rollback and history.
        
        Args:
            operation: Operation dictionary with 'operation' and 'parameters' keys
        """
        self.operations.append({
            "type": operation.get("operation", "UNKNOWN"),
            "params": operation.get("parameters", {}),
            "timestamp": datetime.now().isoformat()
        })
        
        if self.verbose:
            print(f"[TRANSACTION] Added operation: {operation.get('operation', 'UNKNOWN')}")
    
    def commit(self) -> bool:
        """
        Save changes permanently.
        
        Returns:
            True if successful, False otherwise
        """
        if self.committed:
            if self.verbose:
                print("[TRANSACTION] Already committed, skipping")
            return True
        
        if self.rollback_called:
            if self.verbose:
                print("[TRANSACTION] Cannot commit after rollback")
            return False
        
        # Validate timeline before committing
        is_valid, errors = self.timeline_manager.validate_timeline()
        if not is_valid:
            if self.verbose:
                print(f"[TRANSACTION] Validation failed, cannot commit:")
                for error in errors:
                    print(f"  - {error}")
            return False
        
        # Save timeline
        success = self.timeline_manager.save()
        if success:
            self.committed = True
            if self.verbose:
                print(f"[TRANSACTION] Committed {len(self.operations)} operation(s)")
        else:
            if self.verbose:
                print("[TRANSACTION] Failed to save timeline")
        
        return success
    
    def rollback(self) -> bool:
        """
        Restore previous state.
        
        Returns:
            True if rollback successful, False otherwise
        """
        if self.committed:
            if self.verbose:
                print("[TRANSACTION] Cannot rollback after commit")
            return False
        
        if self.rollback_called:
            if self.verbose:
                print("[TRANSACTION] Already rolled back")
            return True
        
        # Restore backup
        self.timeline_manager.chunks = copy.deepcopy(self.backup)
        if self.backup_timeline_data:
            self.timeline_manager.timeline_data = copy.deepcopy(self.backup_timeline_data)
        
        # Save rolled back state
        success = self.timeline_manager.save()
        if success:
            self.rollback_called = True
            if self.verbose:
                print(f"[TRANSACTION] Rolled back {len(self.operations)} operation(s)")
        else:
            if self.verbose:
                print("[TRANSACTION] Failed to save rolled back timeline")
        
        return success
    
    def get_operations(self) -> List[Dict[str, Any]]:
        """
        Get list of tracked operations.
        
        Returns:
            List of operation dictionaries
        """
        return copy.deepcopy(self.operations)
    
    def is_committed(self) -> bool:
        """Check if transaction is committed."""
        return self.committed
    
    def is_rolled_back(self) -> bool:
        """Check if transaction was rolled back."""
        return self.rollback_called
    
    def __enter__(self):
        """Context manager entry."""
        if self.verbose:
            print("[TRANSACTION] Starting transaction")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit.
        
        Automatically rolls back if:
        - An exception occurred (exc_type is not None)
        - Transaction was not committed
        """
        if exc_type is not None:
            # Exception occurred, rollback
            if self.verbose:
                print(f"[TRANSACTION] Exception occurred: {exc_type.__name__}, rolling back")
            self.rollback()
            return False  # Don't suppress exception
        
        if not self.committed and not self.rollback_called:
            # No exception but not committed, rollback
            if self.verbose:
                print("[TRANSACTION] Transaction not committed, rolling back")
            self.rollback()
        
        return False  # Don't suppress exceptions


def execute_with_transaction(
    timeline_manager: TimelineManager,
    operation_func,
    operation_params: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Execute an operation within a transaction.
    
    This is a convenience function that wraps operation execution in a transaction.
    
    Args:
        timeline_manager: TimelineManager instance
        operation_func: Function to execute (should return dict with 'success' key)
        operation_params: Parameters to pass to operation function
        verbose: Whether to print verbose output
        
    Returns:
        Result dictionary with 'success' key and transaction info
    """
    with TimelineTransaction(timeline_manager, verbose=verbose) as tx:
        try:
            # Execute operation
            result = operation_func(operation_params)
            
            # Check if operation was successful
            if result.get("success", False):
                # Validate before committing
                is_valid, errors = timeline_manager.validate_timeline()
                if is_valid:
                    tx.commit()
                    result["transaction"] = {
                        "committed": True,
                        "operations": len(tx.get_operations())
                    }
                else:
                    # Validation failed, rollback
                    if verbose:
                        print(f"[TRANSACTION] Validation failed after operation:")
                        for error in errors:
                            print(f"  - {error}")
                    tx.rollback()
                    result["success"] = False
                    result["error"] = f"Validation failed: {', '.join(errors)}"
                    result["transaction"] = {
                        "committed": False,
                        "rolled_back": True,
                        "validation_errors": errors
                    }
            else:
                # Operation failed, rollback
                tx.rollback()
                result["transaction"] = {
                    "committed": False,
                    "rolled_back": True,
                    "reason": "Operation failed"
                }
            
            return result
            
        except Exception as e:
            # Exception occurred, transaction will auto-rollback
            if verbose:
                print(f"[TRANSACTION] Exception in operation: {e}")
            return {
                "success": False,
                "error": str(e),
                "transaction": {
                    "committed": False,
                    "rolled_back": True,
                    "exception": type(e).__name__
                }
            }

