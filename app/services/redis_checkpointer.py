"""
Redis-based checkpointer for LangGraph persistent memory.
Stores conversation state and enables multi-turn conversations.
"""

import json
import logging
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import redis
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


class RedisCheckpointSaver(BaseCheckpointSaver):
    """
    Redis-based checkpoint saver for LangGraph.
    Stores conversation state with TTL for automatic cleanup.
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        ttl_seconds: int = 86400 * 7  # 7 days default
    ):
        """
        Initialize Redis checkpointer.
        
        Args:
            redis_client: Redis client instance
            ttl_seconds: Time-to-live for checkpoints (default: 7 days)
        """
        self.redis = redis_client
        self.ttl = ttl_seconds
    
    def _make_key(self, thread_id: str, checkpoint_id: str = "latest") -> str:
        """Create Redis key for checkpoint."""
        return f"checkpoint:{thread_id}:{checkpoint_id}"
    
    def _make_metadata_key(self, thread_id: str) -> str:
        """Create Redis key for thread metadata."""
        return f"thread_metadata:{thread_id}"
    
    def _make_index_key(self, thread_id: str) -> str:
        """Create Redis key for checkpoint index (ordered list of checkpoint IDs)."""
        return f"checkpoint_index:{thread_id}"
    
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: Dict[str, Any]
    ) -> RunnableConfig:
        """
        Save a checkpoint to Redis.
        
        Args:
            config: Configuration containing thread_id
            checkpoint: The checkpoint data to save
            metadata: Additional metadata
            
        Returns:
            Updated config with checkpoint_id
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint.get("id", datetime.utcnow().isoformat())
        
        # Store checkpoint data
        key = self._make_key(thread_id, checkpoint_id)
        data = {
            "checkpoint": checkpoint,
            "metadata": metadata,
            "created_at": datetime.utcnow().isoformat()
        }
        
        self.redis.setex(
            key,
            self.ttl,
            json.dumps(data, default=str)
        )
        
        # Update latest pointer
        latest_key = self._make_key(thread_id, "latest")
        self.redis.setex(latest_key, self.ttl, checkpoint_id)
        
        # Add to checkpoint index
        index_key = self._make_index_key(thread_id)
        self.redis.zadd(
            index_key,
            {checkpoint_id: datetime.utcnow().timestamp()}
        )
        self.redis.expire(index_key, self.ttl)
        
        # Update thread metadata
        self._update_thread_metadata(thread_id, metadata)
        
        logger.info(f"Saved checkpoint {checkpoint_id} for thread {thread_id}")
        
        # Return updated config
        new_config = config.copy()
        new_config["configurable"]["checkpoint_id"] = checkpoint_id
        return new_config
    
    def get(self, config: RunnableConfig) -> Optional[Checkpoint]:
        """
        Retrieve a checkpoint from Redis.
        
        Args:
            config: Configuration containing thread_id and optional checkpoint_id
            
        Returns:
            Checkpoint data or None if not found
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id", "latest")
        
        # If requesting latest, get the checkpoint_id first
        if checkpoint_id == "latest":
            latest_key = self._make_key(thread_id, "latest")
            checkpoint_id_bytes = self.redis.get(latest_key)
            if not checkpoint_id_bytes:
                return None
            checkpoint_id = checkpoint_id_bytes.decode("utf-8")
        
        # Get checkpoint data
        key = self._make_key(thread_id, checkpoint_id)
        data_bytes = self.redis.get(key)
        
        if not data_bytes:
            return None
        
        data = json.loads(data_bytes.decode("utf-8"))
        return data["checkpoint"]
    
    def list(self, thread_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List all checkpoints for a thread.
        
        Args:
            thread_id: Thread ID to list checkpoints for
            limit: Maximum number of checkpoints to return
            
        Returns:
            List of checkpoint metadata
        """
        index_key = self._make_index_key(thread_id)
        
        # Get checkpoint IDs in reverse chronological order
        checkpoint_ids = self.redis.zrevrange(index_key, 0, limit - 1)
        
        checkpoints = []
        for checkpoint_id_bytes in checkpoint_ids:
            checkpoint_id = checkpoint_id_bytes.decode("utf-8")
            key = self._make_key(thread_id, checkpoint_id)
            data_bytes = self.redis.get(key)
            
            if data_bytes:
                data = json.loads(data_bytes.decode("utf-8"))
                checkpoints.append({
                    "checkpoint_id": checkpoint_id,
                    "created_at": data["created_at"],
                    "metadata": data["metadata"]
                })
        
        return checkpoints
    
    def _update_thread_metadata(self, thread_id: str, metadata: Dict[str, Any]):
        """Update metadata for a thread."""
        metadata_key = self._make_metadata_key(thread_id)
        
        # Get existing metadata
        existing_bytes = self.redis.get(metadata_key)
        if existing_bytes:
            thread_data = json.loads(existing_bytes.decode("utf-8"))
        else:
            thread_data = {
                "thread_id": thread_id,
                "created_at": datetime.utcnow().isoformat(),
                "message_count": 0
            }
        
        # Update metadata
        thread_data["updated_at"] = datetime.utcnow().isoformat()
        thread_data["message_count"] = thread_data.get("message_count", 0) + 1
        thread_data["last_metadata"] = metadata
        
        # Save back
        self.redis.setex(
            metadata_key,
            self.ttl,
            json.dumps(thread_data, default=str)
        )
    
    def get_thread_metadata(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a thread."""
        metadata_key = self._make_metadata_key(thread_id)
        data_bytes = self.redis.get(metadata_key)
        
        if not data_bytes:
            return None
        
        return json.loads(data_bytes.decode("utf-8"))
    
    def delete_thread(self, thread_id: str) -> bool:
        """
        Delete all checkpoints for a thread.
        
        Args:
            thread_id: Thread ID to delete
            
        Returns:
            True if deleted, False if thread didn't exist
        """
        # Get all checkpoint IDs
        index_key = self._make_index_key(thread_id)
        checkpoint_ids = self.redis.zrange(index_key, 0, -1)
        
        if not checkpoint_ids:
            return False
        
        # Delete all checkpoints
        keys_to_delete = [
            self._make_key(thread_id, cp_id.decode("utf-8"))
            for cp_id in checkpoint_ids
        ]
        
        # Add metadata and index keys
        keys_to_delete.extend([
            self._make_key(thread_id, "latest"),
            self._make_metadata_key(thread_id),
            index_key
        ])
        
        self.redis.delete(*keys_to_delete)
        
        logger.info(f"Deleted thread {thread_id} with {len(checkpoint_ids)} checkpoints")
        return True
    
    def list_threads(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List all active threads.
        
        Args:
            limit: Maximum number of threads to return
            
        Returns:
            List of thread metadata
        """
        # Scan for thread metadata keys
        cursor = 0
        threads = []
        pattern = "thread_metadata:*"
        
        while True:
            cursor, keys = self.redis.scan(
                cursor=cursor,
                match=pattern,
                count=100
            )
            
            for key in keys:
                data_bytes = self.redis.get(key)
                if data_bytes:
                    threads.append(json.loads(data_bytes.decode("utf-8")))
            
            if cursor == 0 or len(threads) >= limit:
                break
        
        # Sort by updated_at (most recent first)
        threads.sort(
            key=lambda x: x.get("updated_at", x.get("created_at", "")),
            reverse=True
        )
        
        return threads[:limit]


class ConversationManager:
    """High-level API for managing conversations."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.checkpointer = RedisCheckpointSaver(redis_client)
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            user_id: Optional user ID to associate with session
            
        Returns:
            New thread_id
        """
        import uuid
        thread_id = f"thread_{uuid.uuid4().hex[:16]}"
        
        # Initialize metadata
        metadata = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "status": "active"
        }
        
        metadata_key = f"thread_metadata:{thread_id}"
        self.redis.setex(
            metadata_key,
            self.checkpointer.ttl,
            json.dumps(metadata, default=str)
        )
        
        logger.info(f"Created new session {thread_id}")
        return thread_id
    
    def get_session(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get session metadata."""
        return self.checkpointer.get_thread_metadata(thread_id)
    
    def get_conversation_history(
        self,
        thread_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a thread.
        
        Args:
            thread_id: Thread ID
            limit: Maximum number of messages
            
        Returns:
            List of messages in chronological order
        """
        checkpoints = self.checkpointer.list(thread_id, limit=limit)
        
        messages = []
        for cp in checkpoints:
            # Extract messages from checkpoint metadata
            if "messages" in cp.get("metadata", {}):
                messages.extend(cp["metadata"]["messages"])
        
        return messages
    
    def delete_session(self, thread_id: str) -> bool:
        """Delete a conversation session."""
        return self.checkpointer.delete_thread(thread_id)
    
    def list_sessions(
        self,
        user_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        List all sessions, optionally filtered by user.
        
        Args:
            user_id: Optional user ID to filter by
            limit: Maximum number of sessions
            
        Returns:
            List of session metadata
        """
        threads = self.checkpointer.list_threads(limit=limit * 2)  # Get extra for filtering
        
        # Filter by user_id if provided
        if user_id:
            threads = [
                t for t in threads
                if t.get("last_metadata", {}).get("user_id") == user_id
            ]
        
        return threads[:limit]
