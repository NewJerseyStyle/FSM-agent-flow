"""
Memory system for LLM State Machine.
Provides 10-bucket working memory and persistent storage.
"""

from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json


@dataclass
class Bucket:
    """A single memory bucket with content and metadata."""
    content: str = ""
    updated_at: Optional[datetime] = None
    
    def is_empty(self) -> bool:
        return not self.content.strip()
    
    def update(self, content: str):
        self.content = content
        self.updated_at = datetime.now()


@dataclass
class WorkingMemory:
    """10-bucket working memory system for LLM context."""
    buckets: List[Bucket] = field(default_factory=lambda: [Bucket() for _ in range(10)])
    
    def set(self, index: int, content: str):
        """Set content in a specific bucket (0-9)."""
        if not 0 <= index <= 9:
            raise ValueError(f"Bucket index must be 0-9, got {index}")
        self.buckets[index].update(content)
    
    def get(self, index: int) -> str:
        """Get content from a specific bucket."""
        if not 0 <= index <= 9:
            raise ValueError(f"Bucket index must be 0-9, got {index}")
        return self.buckets[index].content
    
    def append(self, index: int, content: str):
        """Append content to existing bucket content."""
        if not 0 <= index <= 9:
            raise ValueError(f"Bucket index must be 0-9, got {index}")
        current = self.buckets[index].content
        new_content = f"{current}\n{content}".strip()
        self.buckets[index].update(new_content)
    
    def delete(self, index: int):
        """Clear a specific bucket."""
        if not 0 <= index <= 9:
            raise ValueError(f"Bucket index must be 0-9, got {index}")
        self.buckets[index] = Bucket()
    
    def get_non_empty(self) -> Dict[int, str]:
        """Get all non-empty buckets as dict."""
        return {
            i: bucket.content 
            for i, bucket in enumerate(self.buckets) 
            if not bucket.is_empty()
        }
    
    def to_context(self) -> str:
        """Format non-empty buckets for LLM context."""
        non_empty = self.get_non_empty()
        if not non_empty:
            return ""
        
        lines = ["## Working Memory:"]
        for idx, content in non_empty.items():
            lines.append(f"[Bucket {idx}]: {content}")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "buckets": [
                {
                    "content": b.content,
                    "updated_at": b.updated_at.isoformat() if b.updated_at else None
                }
                for b in self.buckets
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WorkingMemory':
        """Deserialize from dictionary."""
        buckets = []
        for b_data in data.get("buckets", []):
            bucket = Bucket(
                content=b_data.get("content", ""),
                updated_at=datetime.fromisoformat(b_data["updated_at"]) if b_data.get("updated_at") else None
            )
            buckets.append(bucket)
        # Ensure we have exactly 10 buckets
        while len(buckets) < 10:
            buckets.append(Bucket())
        return cls(buckets=buckets[:10])


@dataclass
class BackgroundContext:
    """Persistent background context for the FSM."""
    todo_list: str = ""
    goals: str = ""
    vision_mission: str = ""
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_context(self) -> str:
        """Format background context for LLM."""
        lines = ["## Background Context:"]
        if self.vision_mission:
            lines.append(f"Vision/Mission: {self.vision_mission}")
        if self.goals:
            lines.append(f"Goals: {self.goals}")
        if self.todo_list:
            lines.append(f"TODO List: {self.todo_list}")
        for key, value in self.custom_fields.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BackgroundContext':
        return cls(**data)


@dataclass
class StateHistory:
    """History of state transitions and I/O."""
    entries: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_entry(self, state_name: str, input_data: Any, output_data: Any, 
                  custom_fields: Optional[Dict[str, Any]] = None):
        """Add a state execution entry."""
        entry = {
            "state": state_name,
            "timestamp": datetime.now().isoformat(),
            "input": input_data,
            "output": output_data,
            "custom_fields": custom_fields or {}
        }
        self.entries.append(entry)
    
    def get_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get n most recent entries."""
        return self.entries[-n:]
    
    def to_context(self, n: int = 5) -> str:
        """Format recent history for LLM context."""
        recent = self.get_recent(n)
        if not recent:
            return ""
        
        lines = ["## Recent History:"]
        for entry in recent:
            lines.append(f"[{entry['state']}] Input: {entry['input']} → Output: {entry['output']}")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        return {"entries": self.entries}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StateHistory':
        return cls(entries=data.get("entries", []))


@dataclass
class PersistentMemory:
    """Complete persistent memory system."""
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    background: BackgroundContext = field(default_factory=BackgroundContext)
    history: StateHistory = field(default_factory=StateHistory)
    
    def to_context(self, history_n: int = 5) -> str:
        """Generate complete context for LLM."""
        sections = [
            self.background.to_context(),
            self.working_memory.to_context(),
            self.history.to_context(history_n)
        ]
        return "\n\n".join(s for s in sections if s)
    
    def to_dict(self) -> Dict:
        """Serialize complete memory."""
        return {
            "working_memory": self.working_memory.to_dict(),
            "background": self.background.to_dict(),
            "history": self.history.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PersistentMemory':
        """Deserialize complete memory."""
        return cls(
            working_memory=WorkingMemory.from_dict(data.get("working_memory", {})),
            background=BackgroundContext.from_dict(data.get("background", {})),
            history=StateHistory.from_dict(data.get("history", {}))
        )
    
    def save_to_file(self, filepath: str):
        """Save memory to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'PersistentMemory':
        """Load memory from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save_to_mongodb(self, collection, document_id: str):
        """Save memory to MongoDB."""
        doc = self.to_dict()
        doc['_id'] = document_id
        collection.replace_one({'_id': document_id}, doc, upsert=True)
    
    @classmethod
    def load_from_mongodb(cls, collection, document_id: str) -> Optional['PersistentMemory']:
        """Load memory from MongoDB."""
        doc = collection.find_one({'_id': document_id})
        if doc:
            doc.pop('_id', None)
            return cls.from_dict(doc)
        return None


# Memory tools for LLM to use
def create_memory_tools():
    """Create tools that LLM can use to manipulate working memory."""
    
    def set_memory(bucket_index: int, content: str, memory: PersistentMemory) -> str:
        """Set content in a working memory bucket (0-9)."""
        try:
            memory.working_memory.set(bucket_index, content)
            return f"✓ Bucket {bucket_index} updated"
        except ValueError as e:
            return f"✗ Error: {e}"
    
    def append_memory(bucket_index: int, content: str, memory: PersistentMemory) -> str:
        """Append content to a working memory bucket."""
        try:
            memory.working_memory.append(bucket_index, content)
            return f"✓ Content appended to bucket {bucket_index}"
        except ValueError as e:
            return f"✗ Error: {e}"
    
    def get_memory(bucket_index: int, memory: PersistentMemory) -> str:
        """Get content from a working memory bucket."""
        try:
            content = memory.working_memory.get(bucket_index)
            return content if content else f"Bucket {bucket_index} is empty"
        except ValueError as e:
            return f"✗ Error: {e}"
    
    def clear_memory(bucket_index: int, memory: PersistentMemory) -> str:
        """Clear a working memory bucket."""
        try:
            memory.working_memory.delete(bucket_index)
            return f"✓ Bucket {bucket_index} cleared"
        except ValueError as e:
            return f"✗ Error: {e}"
    
    def view_all_memory(memory: PersistentMemory) -> str:
        """View all non-empty working memory buckets."""
        non_empty = memory.working_memory.get_non_empty()
        if not non_empty:
            return "All memory buckets are empty"
        
        lines = ["Working Memory Contents:"]
        for idx, content in non_empty.items():
            lines.append(f"[Bucket {idx}]: {content}")
        return "\n".join(lines)
    
    return {
        "set_memory": set_memory,
        "append_memory": append_memory,
        "get_memory": get_memory,
        "clear_memory": clear_memory,
        "view_all_memory": view_all_memory
    }
