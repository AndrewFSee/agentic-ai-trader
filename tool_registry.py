"""
Unified Tool Registry
=====================

Single source of truth for the ToolSpec type and registry helpers.
All tool modules (polygon_tools, sentiment_tools, research_tools, tools)
should import from here instead of defining their own.
"""

from typing import Dict, List, Optional, TypedDict, Callable, Any


class ToolSpec(TypedDict):
    """Standard specification for a tool in the agent's toolkit."""
    name: str
    description: str
    parameters: Dict
    fn: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


class ToolRegistry:
    """
    Unified registry with O(1) lookup by name.
    
    Usage:
        registry = ToolRegistry()
        registry.register({"name": "my_tool", "description": "...", "parameters": {...}, "fn": my_fn})
        fn = registry.get_function("my_tool")
        all_specs = registry.get_all_specs()  # without fn for JSON serialization
    """
    
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}
    
    def register(self, spec: ToolSpec) -> None:
        """Register a tool. Overwrites if name already exists."""
        self._tools[spec["name"]] = spec
    
    def get_function(self, name: str) -> Optional[Callable]:
        """Get the callable for a tool by name. O(1) lookup."""
        tool = self._tools.get(name)
        return tool["fn"] if tool else None
    
    def get_spec(self, name: str) -> Optional[ToolSpec]:
        """Get full spec for a tool by name."""
        return self._tools.get(name)
    
    def get_all_specs(self) -> List[Dict]:
        """Get all tool specs without fn field (safe for JSON serialization)."""
        return [
            {k: v for k, v in tool.items() if k != "fn"}
            for tool in self._tools.values()
        ]
    
    def get_all_tools(self) -> List[ToolSpec]:
        """Get all tool specs including fn."""
        return list(self._tools.values())
    
    def names(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __repr__(self) -> str:
        return f"ToolRegistry({len(self._tools)} tools: {', '.join(self.names())})"
