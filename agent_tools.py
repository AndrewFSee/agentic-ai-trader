# agent_tools.py
from dotenv import load_dotenv
load_dotenv()

import time
from typing import List, Dict, Any
from polygon_tools import get_all_tools, get_tool_function
from sentiment_tools import get_all_tools as get_sentiment_tools, get_tool_function as get_sentiment_tool_function
from research_tools import get_all_tools as get_research_tools, get_tool_function as get_research_tool_function
from tools import TOOL_REGISTRY as regime_tools_registry
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Smart rate limiter that tracks API call timestamps and enforces limits.
    Only sleeps when necessary to respect rate limits.
    """
    def __init__(self, calls_per_minute: int = 5):
        self.calls = []  # Timestamps of recent calls
        self.limit = calls_per_minute
        self.window = 60  # seconds
    
    def wait_if_needed(self, api_name: str = "API") -> None:
        """
        Wait if necessary to respect rate limit.
        Removes old calls from tracking and sleeps if at limit.
        """
        now = time.time()
        
        # Remove calls older than the time window
        self.calls = [t for t in self.calls if now - t < self.window]
        
        # If at limit, wait until oldest call expires
        if len(self.calls) >= self.limit:
            sleep_time = self.window - (now - self.calls[0]) + 0.5  # Add 0.5s buffer
            # Don't log for paid subscriptions (limit >= 100)
            if self.limit < 100:
                logger.info("%s rate limit: waiting %.1fs (at %s calls/min)...", api_name, sleep_time, self.limit)
            time.sleep(sleep_time)
            
            # Clean up again after sleeping
            now = time.time()
            self.calls = [t for t in self.calls if now - t < self.window]
        
        # Record this call
        self.calls.append(time.time())
    
    def reset(self) -> None:
        """Clear call history."""
        self.calls = []


# Global rate limiter for Polygon.io
# Set to 1000 calls/min for paid tier to avoid unnecessary delays
# (Actual limits are higher: 1000/min on Advanced, 10k+/min on higher tiers)
_polygon_limiter = RateLimiter(calls_per_minute=1000)


def run_tools(
    state: Dict[str, Any], planned_tool_calls: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Execute a list of planned tool calls.

    Updates and returns the state dict.
    
    Uses smart rate limiting that only delays when necessary to respect
    the Polygon.io Massive tier limit of 5 calls per minute (100k calls/month).
    """
    if "tool_results" not in state:
        state["tool_results"] = {}

    # Proactively fetch macro indicators (FRED) so the decision LLM always
    # has a macro backdrop when available. This is done before planned tools
    # to avoid surprising order-dependencies in prompts.
    try:
        fred_fn = get_research_tool_function("fred_macro_indicators")
        if fred_fn is not None:
            # Only fetch if not already present in state
            if "fred_macro_indicators" not in state["tool_results"]:
                fred_fn(state, {})
    except Exception as e:
        # Non-fatal; store a warning in state for downstream visibility
        state.setdefault("tool_results", {}).setdefault("_warnings", []).append(
            f"fred_macro_indicators prefetch failed: {e}"
        )

    # Also prefetch quick fundamentals when a symbol is provided in planned_tool_calls
    try:
        # Look for a symbol argument in planned tool calls
        symbol = None
        for c in planned_tool_calls:
            args = c.get("arguments", {})
            if args and isinstance(args, dict):
                if args.get("symbol"):
                    symbol = args.get("symbol")
                    break

        if symbol and "fetch_fundamentals" not in state["tool_results"]:
            fund_fn = get_research_tool_function("fetch_fundamentals")
            if fund_fn is not None:
                fund_fn(state, {"symbol": symbol})
    except Exception as e:
        state.setdefault("tool_results", {}).setdefault("_warnings", []).append(
            f"fetch_fundamentals prefetch failed: {e}"
        )

    for i, call in enumerate(planned_tool_calls):
        name = call["tool_name"]
        args = call.get("arguments", {})
        
        # Try to get tool from polygon_tools, sentiment_tools, research_tools, or regime tools
        tool_fn = get_tool_function(name)
        if tool_fn is None:
            tool_fn = get_sentiment_tool_function(name)
        if tool_fn is None:
            tool_fn = get_research_tool_function(name)
        if tool_fn is None:
            # Check regime tools registry
            regime_tool = regime_tools_registry.get(name)
            if regime_tool:
                tool_fn = regime_tool["fn"]
        
        if tool_fn is None:
            logger.warning("Unknown tool: %s, skipping.", name)
            continue
        
        # Smart rate limiting for Polygon.io calls
        if name.startswith("polygon_"):
            _polygon_limiter.wait_if_needed("Polygon.io")
        
        # Research tools take longer (2-5 minutes), show progress
        if name.startswith("gpt_researcher_"):
            logger.info("Running deep research: %s (this may take 2-5 minutes)...", name)
        
        # Execute the tool
        state = tool_fn(state, args)

    return state
