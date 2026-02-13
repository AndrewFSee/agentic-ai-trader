"""
Structured Trading Decision Schema
===================================

Defines the JSON schema for structured LLM output from the Reflexion agent.
Used with OpenAI's response_format parameter to get typed, parseable decisions
instead of free-text that requires fragile regex parsing.
"""

from typing import Dict, Any

# The verdict enum values
VERDICT_ATTRACTIVE = "ATTRACTIVE"
VERDICT_NOT_ATTRACTIVE = "NOT_ATTRACTIVE"
VERDICT_UNCLEAR = "UNCLEAR"


# JSON Schema for OpenAI structured output (response_format)
TRADING_DECISION_SCHEMA: Dict[str, Any] = {
    "name": "trading_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["ATTRACTIVE", "NOT_ATTRACTIVE", "UNCLEAR"],
                "description": "The overall trade verdict"
            },
            "confidence": {
                "type": "integer",
                "description": "Conviction score from 1 (very low) to 10 (extremely high)"
            },
            "summary": {
                "type": "string",
                "description": "One paragraph summary of the decision and key reasoning"
            },
            "vix_roc_assessment": {
                "type": "object",
                "properties": {
                    "signal": {
                        "type": "string",
                        "enum": ["FAVORABLE", "UNFAVORABLE", "NEUTRAL"],
                        "description": "VIX ROC signal interpretation"
                    },
                    "detail": {
                        "type": "string",
                        "description": "Explanation of market timing assessment"
                    }
                },
                "required": ["signal", "detail"],
                "additionalProperties": False
            },
            "position_sizing": {
                "type": "object",
                "properties": {
                    "recommended_pct": {
                        "type": "number",
                        "description": "Recommended position size as % of account (0-25)"
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why this position size"
                    }
                },
                "required": ["recommended_pct", "rationale"],
                "additionalProperties": False
            },
            "technical_summary": {
                "type": "string",
                "description": "Key technical analysis points: trend, momentum, support/resistance"
            },
            "risk_management": {
                "type": "object",
                "properties": {
                    "entry_price": {
                        "type": ["number", "null"],
                        "description": "Suggested entry price, or null if not attractive"
                    },
                    "stop_loss": {
                        "type": ["number", "null"],
                        "description": "Stop-loss price, or null if not attractive"
                    },
                    "stop_loss_pct": {
                        "type": ["number", "null"],
                        "description": "Stop-loss distance as % below entry (e.g. 5.0 means 5% below)"
                    },
                    "target_price": {
                        "type": ["number", "null"],
                        "description": "Profit target price, or null if not attractive"
                    },
                    "risk_reward_ratio": {
                        "type": ["number", "null"],
                        "description": "Reward-to-risk ratio (e.g. 2.5 means 2.5:1)"
                    },
                    "max_loss_pct": {
                        "type": ["number", "null"],
                        "description": "Maximum acceptable loss as % of account"
                    }
                },
                "required": [
                    "entry_price", "stop_loss", "stop_loss_pct",
                    "target_price", "risk_reward_ratio", "max_loss_pct"
                ],
                "additionalProperties": False
            },
            "acknowledged_risks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of key risks and uncertainties to monitor"
            },
            "checklist": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Pre-trade checklist items to confirm before entry"
            }
        },
        "required": [
            "verdict", "confidence", "summary",
            "vix_roc_assessment", "position_sizing",
            "technical_summary", "risk_management",
            "acknowledged_risks", "checklist"
        ],
        "additionalProperties": False
    }
}


def parse_structured_decision(decision_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse and validate a structured trading decision.
    
    Returns the decision dict with normalized fields.
    Raises ValueError if the decision is malformed.
    """
    verdict = decision_json.get("verdict")
    if verdict not in (VERDICT_ATTRACTIVE, VERDICT_NOT_ATTRACTIVE, VERDICT_UNCLEAR):
        raise ValueError(f"Invalid verdict: {verdict}")
    
    confidence = decision_json.get("confidence", 5)
    if not isinstance(confidence, (int, float)):
        confidence = 5
    confidence = max(1, min(10, int(confidence)))
    
    return {
        "verdict": verdict,
        "confidence": confidence,
        "summary": decision_json.get("summary", ""),
        "vix_roc_assessment": decision_json.get("vix_roc_assessment", {}),
        "position_sizing": decision_json.get("position_sizing", {}),
        "technical_summary": decision_json.get("technical_summary", ""),
        "risk_management": decision_json.get("risk_management", {}),
        "acknowledged_risks": decision_json.get("acknowledged_risks", []),
        "checklist": decision_json.get("checklist", []),
    }
