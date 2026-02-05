"""
State definitions for LangGraph workflows.
Defines the shared state structure passed between agents.
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal


class AgentState(TypedDict, total=False):
    """
    Shared state for multi-agent workflow.
    
    This state is passed between all agents and modified as the workflow progresses.
    Using total=False allows optional fields.
    """
    # Input
    user_input: str
    messages: List[Dict[str, str]]
    intent: str
    
    # Drug Information
    drug_info: Optional[Dict[str, Any]]
    retrieved_documents: Optional[List[Dict[str, Any]]]
    
    # Routing
    next_agent: str
    
    # Content Generation
    generated_content: str
    
    # Compliance Validation
    compliance_status: Optional[Literal["approved", "needs_review", "rejected"]]
    compliance_issues: Optional[List[str]]
    
    # Risk Assessment (legacy, kept for backwards compatibility)
    risk_level: Optional[Literal["low", "medium", "high"]]
    risk_factors: Optional[List[str]]
    
    # Output
    response: str
    processing_steps: List[str]
    error: Optional[str]


class SupervisorDecision(TypedDict):
    """
    Supervisor routing decision.
    """
    next_agent: str
    reasoning: str


class AgentMetrics(TypedDict):
    """
    Metrics for monitoring agent performance.
    """
    agent_name: str
    execution_time: float
    success: bool
    tokens_used: Optional[int]
