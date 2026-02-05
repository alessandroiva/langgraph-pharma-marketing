"""
Supervisor agent for orchestrating compliant marketing phrase generation.
Simplified architecture focused on a single use case.
"""

import logging
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..config import settings
from ..utils import retry_with_exponential_backoff, LLMException
from ..graph.state import AgentState, SupervisorDecision

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """
    Supervisor agent that routes requests to specialized agents.
    
    Demonstrates:
    - Multi-agent orchestration
    - Intelligent routing with LLM
    - Structured output parsing
    - Decision-making logic
    
    This is the core pattern for hierarchical agent systems.
    """
    
    AVAILABLE_AGENTS = {
        "drug_info": "Retrieves pharmaceutical information about drugs",
        "marketing_phrases": "Generates compliant marketing phrases for drugs",
        "safety_compliance": "Validates safety and regulatory compliance",
        "END": "Task is complete, no further processing needed"
    }
    
    def __init__(self):
        """Initialize supervisor agent."""
        self.llm = ChatOpenAI(
            model=settings.default_llm_model,
            temperature=0.3,
            api_key=settings.openai_api_key
        )
        logger.info("SupervisorAgent initialized")
    
    @retry_with_exponential_backoff(max_attempts=2)
    def route(self, state: AgentState) -> AgentState:
        """
        Route to appropriate specialized agent.
        
        Demonstrates supervisor routing in multi-agent systems.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with routing decision
        """
        user_input = state["user_input"]
        current_step = len(state.get("processing_steps", []))
        
        # Create routing prompt
        agents_desc = "\n".join(
            f"- {name}: {desc}" 
            for name, desc in self.AVAILABLE_AGENTS.items()
        )
        
        prompt = f"""You are a supervisor coordinating specialized pharmaceutical AI agents.

Available Agents:
{agents_desc}

Current Request: "{user_input}"

Processing Steps So Far: {state.get('processing_steps', [])}

Current State:
- Intent: {state.get('intent', 'unknown')}
- Has Drug Info: {state.get('drug_info') is not None}
- Compliance Status: {state.get('compliance_status', 'not checked')}
- Risk Level: {state.get('risk_level', 'not assessed')}

Determine which agent should handle this next, or if processing is complete (END).

Respond with ONLY the agent name (one of: {', '.join(self.AVAILABLE_AGENTS.keys())})"""

        try:
            # Get routing decision
            messages = [SystemMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            next_agent = response.content.strip().lower().replace(" ", "_")
            
            # Validate agent name
            if next_agent not in self.AVAILABLE_AGENTS:
                # Fallback: try to match partial
                for agent_name in self.AVAILABLE_AGENTS:
                    if agent_name in next_agent:
                        next_agent = agent_name
                        break
                else:
                    # Default to drug_info if unsure
                    next_agent = "drug_info"
            
            logger.info(f"Supervisor routing to: {next_agent}")
            
            # Update state
            state["next_agent"] = next_agent
            state["processing_steps"] = state.get("processing_steps", []) + [f"routed_to_{next_agent}"]
            
            return state
            
        except Exception as e:
            logger.error(f"Routing error: {str(e)}")
            state["error"] = f"Routing failed: {str(e)}"
            state["next_agent"] = "END"
            return state
    
    def should_continue(self, state: AgentState) -> str:
        """
        Determine if workflow should continue or end.
        
        This is used as a conditional edge function in LangGraph.
        
        Args:
            state: Current state
            
        Returns:
            Next agent name or "END"
        """
        next_agent = state.get("next_agent", "END")
        
        # Safety check: max 5 agents to prevent infinite loops
        if len(state.get("processing_steps", [])) > 5:
            logger.warning("Max processing steps reached, ending workflow")
            return "END"
        
        # If there's an error, end workflow
        if state.get("error"):
            return "END"
        
        return next_agent


def create_supervisor_node():
    """
    Factory function to create supervisor node for LangGraph.
    
    Returns:
        Supervisor routing function
    """
    supervisor = SupervisorAgent()
    
    def supervisor_node(state: AgentState) -> AgentState:
        """Supervisor node function for LangGraph."""
        return supervisor.route(state)
    
    return supervisor_node
