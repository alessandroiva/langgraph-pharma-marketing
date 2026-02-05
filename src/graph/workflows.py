"""
LangGraph workflow for compliant pharmaceutical marketing phrase generation.
Focused on a single use case: generating compliant marketing phrases.
"""

import logging
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END

from .state import AgentState
from ..agents import create_supervisor_node,create_agent_nodes
from ..retrieval import HybridRetriever, VectorStore, load_pharmaceutical_data
from ..monitoring.logging_config import setup_logging

logger = logging.getLogger(__name__)


class PharmaMarketingWorkflow:
    """
    Workflow for generating compliant pharmaceutical marketing phrases.
    
    Demonstrates:
    - StateGraph with specialized agents
    - Conditional routing
    - Supervisor pattern
    - Compliance validation
    - Production workflow patterns
    
    Focused on one use case: compliant marketing phrase generation.
    """
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize workflow.
        
        Args:
            vector_store: Optional pre-initialized vector store
        """
        # Initialize retrieval system
        if vector_store is None:
            logger.info("Loading pharmaceutical data into vector store...")
            vector_store = load_pharmaceutical_data()
        
        self.vector_store = vector_store
        self.retriever = HybridRetriever(vector_store)
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info("PharmaMarketingWorkflow initialized")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Demonstrates complex graph construction with:
        - Multiple specialized nodes
        - Supervisor routing
        - Conditional edges
        - Proper error handling
        
        Returns:
            Compiled StateGraph
        """
        # Create workflow graph
        workflow = StateGraph(AgentState)
        
        # Create supervisor node
        supervisor_node = create_supervisor_node()
        
        # Create specialist agent nodes
        agent_nodes = create_agent_nodes(self.retriever)
        
        # Add all nodes to graph
        workflow.add_node("supervisor", supervisor_node)
        
        for agent_name, agent_func in agent_nodes.items():
            workflow.add_node(agent_name, agent_func)
        
        # Set entry point - start with supervisor
        workflow.set_entry_point("supervisor")
        
        # Add conditional edges from supervisor to agents
        # The supervisor decides which agent to route to
        workflow.add_conditional_edges(
            "supervisor",
            lambda state: state.get("next_agent", "END"),
            {
                "drug_info": "drug_info",
                "marketing_phrases": "marketing_phrases",
                "safety_compliance": "safety_compliance",
                "END": END
            }
        )
        
        # After each specialist agent, route to next agent
        for agent_name in agent_nodes.keys():
            workflow.add_conditional_edges(
                agent_name,
                lambda state: state.get("next_agent", "END"),
                {
                    "drug_info": "drug_info",
                    "marketing_phrases": "marketing_phrases",
                    "safety_compliance": "safety_compliance",
                    "supervisor": "supervisor",
                    "END": END
                }
            )
        
        # Compile the graph
        compiled_graph = workflow.compile()
        
        logger.info("LangGraph workflow compiled successfully")
        return compiled_graph
    
    def run(self, user_input: str) -> Dict[str, Any]:
        """
        Execute the workflow for a user query.
        
        Args:
            user_input: User's question or request
            
        Returns:
            Final state with response and metadata
        """
        logger.info(f"Running workflow for input: {user_input[:100]}...")
        
        # Create initial state
        initial_state: AgentState = {
            "user_input": user_input,
            "messages": [],
            "intent": "",
            "drug_info": None,
            "retrieved_documents": None,
            "next_agent": "",
            "generated_content": "",
            "compliance_status": None,
            "compliance_issues": None,
            "risk_level": None,
            "risk_factors": None,
            "response": "",
            "processing_steps": [],
            "error": None
        }
        
        try:
            # Execute workflow
            final_state = self.graph.invoke(initial_state)
            
            logger.info(f"Workflow completed. Steps: {final_state.get('processing_steps', [])}")
            
            return {
                "response": final_state.get("response", ""),
                "drug_info": final_state.get("drug_info"),
                "compliance_status": final_state.get("compliance_status"),
                "compliance_issues": final_state.get("compliance_issues", []),
                "risk_level": final_state.get("risk_level"),
                "risk_factors": final_state.get("risk_factors", []),
                "processing_steps": final_state.get("processing_steps", []),
                "error": final_state.get("error"),
                "success": final_state.get("error") is None
            }
            
        except Exception as e:
            logger.error(f"Workflow execution error: {str(e)}", exc_info=True)
            return {
                "response": f"An error occurred: {str(e)}",
                "error": str(e),
                "success": False
            }
    
    async def arun(self, user_input: str) -> Dict[str, Any]:
        """
        Async version of run for production APIs.
        
        Args:
            user_input: User's question or request
            
        Returns:
            Final state with response and metadata
        """
        # Note: LangGraph's invoke can be async
        # This is a placeholder for async implementation
        return self.run(user_input)
    
    def stream(self, user_input: str):
        """
        Stream workflow execution for real-time updates.
        
        Demonstrates streaming pattern for long-running workflows.
        
        Args:
            user_input: User input
            
        Yields:
            State updates as workflow progresses
        """
        initial_state: AgentState = {
            "user_input": user_input,
            "messages": [],
            "intent": "",
            "drug_info": None,
            "retrieved_documents": None,
            "next_agent": "",
            "generated_content": "",
            "compliance_status": None,
            "compliance_issues": None,
            "risk_level": None,
            "risk_factors": None,
            "response": "",
            "processing_steps": [],
            "error": None
        }
        
        try:
            # Stream updates
            for state in self.graph.stream(initial_state):
                yield state
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield {"error": str(e)}


# Convenience function for quick testing
def create_workflow(vector_store: Optional[VectorStore] = None) -> PharmaMarketingWorkflow:
    """
    Factory function to create workflow instance.
    
    Args:
        vector_store: Optional vector store instance
        
    Returns:
        Configured workflow
    """
    return PharmaMarketingWorkflow(vector_store)
