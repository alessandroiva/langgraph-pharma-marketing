"""
Specialized agents for compliant pharmaceutical marketing phrase generation.
Focused on a single use case: generating compliant marketing phrases for drugs.
"""

import logging
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..config import settings
from ..retrieval import HybridRetriever, load_pharmaceutical_data
from ..utils import retry_with_exponential_backoff, LLMException
from ..graph.state import AgentState

logger = logging.getLogger(__name__)


class DrugInformationAgent:
    """
    Specialized agent for pharmaceutical information retrieval.
    
    Demonstrates:
    - RAG integration
    - Structured information extraction
    - Domain-specific prompting
    """
    
    def __init__(self, retriever: HybridRetriever):
        """
        Initialize drug information agent.
        
        Args:
            retriever: Hybrid retriever for document search
        """
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model=settings.default_llm_model,
            temperature=0.3,
            api_key=settings.openai_api_key
        )
        logger.info("DrugInformationAgent initialized")
    
    @retry_with_exponential_backoff(max_attempts=2)
    def process(self, state: AgentState) -> AgentState:
        """
        Retrieve and synthesize drug information.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with drug information
        """
        user_input = state["user_input"]
        
        logger.info(f"DrugInformationAgent processing: {user_input[:50]}...")
        
        try:
            # Retrieve relevant documents
            results = self.retriever.hybrid_search(user_input, top_k=3)
            
            if not results:
                state["response"] = "I couldn't find information about that medication in our database."
                state["next_agent"] = "END"
                return state
            
            # Extract drug info from top result
            top_result = results[0]
            state["drug_info"] = top_result.metadata
            state["retrieved_documents"] = [
                {"document": r.document, "score": r.score}
                for r in results
            ]
            
            # Create context from retrieved documents
            context = "\n\n---\n\n".join([r.document for r in results])
            
            # Generate comprehensive response
            prompt = f"""You are a pharmaceutical information specialist. Provide accurate, comprehensive information based on the following drug database entries.

User Question: {user_input}

Relevant Drug Information:
{context}

Provide a clear, detailed response covering the relevant aspects. Be precise and cite specific information from the database.

Important: Always include appropriate warnings and remind users to consult healthcare professionals."""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            state["response"] = response.content
            state["processing_steps"] = state.get("processing_steps", []) + ["drug_info_retrieved"]
            
            # Route to marketing phrase generation if it's a marketing request
            if any(word in user_input.lower() for word in ["generate", "create", "marketing", "phrase", "campaign", "slogan"]):
                state["next_agent"] = "marketing_phrases"
            else:
                state["next_agent"] = "END"
            
            logger.info("DrugInformationAgent completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"DrugInformationAgent error: {str(e)}")
            state["error"] = f"Drug info retrieval failed: {str(e)}"
            state["next_agent"] = "END"
            return state


class SafetyComplianceAgent:
    """
    Agent for regulatory compliance and safety validation.
    
    Demonstrates:
    - Compliance checking
    - Risk detection
    - Regulatory validation
    - Business logic integration
    """
    
    PROHIBITED_CLAIMS = [
        "cure",
        "miracle",
        "guaranteed",
        "100% effective",
        "completely safe",
        "no side effects",
        "FDA approved for" # Without specific indication
    ]
    
    REQUIRED_DISCLAIMERS = [
        "consult",
        "healthcare professional",
        "doctor",
        "physician"
    ]
    
    def __init__(self):
        """Initialize safety compliance agent."""
        self.llm = ChatOpenAI(
            model=settings.default_llm_model,
            temperature=0.2,
            api_key=settings.openai_api_key
        )
        logger.info("SafetyComplianceAgent initialized")
    
    @retry_with_exponential_backoff(max_attempts=2)
    def process(self, state: AgentState) -> AgentState:
        """
        Validate safety and regulatory compliance.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with compliance status
        """
        content_to_check = state.get("generated_content") or state.get("response", "")
        
        logger.info("SafetyComplianceAgent checking compliance...")
        
        try:
            # Check for prohibited claims
            issues = []
            content_lower = content_to_check.lower()
            
            for claim in self.PROHIBITED_CLAIMS:
                if claim in content_lower:
                    issues.append(f"Prohibited claim detected: '{claim}'")
            
            # Check for required disclaimers
            has_disclaimer = any(
                disclaimer in content_lower 
                for disclaimer in self.REQUIRED_DISCLAIMERS
            )
            
            if not has_disclaimer:
                issues.append("Missing required healthcare professional consultation disclaimer")
            
            # LLM-based compliance check
            prompt = f"""You are a pharmaceutical regulatory compliance expert. Review the following content for FDA/EMA compliance issues.

Content to Review:
{content_to_check}

Check for:
1. Unsubstantiated claims
2. Missing safety warnings
3. Incorrect dosage information
4. Misleading statements
5. Required disclaimers

Provide a compliance assessment. If there are issues, list them clearly.
If compliant, respond with "COMPLIANT"."""

            response = self.llm.invoke([SystemMessage(content=prompt)])
            llm_assessment = response.content
            
            if "COMPLIANT" not in llm_assessment.upper():
                issues.append(f"LLM Compliance Check: {llm_assessment}")
            
            # Determine status
            if not issues:
                state["compliance_status"] = "approved"
                state["compliance_issues"] = []
                logger.info("Content approved by compliance check")
            elif len(issues) <= 2:
                state["compliance_status"] = "needs_review"
                state["compliance_issues"] = issues
                logger.warning(f"Content needs review: {len(issues)} issues")
            else:
                state["compliance_status"] = "rejected"
                state["compliance_issues"] = issues
                logger.warning(f"Content rejected: {len(issues)} issues")
            
            state["processing_steps"] = state.get("processing_steps", []) + ["compliance_checked"]
            state["next_agent"] = "END"
            
            return state
            
        except Exception as e:
            logger.error(f"SafetyComplianceAgent error: {str(e)}")
            state["error"] = f"Compliance check failed: {str(e)}"
            state["compliance_status"] = "needs_review"
            state["next_agent"] = "END"
            return state


class MarketingPhraseAgent:
    """
    Agent for generating compliant marketing phrases.
    
    Focused on creating targeted, compliant marketing phrases
    for pharmaceutical products based on drug info and target persona.
    """
    
    def __init__(self):
        """Initialize marketing phrase generator agent."""
        self.llm = ChatOpenAI(
            model=settings.default_llm_model,
            temperature=0.7,
            api_key=settings.openai_api_key
        )
        logger.info("MarketingPhraseAgent initialized")
    
    @retry_with_exponential_backoff(max_attempts=2)
    def process(self, state: AgentState) -> AgentState:
        """
        Generate compliant marketing phrases.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with generated marketing phrases
        """
        user_input = state["user_input"]
        drug_info = state.get("drug_info", {})
        
        logger.info("MarketingPhraseAgent generating phrases...")
        
        try:
            # Get drug context
            drug_context = ""
            if drug_info:
                drug_context = f"""
Drug: {drug_info.get('name', 'Unknown')}
Category: {drug_info.get('category', 'Unknown')}
Active Substance: {drug_info.get('active_substance', 'Unknown')}
Indications: {drug_info.get('indications', 'Unknown')}
Key Benefits: Focus on approved indications
"""
            
            # Generate marketing phrases
            prompt = f"""You are a pharmaceutical marketing expert. Generate the BEST 5 compliant marketing phrases/taglines.

Request: {user_input}

{drug_context}

Generate marketing phrases that:
1. Are catchy and memorable (5-10 words each)
2. Highlight key benefits relevant to the target audience
3. Are compliant with pharmaceutical advertising regulations
4. Are appropriate for the target demographic

Format your response EXACTLY as:
**Marketing Phrases for [Drug] - [Persona]:**

1. [Phrase 1]
2. [Phrase 2]
3. [Phrase 3]
4. [Phrase 4]
5. [Phrase 5]

**Why These Work:**

• [Bullet point explaining why these phrases are effective for this persona]

• [Bullet point about the key benefit highlighted]

• [Bullet point about the approach/tone used]

• [Bullet point about compliance/safety consideration]

IMPORTANT: Put each bullet point on its own line with blank lines between them for readability.

CRITICAL REQUIREMENTS:
- Generate EXACTLY 5 phrases (no more, no less)
- Avoid absolute claims like "cure", "miracle", "100% effective", or "completely safe"
- Focus ONLY on approved indications
- Ensure phrases are appropriate for the target demographic
- Follow pharmaceutical advertising regulations
- The bullet points should justify why these phrases work for this specific persona and drug

Generate creative, compliant marketing phrases:"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            state["generated_content"] = response.content
            state["response"] = response.content
            state["processing_steps"] = state.get("processing_steps", []) + ["phrases_generated"]
            
            # Route to compliance check
            state["next_agent"] = "safety_compliance"
            
            logger.info("MarketingPhraseAgent completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"MarketingPhraseAgent error: {str(e)}")
            state["error"] = f"Phrase generation failed: {str(e)}"
            state["next_agent"] = "END"
            return state


# Factory functions for LangGraph nodes
def create_agent_nodes(retriever: HybridRetriever) -> Dict[str, Any]:
    """
    Create specialist agent nodes for LangGraph.
    
    Focused on marketing phrase generation with compliance checking.
    
    Args:
        retriever: Hybrid retriever instance
        
    Returns:
        Dictionary of agent node functions
    """
    # Initialize agents
    drug_info_agent = DrugInformationAgent(retriever)
    safety_agent = SafetyComplianceAgent()
    marketing_agent = MarketingPhraseAgent()
    
    # Create node functions
    return {
        "drug_info": lambda state: drug_info_agent.process(state),
        "safety_compliance": lambda state: safety_agent.process(state),
        "marketing_phrases": lambda state: marketing_agent.process(state),
    }
