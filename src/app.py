import streamlit as st
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

# Page Config - MUST be first Streamlit command
st.set_page_config(
    page_title="Pharma Marketing Generator",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # Try to load from .env file
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    OPENAI_API_KEY = line.split("=", 1)[1].strip()
                    break

# Custom CSS
st.markdown("""
<style>
    .reportview-container {background: #f0f2f6;}
    .stChatMessage {border-radius: 10px; padding: 10px;}
</style>
""", unsafe_allow_html=True)


@dataclass
class SearchResult:
    document: str
    metadata: Dict[str, Any]
    score: float


class SimpleRetriever:
    """Fast keyword search."""
    
    def __init__(self, documents: List[Dict]):
        self.documents = documents
    
    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        query_lower = query.lower()
        results = []
        
        for doc in self.documents:
            doc_text = json.dumps(doc).lower()
            score = sum(1 for word in query_lower.split() if word in doc_text)
            
            if score > 0:
                doc_str = f"**{doc.get('name', 'Unknown')}**\n"
                doc_str += f"Category: {doc.get('category', 'Unknown')}\n"
                doc_str += f"Active Substance: {doc.get('active_substance', 'Unknown')}\n"
                doc_str += f"Indications: {doc.get('indications', 'N/A')}\n"
                doc_str += f"Dosage: {doc.get('dosage', 'N/A')}\n"
                doc_str += f"Side Effects: {doc.get('side_effects', 'N/A')}"
                
                results.append(SearchResult(
                    document=doc_str,
                    metadata=doc,
                    score=score
                ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def get_drug_by_name(self, drug_name: str) -> Dict:
        """Get a specific drug by name."""
        for doc in self.documents:
            if doc.get('name', '').lower() == drug_name.lower():
                return doc
        return None


def validate_drug_for_persona(drug_info: Dict, persona: str) -> tuple[bool, str, str]:
    """
    Validate if a drug is appropriate for marketing to a specific persona.
    Returns (is_valid, rejection_message, safety_context)
    - is_valid: bool - whether campaign can proceed
    - rejection_message: str - user-facing rejection message (if invalid)
    - safety_context: str - safety info to include in LLM prompt (if valid)
    """
    if not drug_info:
        return False, "Drug information not found", ""
    
    drug_name = drug_info.get('name', 'Unknown')
    indications = drug_info.get('indications', '').lower()
    contraindications = drug_info.get('contraindications', '').lower()
    dosage = drug_info.get('dosage', '').lower()
    warnings = drug_info.get('warnings', '').lower()
    
    # Age-based validation
    if "Children" in persona:
        # Check if drug is explicitly contraindicated for children
        child_contraindications = ['children under', 'not for children', 'pediatric contraindication']
        if any(contra in contraindications for contra in child_contraindications):
            return False, f"**⚠️ Cannot create campaign for children**\n\n{drug_name} has contraindications for pediatric use. Creating marketing materials for children would be dangerous and irresponsible.\n\n**Contraindications:** {drug_info.get('contraindications', 'N/A')}", ""
        
        # Check for explicit age restrictions
        if "children under 16" in contraindications or "children under 16" in warnings:
            return False, f"**⚠️ Cannot create campaign for children**\n\n{drug_name} is not indicated for children under 16 years old due to safety concerns (e.g., Reye's syndrome risk).\n\n**Warning:** {drug_info.get('warnings', 'N/A')}", ""
        
        # Check if there's pediatric dosing information
        if 'children' not in dosage and 'pediatric' not in indications:
            return False, f"**⚠️ Cannot create campaign for children**\n\n{drug_name} does not have established pediatric indications or dosing guidelines. Marketing to children without proper pediatric approval would be inappropriate.\n\n**Current indications:** {drug_info.get('indications', 'N/A')}", ""
    
    elif "Teenager" in persona:
        # Similar check for teenagers
        if "adolescent" not in dosage and "over 12" not in dosage and "children" not in dosage:
            if "adults and adolescents over 12" not in dosage:
                return False, f"**⚠️ Cannot create campaign for teenagers**\n\n{drug_name} does not have established dosing for adolescents. Marketing to teenagers without proper age-specific guidance would be inappropriate.\n\n**Current dosage info:** {drug_info.get('dosage', 'N/A')}", ""
    
    elif "Elderly" in persona:
        # Check for elderly-specific warnings
        if "not recommended in elderly" in contraindications or "contraindicated in elderly" in contraindications:
            return False, f"**⚠️ Cannot create campaign for elderly patients**\n\n{drug_name} has specific contraindications for elderly patients.\n\n**Contraindications:** {drug_info.get('contraindications', 'N/A')}", ""
    
    # Build safety context for approved campaigns
    safety_context = ""
    
    # Pregnancy-related validation for Young/Middle-aged adults
    if "Young Adults" in persona or "Middle-aged Adults" in persona:
        pregnancy_cat = drug_info.get('pregnancy_category', '').upper()
        if 'X' in pregnancy_cat or 'CONTRAINDICATED IN PREGNANCY' in pregnancy_cat:
            safety_context = f"IMPORTANT: This drug is pregnancy category {pregnancy_cat} (contraindicated in pregnancy). The campaign MUST include prominent pregnancy warnings."
    
    # Healthcare Professionals - mention if prescription required
    if "Healthcare Professionals" in persona:
        regulatory = drug_info.get('regulatory_status', '')
        if 'prescription required' in regulatory.lower():
            safety_context = "This is a prescription medication. Campaign should emphasize clinical indications, efficacy, and safety profile for prescribing decisions."
    
    # If all checks pass
    return True, "", safety_context


@st.cache_resource(show_spinner="⚡ Loading system...")
def load_system():
    """Load data and LLM."""
    # Load drug data
    data_path = Path(__file__).parent.parent / "data" / "drug_notices.json"
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    documents = list(data.get("drugs", {}).values())
    retriever = SimpleRetriever(documents)
    
    # Initialize LLM - import only when needed!
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=OPENAI_API_KEY
    )
    
    return retriever, llm


# Load system
try:
    if not OPENAI_API_KEY:
        st.error("❌ OPENAI_API_KEY not found in environment or .env file!")
        st.stop()
    
    retriever, llm = load_system()
    system_ready = True
except Exception as e:
    st.error(f"❌ Failed to initialize: {str(e)}")
    st.stop()


# Session state
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": """Hello! I'm your Pharma Marketing Generator. 

**Quick Start:**
- Type a drug name (e.g., "Metformin", "Lipitor") to create marketing phrases
- Ask for "information about [drug]" or "details on [drug]" to learn more about a medication

Ready to create compliant marketing campaigns!"""
    }]


# Sidebar
with st.sidebar:
    st.title("💊 Marketing Generator")
    st.markdown("---")
    
    st.subheader("🎯 How It Works")
    st.markdown("""
    **Marketing Campaign:**
    - Type a drug name (e.g., "Metformin")
    - Choose target persona
    - Get best 5 phrases + justification
    
    **Drug Information:**
    - Ask "What is [drug]?"
    - Or "Tell me about [drug]"
    - Get detailed information
    """)
    
    st.markdown("---")
    
    # Drug list with expander
    with st.expander(f"📚 Available Drugs ({len(retriever.documents)})", expanded=False):
        drug_names = [doc.get('name', 'Unknown') for doc in retriever.documents]
        for i, drug_name in enumerate(sorted(drug_names), 1):
            st.markdown(f"{i}. {drug_name}")
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Chat cleared! Ready to generate marketing phrases."
        }]
        if 'pending_options' in st.session_state:
            st.session_state['pending_options'] = None
        st.rerun()


# Main UI
st.title("Compliant Pharma Marketing Generator")
st.markdown("Generate compliant marketing phrases for pharmaceutical products 🎯")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Display option buttons if there are pending options
if 'pending_options' in st.session_state and st.session_state['pending_options'] is not None:
    options_data = st.session_state['pending_options']
    
    st.markdown("---")
    st.markdown("**👇 Select an option:**")
    
    # Display buttons in columns
    cols = st.columns(2)
    for i, option in enumerate(options_data['options'][:10]):  # Show up to 10 options
        col_idx = i % 2
        with cols[col_idx]:
            if st.button(f"🔹 {option}", key=f"opt_btn_{i}", use_container_width=True):
                # Determine follow-up based on selection type
                if options_data.get('type') == 'drug_selection':
                    # User selected a drug, now ask for persona
                    personas = ["Children (3-12 years)", "Teenagers (13-17 years)", "Young Adults (18-35 years)", 
                               "Middle-aged Adults (36-55 years)", "Elderly (55+ years)", "Healthcare Professionals"]
                    
                    st.session_state['pending_options'] = {
                        'query': f"Create marketing phrases for {option}",
                        'drugs': [option],
                        'options': personas,
                        'type': 'persona_selection'
                    }
                    
                    response_text = f"Great choice! Now, which target persona would you like to create marketing phrases for **{option}**?\n\nPlease select a persona below:"
                    st.session_state.messages.append({"role": "user", "content": option})
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.rerun()
                
                elif options_data.get('type') == 'persona_selection':
                    # User selected a persona, validate drug safety first
                    drugs = options_data.get('drugs', [])
                    if drugs:
                        drug_name = drugs[0]  # Get first drug
                        drug_info = retriever.get_drug_by_name(drug_name)
                        
                        # Validate if drug is appropriate for persona
                        is_valid, rejection_message, safety_context = validate_drug_for_persona(drug_info, option)
                        
                        if not is_valid:
                            # Drug is not appropriate, show rejection message
                            st.session_state.messages.append({"role": "user", "content": option})
                            st.session_state.messages.append({"role": "assistant", "content": rejection_message})
                            # Clear pending options
                            st.session_state['pending_options'] = None
                            st.rerun()
                        else:
                            # Drug is appropriate, proceed with campaign generation via LLM
                            # Build the query for the LLM (safety context is internal, not shown to user)
                            base_query = f"Create an engaging marketing campaign for {', '.join(drugs)} targeting {option}"
                            
                            # Store safety context separately for LLM prompt (not shown in UI)
                            st.session_state['safety_context'] = safety_context
                            
                            # Add only the persona name to user messages (not the full query)
                            st.session_state.messages.append({"role": "user", "content": option})
                            
                            # Set flag to process this query on next render
                            st.session_state['button_clicked'] = True
                            st.session_state['button_query'] = base_query
                            st.session_state['skip_user_display'] = True  # Skip showing query in chat
                            # Clear pending options
                            st.session_state['pending_options'] = None
                            st.rerun()
                    else:
                        # No drug specified, show error
                        st.session_state.messages.append({"role": "user", "content": option})
                        st.session_state.messages.append({"role": "assistant", "content": "Error: No drug specified for campaign."})
                        st.session_state['pending_options'] = None
                        st.rerun()
else:
    # No pending options - normal state
    pass


# Chat input (or button-triggered query)
prompt = st.chat_input("Type a drug name (e.g., 'Metformin') or ask for information...")
if not prompt and st.session_state.get('button_clicked', False):
    prompt = st.session_state.get('button_query', '')
    st.session_state['button_clicked'] = False
    st.session_state['button_query'] = None

if prompt:
    # Clear pending options when new query is submitted
    if 'pending_options' in st.session_state:
        st.session_state['pending_options'] = None
    
    # Check if we should skip displaying the user message (for button-triggered queries)
    skip_display = st.session_state.get('skip_user_display', False)
    if skip_display:
        # Don't show or add the full query - persona name already added
        st.session_state['skip_user_display'] = False
    else:
        # Normal flow - add and display the user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("💭 Thinking..."):
            try:
                # Check if this is a greeting or general question
                prompt_lower = prompt.lower().strip()
                greeting_keywords = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
                help_keywords = ['help', 'what can you do', 'how do you work', 'what are you', 'capabilities']
                marketing_keywords = ['campaign', 'marketing', 'design', 'create', 'generate', 'phrase', 'slogan']
                info_keywords = ['information', 'info', 'tell me about', 'what is', 'what are', 'details', 'explain', 'describe', 'side effects', 'dosage', 'contraindications', 'warnings', 'how does', 'mechanism']
                
                is_greeting = any(keyword in prompt_lower for keyword in greeting_keywords)
                is_help = any(keyword in prompt_lower for keyword in help_keywords)
                is_marketing = any(keyword in prompt_lower for keyword in marketing_keywords)
                is_info_request = any(keyword in prompt_lower for keyword in info_keywords)
                
                if is_greeting or is_help:
                    # Respond conversationally without drug search
                    from langchain_core.messages import HumanMessage
                    response = llm.invoke([HumanMessage(content=f"""You are a friendly pharmaceutical marketing assistant. The user said: "{prompt}"

Respond warmly and briefly explain that you specialize in generating compliant marketing phrases for pharmaceutical products. Mention that they can create marketing campaigns for any drug in the database by selecting a drug and target persona. You can also answer general questions. Keep your response concise and friendly (2-3 sentences max).""")])
                    
                    response_text = response.content
                    st.markdown(response_text)
                    
                elif is_marketing:
                    # Check if this is already a complete campaign request (from persona selection)
                    # Pattern: "Create an engaging marketing campaign for [DRUG] targeting [PERSONA]"
                    is_complete_campaign = (
                        'targeting' in prompt_lower and 
                        ('children' in prompt_lower or 'teenager' in prompt_lower or 
                         'young adult' in prompt_lower or 'middle-aged adult' in prompt_lower or 
                         'elderly' in prompt_lower or 'healthcare professional' in prompt_lower)
                    )
                    
                    if is_complete_campaign:
                        # This is a complete campaign request, generate it with LLM
                        results = retriever.search(prompt, top_k=1)
                        
                        if results:
                            context = results[0].document
                            
                            # Get safety context if stored (from persona validation)
                            safety_context = st.session_state.get('safety_context', '')
                            
                            # Build the LLM prompt with safety context (internal only)
                            safety_instruction = ""
                            if safety_context:
                                safety_instruction = f"\n\nIMPORTANT SAFETY CONTEXT: {safety_context}\nEnsure the marketing phrases and justification reflect this safety consideration."
                            
                            from langchain_core.messages import HumanMessage
                            response = llm.invoke([HumanMessage(content=f"""You are a pharmaceutical marketing expert. Create compliant marketing phrases based on this request:

"{prompt}"

DRUG INFORMATION:
{context}{safety_instruction}

Generate the BEST 5 marketing phrases/taglines that:
1. Are catchy and memorable
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

IMPORTANT: 
- Generate EXACTLY 5 phrases (no more, no less)
- Ensure all phrases are appropriate for the target demographic
- Follow pharmaceutical advertising regulations
- Avoid absolute claims like "cure" or "completely safe"
- Focus on approved indications only
- The bullet points should justify why these phrases work for this specific persona and drug

Provide creative, compliant marketing phrases that would resonate with the target audience.""")])
                            
                            # Clear the safety context after use
                            if 'safety_context' in st.session_state:
                                del st.session_state['safety_context']
                            
                            response_text = response.content
                            st.markdown(response_text)
                        else:
                            response_text = "I couldn't find drug information to create the campaign. Please try again."
                            st.markdown(response_text)
                    
                    else:
                        # Check if user used generic terms like "specific drugs", "medications", etc.
                        prompt_lower = prompt.lower()
                        generic_drug_terms = ['specific drug', 'specific medication', 'drugs', 'medications', 'medicine']
                        
                        # If query contains generic terms, ask for drug selection
                        has_generic_terms = any(term in prompt_lower for term in generic_drug_terms)
                        
                        if has_generic_terms:
                            # User used generic terms, ask them to select a drug
                            all_drugs = sorted([doc.get('name', 'Unknown') for doc in retriever.documents])
                            
                            # Store for drug selection buttons
                            st.session_state['pending_options'] = {
                                'query': prompt,
                                'drugs': None,  # No drugs selected yet
                                'options': all_drugs[:10],  # Show up to 10 drugs
                                'type': 'drug_selection'
                            }
                            
                            response_text = "I can help you with that! First, which medication would you like to focus on?\n\nPlease select a drug from the options below:"
                        
                        else:
                            # No generic terms, search for specific drug names in query
                            results = retriever.search(prompt, top_k=1)  # Only get the best match
                            
                            if results and results[0].score > 0:
                                # Specific drug(s) were mentioned, go directly to persona selection
                                drug_names = [r.metadata.get('name', 'Unknown') for r in results]
                                personas = ["Children (3-12 years)", "Teenagers (13-17 years)", "Young Adults (18-35 years)", 
                                           "Middle-aged Adults (36-55 years)", "Elderly (55+ years)", "Healthcare Professionals"]
                                
                                st.session_state['pending_options'] = {
                                    'query': f"Create marketing phrases for {', '.join(drug_names)}",
                                    'drugs': drug_names,
                                    'options': personas,
                                    'type': 'persona_selection'
                                }
                                
                                response_text = f"Great! I'll help you create marketing phrases for **{', '.join(drug_names)}**. Which target persona would you like to focus on?\n\nPlease select a persona below:"
                            else:
                                # No drugs found, ask for selection
                                all_drugs = sorted([doc.get('name', 'Unknown') for doc in retriever.documents])
                                
                                st.session_state['pending_options'] = {
                                    'query': prompt,
                                    'drugs': None,
                                    'options': all_drugs[:10],
                                    'type': 'drug_selection'
                                }
                                
                                response_text = "I can help you with that! First, which medication would you like to create marketing phrases for?\n\nPlease select a drug from the options below:"
                        
                        # Display the response
                        st.markdown(response_text)
                
                elif is_info_request:
                    # User explicitly asked for information - provide detailed drug info
                    results = retriever.search(prompt, top_k=2)
                    
                    if results:
                        context = "\n\n".join([r.document for r in results])
                        
                        # Generate response
                        from langchain_core.messages import HumanMessage
                        response = llm.invoke([HumanMessage(content=f"""You are a pharmaceutical expert. Answer this question based on the provided drug information.

DRUG INFORMATION:
{context}

QUESTION: {prompt}

Provide a clear, accurate answer. Always remind users to consult healthcare professionals for medical advice.""")])
                        
                        response_text = response.content
                        
                        # Show response
                        st.markdown(response_text)
                        
                        # Show sources
                        with st.expander(f"📚 {len(results)} source(s)"):
                            for i, r in enumerate(results, 1):
                                st.markdown(f"**{i}.** {r.metadata.get('name', 'Unknown')}")
                    else:
                        response_text = "I couldn't find information about that medication in our database. Please try asking about a different drug or check the drug list in the sidebar."
                        st.markdown(response_text)
                
                else:
                    # Default: Assume user wants to create a marketing campaign
                    # Search for drug in the query
                    results = retriever.search(prompt, top_k=1)
                    
                    if results and results[0].score > 0:
                        # Found a drug - go directly to persona selection for marketing
                        drug_names = [r.metadata.get('name', 'Unknown') for r in results]
                        personas = ["Children (3-12 years)", "Teenagers (13-17 years)", "Young Adults (18-35 years)", 
                                   "Middle-aged Adults (36-55 years)", "Elderly (55+ years)", "Healthcare Professionals"]
                        
                        st.session_state['pending_options'] = {
                            'query': f"Create marketing phrases for {', '.join(drug_names)}",
                            'drugs': drug_names,
                            'options': personas,
                            'type': 'persona_selection'
                        }
                        
                        response_text = f"Perfect! I'll create marketing phrases for **{', '.join(drug_names)}**. Which target persona would you like to focus on?\n\nPlease select a persona below:"
                    else:
                        # No drug found - ask them to select one
                        all_drugs = sorted([doc.get('name', 'Unknown') for doc in retriever.documents])
                        
                        st.session_state['pending_options'] = {
                            'query': prompt,
                            'drugs': None,
                            'options': all_drugs[:10],
                            'type': 'drug_selection'
                        }
                        
                        response_text = "I'll help you create marketing phrases! Which medication would you like to focus on?\n\nPlease select a drug from the options below:"
                    
                    st.markdown(response_text)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text
                })
                
                # If we just set pending options, rerun to show buttons
                if 'pending_options' in st.session_state and st.session_state['pending_options'] is not None:
                    st.rerun()

                
            except Exception as e:
                error = f"❌ Error: {str(e)}"
                st.error(error)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error
                })
