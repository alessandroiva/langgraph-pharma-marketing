# 💊 Compliant Pharma Marketing Generator

A focused GenAI solution for generating compliant marketing phrases for pharmaceutical products using LangChain/LangGraph.


The process:
1. **Select Drug** - Choose from 10 pharmaceutical products in the database
2. **Choose Persona** - Select target audience (children, teenagers, adults, elderly, healthcare professionals)
3. **Generate Phrases** - Get the best 5 compliant marketing phrases tailored to the persona
4. **Justification** - Receive bullet points explaining why these phrases work
5. **Validate Compliance** - Automatic safety and regulatory compliance checking
6. **General Chat** - GPT API for general conversations about drugs

## 🏗️ Architecture

### LangGraph Multi-Agent System

```
User Query
    ↓
Supervisor Agent (Routing)
    ↓
Drug Info Agent (RAG Retrieval)
    ↓
Marketing Phrase Agent (Generation)
    ↓
Safety Compliance Agent (Validation)
    ↓
Response to User
```

### Key Components

- **Streamlit UI** - Interactive web interface
- **LangGraph Workflow** - Multi-agent orchestration
- **LangChain RAG** - Retrieval from drug database
- **OpenAI GPT-4** - Phrase generation & general chat
- **ChromaDB** - Vector store for drug information
- **Compliance Engine** - Automatic safety validation



## 🚀 Usage

### Run Streamlit App

```bash
streamlit run src/app.py
```

Then open http://localhost:8501

### Run Demo Script

```bash
python demo.py
```

## 💡 How to Use the App

1. **Start the app** - Run `streamlit run src/app.py`
2. **Type a drug name** - Simply type the drug name (e.g., "Metformin", "Lipitor")
   - The system assumes you want to create a marketing campaign
3. **Select Persona** - Pick target audience from the options
4. **Get Phrases** - Receive the best 5 compliant marketing phrases
5. **Review Justification** - See why these phrases work for your target persona
6. **Check Compliance** - Automatic safety validation

**For Drug Information:**
- Ask explicitly: "What is Doliprane?" or "Tell me about Metformin"
- Or: "Information about Lipitor", "Details on Aspirin"

## 📊 Example Queries

### Marketing Phrase Generation (Just type the drug name!)
```
"Metformin"
"Lipitor"
"Advil"
"Doliprane"
```

Or be explicit:
```
"Create marketing phrases for Metformin"
"Generate campaign for Lipitor"
"Marketing for Advil"
```

### Drug Information (Ask explicitly)
```
"What is Doliprane?"
"Tell me about Ventoline"
"Information about Aspirin"
"What are the side effects of Metformin?"
"How does Omeprazole work?"
```

## 🛡️ Compliance Features

### Automatic Safety Validation

- ✅ Checks for prohibited claims ("cure", "miracle", "100% effective")
- ✅ Ensures required disclaimers present
- ✅ Validates age-appropriate messaging
- ✅ Confirms pregnancy warnings where needed
- ✅ Verifies regulatory compliance

### Age-Based Safety

- 🚫 Blocks campaigns for children if contraindicated
- 🚫 Prevents marketing without pediatric approval
- ⚠️ Includes pregnancy warnings for reproductive age groups
- ✅ Tailors safety messaging to persona

## 📁 Project Structure

```
 src/
│   ├── app.py                 # Streamlit UI (main app)
│   ├── agents/
│   │   ├── specialist_agents.py   # Drug Info, Marketing Phrase, Compliance Agents
│   │   └── supervisor.py          # Routing Agent
│   ├── graph/
│   │   └── workflows.py           # LangGraph Workflow
│   ├── retrieval/
│   │   ├── vector_store.py        # ChromaDB setup
│   │   └── retrieval_strategies.py # RAG logic
│   └── config/
│       └── settings.py            # Configuration
├── data/
│   └── drug_notices.json          # 10 pharmaceutical products
├── demo.py                        # Demo script
└── pyproject.toml                 # Dependencies
```

## 🗄️ Drug Database

10 pharmaceutical products included:

1. Doliprane (Paracetamol) - Pain relief
2. Ventoline (Salbutamol) - Asthma
3. Advil (Ibuprofen) - Anti-inflammatory
4. Lipitor (Atorvastatin) - Cholesterol
5. Metformin - Diabetes
6. Aspirin - Pain relief / cardiovascular
7. Synthroid (Levothyroxine) - Thyroid
8. Amoxicillin - Antibiotic
9. Zoloft (Sertraline) - Antidepressant
10. Omeprazole - Acid reflux

Each drug includes:
- Active substance
- Indications
- Dosage
- Contraindications
- Side effects
- Warnings
- Drug interactions
- Pregnancy category
- Regulatory status

## 🎨 Technology Stack

- **LangGraph** - Multi-agent orchestration
- **LangChain** - RAG pipeline
- **OpenAI GPT-4** - LLM for generation & chat
- **ChromaDB** - Vector database
- **Streamlit** - Web interface
- **Sentence Transformers** - Embeddings
- **Pydantic** - Configuration management

## 🔧 Configuration

Edit `src/config/settings.py` or `.env`:

```python
OPENAI_API_KEY=your_key
DEFAULT_LLM_MODEL=gpt-4o-mini  # or gpt-4o
LLM_TEMPERATURE=0.7
RETRIEVAL_TOP_K=3
```