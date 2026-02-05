# 🚀 Quick Start Guide

Get started with the Compliant Pharma Marketing Generator in 5 minutes.

## Prerequisites

- Python 3.9+
- OpenAI API key

## Installation

### 1. Install Dependencies

```bash
pip install -e .
```

### 2. Set API Key

Create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

Or export it:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

## Usage

### Option 1: Streamlit Web App (Recommended)

```bash
streamlit run src/app.py
```

Open http://localhost:8501 in your browser.

#### Using the Web App

**For Marketing Phrases (Default):**
1. **Type just the drug name** in the chat input:
   - "Metformin"
   - "Lipitor"
   - "Advil"

2. **Choose target persona** from the options:
   - Children (3-12 years)
   - Teenagers (13-17 years)
   - Young Adults (18-35 years)
   - Middle-aged Adults (36-55 years)
   - Elderly (55+ years)
   - Healthcare Professionals

3. **Receive marketing phrases** - Get the best 5 compliant phrases with justification

**For Drug Information:**
- Ask explicitly: "What is Doliprane?"
- Or: "Tell me about Metformin", "Information about Lipitor"

### Option 2: Demo Script

```bash
python demo.py
```

Runs 3 demonstrations:
1. Marketing phrase generation for Middle-aged Adults
2. Healthcare Professional campaign
3. General conversation capability

### Option 3: Python API

```python
from src.graph import create_workflow

# Create workflow instance
workflow = create_workflow()

# Generate marketing phrases
query = "Create marketing phrases for Metformin targeting Middle-aged Adults"
result = workflow.run(query)

# Print results
print(result['response'])
print(f"Compliance: {result['compliance_status']}")
```

## Example Queries

### Marketing Phrase Generation (Just Type Drug Name!)

```
✅ "Metformin"
✅ "Lipitor"
✅ "Advil"
✅ "Doliprane"
```

Or be explicit:
```
✅ "Create marketing phrases for Lipitor"
✅ "Generate campaign for Metformin"
✅ "Marketing for Advil"
```

### Drug Information (Ask Explicitly)

```
✅ "What is Ventoline?"
✅ "Tell me about Aspirin"
✅ "Information about Metformin"
✅ "What are the side effects of Zoloft?"
✅ "How does Omeprazole work?"
```

## Available Drugs

1. **Doliprane** (Paracetamol) - Pain relief / Fever
2. **Ventoline** (Salbutamol) - Asthma
3. **Advil** (Ibuprofen) - Anti-inflammatory
4. **Lipitor** (Atorvastatin) - Cholesterol
5. **Metformin** - Diabetes
6. **Aspirin** - Pain relief / Cardiovascular
7. **Synthroid** (Levothyroxine) - Thyroid
8. **Amoxicillin** - Antibiotic
9. **Zoloft** (Sertraline) - Antidepressant
10. **Omeprazole** - Acid reflux

## Understanding the Output

### Marketing Phrases Example

```
Marketing Phrases for Metformin - Middle-aged Adults (36-55 years):

1. "Take Control of Your Blood Sugar Naturally"
2. "The First-Line Choice for Type 2 Diabetes"
3. "Supporting Your Health Journey, One Day at a Time"
4. "Proven Diabetes Management for Active Adults"
5. "Trusted by Doctors, Preferred by Patients"

Why These Work:
• Empowers middle-aged adults with action-oriented language ("Take Control", "Supporting")
• Emphasizes clinical credibility ("First-Line Choice", "Proven", "Trusted by Doctors")
• Uses lifestyle-focused messaging that resonates with active, health-conscious adults
• Maintains pharmaceutical compliance by avoiding absolute claims and focusing on approved indications
```

### Compliance Validation

The system automatically checks for:
- ✅ Healthcare professional disclaimer present
- ✅ No prohibited claims ("cure", "miracle", etc.)
- ✅ Age-appropriate messaging
- ✅ Required safety warnings

## Safety Features

### Age-Based Validation

The system will **automatically block** campaigns when:
- Drug is contraindicated for children
- No pediatric dosing information available
- Drug not approved for specific age group

Example:
```
❌ Creating campaign for Children (3-12 years) with Aspirin
→ BLOCKED: Risk of Reye's syndrome in children with viral infections
```

### Compliance Status

- **✅ Approved** - All compliance checks passed
- **⚠️ Needs Review** - Minor issues found (1-2)
- **❌ Rejected** - Major compliance issues (3+)

## Troubleshooting

### Error: "OPENAI_API_KEY not found"

**Solution**: Set your API key in `.env` file or environment variable

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

### Error: "Module not found"

**Solution**: Install dependencies

```bash
pip install -e .
```

### Streamlit shows blank page

**Solution**: Check terminal for errors, ensure port 8501 is free

```bash
# Try a different port
streamlit run src/app.py --server.port 8502
```

### ChromaDB errors

**Solution**: Delete and recreate vector store

```bash
rm -rf data/chroma_db
python demo.py  # Will recreate on first run
```

## Configuration

### Change LLM Model

Edit `src/config/settings.py`:

```python
default_llm_model: str = "gpt-4o"  # or "gpt-4o-mini"
```

### Adjust Temperature

```python
llm_temperature: float = 0.7  # 0.0 = deterministic, 2.0 = creative
```

### Change Number of Phrases

In `src/agents/specialist_agents.py`, modify the prompt in `MarketingPhraseAgent`:

```python
Generate 5-7 marketing phrases  # Change to desired number
```

## Next Steps

1. ✅ Run the demo: `python demo.py`
2. ✅ Try the web app: `streamlit run src/app.py`
3. ✅ Experiment with different drugs and personas
4. ✅ Review the generated phrases and compliance status
5. ✅ Explore the code in `src/` directory

## Support

For issues or questions:
- Check `README.md` for architecture details
- Review code comments in `src/` directory
- Examine `demo.py` for usage examples

## Tips

💡 **Start with general questions** to understand the system
💡 **Use Healthcare Professionals persona** for technical campaigns
💡 **Check compliance status** before using generated phrases
💡 **Clear chat** between different drug campaigns for fresh context
💡 **Review drug database** (`data/drug_notices.json`) to understand available information
