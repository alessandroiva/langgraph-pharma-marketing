"""
Demonstration script for Compliant Pharma Marketing Generator.
Focused on generating compliant marketing phrases for pharmaceutical products.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.graph import create_workflow
from src.monitoring.logging_config import setup_logging


def demo_marketing_phrase_generation():
    """Demonstrate compliant marketing phrase generation."""
    print("\n" + "="*80)
    print("🎯 DEMO 1: Compliant Marketing Phrase Generation")
    print("="*80)
    
    workflow = create_workflow()
    
    query = "Create marketing phrases for Metformin targeting Middle-aged Adults (36-55 years)"
    print(f"\n📝 Query: {query}\n")
    
    result = workflow.run(query)
    
    print(f"💬 Generated Phrases:\n{result['response']}\n")
    print(f"📊 Processing Steps: {' → '.join(result['processing_steps'])}")
    print(f"✅ Compliance Status: {result.get('compliance_status', 'N/A')}")
    
    if result.get('compliance_issues'):
        print(f"⚠️  Compliance Issues:")
        for issue in result['compliance_issues']:
            print(f"   - {issue}")


def demo_healthcare_professional_campaign():
    """Demonstrate marketing phrase generation for healthcare professionals."""
    print("\n" + "="*80)
    print("👨‍⚕️ DEMO 2: Healthcare Professional Campaign")
    print("="*80)
    
    workflow = create_workflow()
    
    query = "Create marketing phrases for Lipitor targeting Healthcare Professionals"
    print(f"\n📝 Query: {query}\n")
    
    result = workflow.run(query)
    
    print(f"💬 Generated Phrases:\n{result['response']}\n")
    print(f"📊 Processing Steps: {' → '.join(result['processing_steps'])}")
    print(f"✅ Compliance Status: {result.get('compliance_status', 'N/A')}")


def demo_general_conversation():
    """Demonstrate general conversation capability."""
    print("\n" + "="*80)
    print("💬 DEMO 3: General Conversation")
    print("="*80)
    
    workflow = create_workflow()
    
    query = "What information do you have about Ventoline?"
    print(f"\n📝 Query: {query}\n")
    
    result = workflow.run(query)
    
    print(f"💬 Response:\n{result['response']}\n")
    print(f"📊 Processing Steps: {' → '.join(result['processing_steps'])}")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "💊 Compliant Pharma Marketing Generator" + " " *23 + "║")
    print("║" + " "*10 + "Focused GenAI Solution with LangGraph Orchestration" + " "*17 + "║")
    print("╚" + "="*78 + "╝")
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    try:
        # Run demos
        demo_marketing_phrase_generation()
        demo_healthcare_professional_campaign()
        demo_general_conversation()
        
        print("\n" + "="*80)
        print("✨ Demo Complete!")
        print("="*80)
        print("\n📚 What was demonstrated:")
        print("  1. Compliant marketing phrase generation")
        print("  2. LangGraph multi-agent orchestration")
        print("  3. RAG-based drug information retrieval")
        print("  4. Automated safety & compliance validation")
        print("  5. Persona-targeted marketing content")
        print("  6. General conversation capability with GPT")
        print("\n💡 Key Features:")
        print("  - Single focused use case: Marketing phrase generation")
        print("  - Compliance-first approach")
        print("  - LangChain/LangGraph from data sources")
        print("  - GPT API for general conversations")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("  1. Make sure OPENAI_API_KEY is set in .env file")
        print("  2. Install dependencies: pip install -e .")
        print("  3. Check that data/drug_notices.json exists")
        print("\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
