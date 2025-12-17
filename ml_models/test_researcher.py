"""
Test GPT-Researcher with a simple query to diagnose issues.
"""
import asyncio
import os
from datetime import datetime
from gpt_researcher import GPTResearcher
from dotenv import load_dotenv

load_dotenv()

# Ensure OpenAI API key is set
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY not set in .env file")


async def test_simple_query():
    """Test with a simple, focused query."""
    
    print("\n" + "="*80)
    print("TESTING GPT-RESEARCHER")
    print("="*80)
    print("\nRunning a simple test query...")
    print("This should take 2-3 minutes.\n")
    
    # Simple, focused query
    query = """What are the top 5 technical indicators used by quantitative traders 
    for predicting stock prices? Focus on indicators that have been validated 
    in academic research. Provide brief descriptions and formulas."""
    
    try:
        print(f"Query: {query}\n")
        print("Initializing researcher...")
        
        researcher = GPTResearcher(
            query=query,
            report_type="research_report",
            verbose=True
        )
        
        print("\nConducting research...")
        await researcher.conduct_research()
        
        print("\nGenerating report...")
        report = await researcher.write_report()
        
        # Save report
        output_dir = "research_reports"
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, "test_report.md")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Test Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Query**: {query}\n\n")
            f.write("---\n\n")
            f.write(report)
        
        print(f"\n[SUCCESS] Report saved to: {filepath}")
        print(f"Report length: {len(report)} characters")
        
        # Show first 500 chars
        print("\nFirst 500 characters of report:")
        print("="*80)
        print(report[:500])
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Research failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_simple_query())
    
    if success:
        print("\n[OK] GPT-Researcher is working correctly!")
        print("You can now run the full research_features.py script.")
    else:
        print("\n[FAIL] GPT-Researcher encountered errors.")
        print("Check the error message above for debugging.")
