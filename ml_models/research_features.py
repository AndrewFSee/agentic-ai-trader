"""
GPT-Researcher Feature Discovery

Uses gpt-researcher to find novel alpha factors and predictive features
for stock price prediction. Focuses on academic papers, quant research,
and alternative data sources.
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


async def research_features(query: str, report_type: str = "research_report") -> str:
    """
    Run GPT-Researcher on a specific query.
    
    Args:
        query: Research question
        report_type: Type of report to generate
            - research_report: Detailed academic research
            - quick_analysis: Fast summary
            - outline_report: Structured outline
    
    Returns:
        Markdown report with findings
    """
    print(f"\n{'='*80}")
    print(f"RESEARCHING: {query}")
    print(f"{'='*80}\n")
    
    researcher = GPTResearcher(
        query=query,
        report_type=report_type,
        verbose=True
    )
    
    # Conduct research
    await researcher.conduct_research()
    
    # Generate report
    report = await researcher.write_report()
    
    return report


async def main():
    """Run comprehensive feature discovery research."""
    
    print("\n" + "="*80)
    print("GPT-RESEARCHER: ALPHA FACTOR DISCOVERY")
    print("="*80)
    print("\nSearching academic papers, quant blogs, and research for novel features...")
    print("This will take 5-10 minutes per query.\n")
    
    # Research queries focused on different aspects
    queries = [
        # Core alpha factors
        {
            "query": """What are the most promising alpha factors for stock price prediction in 2024-2025? 
            Focus on academic papers and quantitative research. Include:
            - Alternative data sources (not just price/volume)
            - Machine learning features that have shown statistical significance
            - Signal processing techniques for time series
            - Cross-asset features (correlations with bonds, commodities, etc.)
            Provide specific feature formulas and references.""",
            "filename": "alpha_factors_2024_2025.md"
        },
        
        # Alternative data
        {
            "query": """What alternative data sources are used for stock prediction in quantitative finance? 
            Focus on:
            - Unconventional data (satellite imagery, web traffic, credit card data)
            - Social media and sentiment analysis techniques beyond simple polarity
            - Corporate event data and calendar effects
            - Supply chain and network analysis features
            Provide implementation details and data sources.""",
            "filename": "alternative_data_sources.md"
        },
        
        # Signal processing
        {
            "query": """What signal processing and time series techniques are used for stock price forecasting? 
            Focus on:
            - Advanced wavelet transforms and their parameters
            - Kalman filtering variants (extended, unscented, particle filters)
            - Empirical Mode Decomposition and Hilbert-Huang Transform
            - Spectral analysis and frequency domain features
            Provide Python implementation examples.""",
            "filename": "signal_processing_techniques.md"
        },
        
        # Market microstructure
        {
            "query": """What market microstructure features predict stock returns? 
            Focus on:
            - Order book features (bid-ask spread, depth, imbalance)
            - Trade-based features (VWAP deviation, trade size distribution)
            - High-frequency patterns that persist at daily frequency
            - Liquidity measures and their predictive power
            Provide feature formulas suitable for daily data.""",
            "filename": "market_microstructure_features.md"
        },
        
        # Regime detection
        {
            "query": """What are the state-of-the-art methods for market regime detection? 
            Focus on:
            - Hidden Markov Models and variants
            - Clustering methods for regime identification
            - Change point detection algorithms
            - Dynamic time warping and pattern matching
            Provide implementation details and optimal hyperparameters.""",
            "filename": "regime_detection_methods.md"
        }
    ]
    
    # Output directory
    output_dir = "research_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run each research query
    all_reports = {}
    
    for i, q in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"QUERY {i}/{len(queries)}: {q['filename']}")
        print(f"{'='*80}\n")
        
        try:
            report = await research_features(q['query'], report_type="research_report")
            
            # Save report
            filepath = os.path.join(output_dir, q['filename'])
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {q['filename'].replace('_', ' ').replace('.md', '').title()}\n\n")
                f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Query**: {q['query']}\n\n")
                f.write("---\n\n")
                f.write(report)
            
            print(f"\n✅ Saved: {filepath}")
            print(f"Length: {len(report)} characters\n")
            
            all_reports[q['filename']] = report
            
        except Exception as e:
            print(f"\n❌ Error researching {q['filename']}: {e}\n")
            continue
    
    # Generate master summary
    print("\n" + "="*80)
    print("GENERATING MASTER SUMMARY")
    print("="*80 + "\n")
    
    summary_query = f"""Based on the research findings, create a prioritized implementation plan for adding new features to a stock prediction ML pipeline. 
    Current features include: technical indicators, sentiment analysis, regime detection, fundamentals, volatility metrics, VIX, Kalman filters, and wavelets.
    
    Prioritize by:
    1. Implementation difficulty (easier = higher priority)
    2. Data availability (free/accessible data = higher priority)
    3. Expected impact on prediction accuracy
    
    Provide:
    - Top 10 features to implement next
    - Specific formulas and code snippets
    - Data sources and APIs
    - Expected alpha contribution (qualitative estimate)
    """
    
    try:
        summary = await research_features(summary_query, report_type="research_report")
        
        summary_file = os.path.join(output_dir, "implementation_plan.md")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"# Feature Implementation Plan\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Context\n\n")
            f.write("Based on comprehensive research of alpha factors, alternative data, ")
            f.write("signal processing, market microstructure, and regime detection.\n\n")
            f.write("---\n\n")
            f.write(summary)
        
        print(f"\n✅ Saved: {summary_file}\n")
        
    except Exception as e:
        print(f"\n❌ Error generating summary: {e}\n")
    
    print("\n" + "="*80)
    print("RESEARCH COMPLETE!")
    print("="*80)
    print(f"\nAll reports saved to: {output_dir}/")
    print(f"Total reports: {len(all_reports)}")
    print("\nNext steps:")
    print("1. Review implementation_plan.md for prioritized features")
    print("2. Implement top features in new modules")
    print("3. Test incrementally on 3-stock subset")
    print("4. Add to full pipeline if performance improves\n")


if __name__ == "__main__":
    asyncio.run(main())
