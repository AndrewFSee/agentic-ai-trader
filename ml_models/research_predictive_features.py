"""
Deep Research: Predictive Features for ML Trading Models

Research Topics:
1. What features actually work in ML trading models? (academic literature)
2. Historical news sentiment data sources and methods
3. Options data from Polygon.io or alternatives
4. Alternative data sources for alpha generation
5. Feature engineering best practices for financial ML

Uses GPT-Researcher for autonomous deep research with citations.
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to path to import research_tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from gpt_researcher import GPTResearcher
    print("✓ GPT-Researcher installed")
except ImportError:
    print("✗ GPT-Researcher not found. Installing...")
    os.system("pip install gpt-researcher")
    from gpt_researcher import GPTResearcher


async def research_ml_trading_features():
    """Research what features actually predict stock returns."""
    
    query = """
    What features and data sources provide the most predictive power for machine learning 
    models forecasting stock price movements (3-10 day horizons)?
    
    Focus on:
    1. Academic research on ML trading (what actually works vs overhyped?)
    2. Industry practitioner insights (quant funds, systematic traders)
    3. Feature categories: technical, fundamental, sentiment, alternative data
    4. Why technical indicators alone fail (efficient markets at short horizons)
    5. Forward-looking vs backward-looking features
    6. Data sources that are accessible (not proprietary/expensive)
    
    Prioritize evidence-based findings with citations.
    """
    
    print("\n" + "="*80)
    print("RESEARCH 1: Predictive Features for ML Trading Models")
    print("="*80 + "\n")
    
    researcher = GPTResearcher(query=query, report_type="research_report")
    await researcher.conduct_research()
    report = await researcher.write_report()
    
    return {
        "topic": "ML Trading Features",
        "report": report,
        "sources": researcher.get_source_urls()
    }


async def research_historical_news_sentiment():
    """Research historical news sentiment data sources."""
    
    query = """
    How can I obtain historical news sentiment data for stocks to use as features 
    in machine learning trading models?
    
    Focus on:
    1. Free or affordable news data sources with historical archives
    2. Methods to scrape historical news (Finviz, Google News, etc.)
    3. Pre-computed sentiment datasets (Kaggle, academic sources)
    4. FinBERT and other financial sentiment models
    5. Techniques to backfill sentiment for past dates
    6. Legal/ethical considerations for news scraping
    7. API-based solutions (NewsAPI, GDELT, Bloomberg Terminal alternatives)
    
    Prioritize practical, implementable solutions.
    """
    
    print("\n" + "="*80)
    print("RESEARCH 2: Historical News Sentiment Data")
    print("="*80 + "\n")
    
    researcher = GPTResearcher(query=query, report_type="research_report")
    await researcher.conduct_research()
    report = await researcher.write_report()
    
    return {
        "topic": "Historical News Sentiment",
        "report": report,
        "sources": researcher.get_source_urls()
    }


async def research_options_data():
    """Research options data from Polygon and alternatives."""
    
    query = """
    How can I access historical options data (implied volatility, put/call ratios, 
    open interest) to use as features in ML trading models?
    
    Focus on:
    1. Polygon.io options endpoints and pricing tiers
    2. Alternative free/affordable options data providers
    3. Key options metrics for ML models (IV percentile, skew, put/call ratio)
    4. Options data as predictor of future stock moves (academic evidence)
    5. Historical options data availability (how far back?)
    6. Data quality and reliability considerations
    
    Prioritize accessible data sources for individual developers.
    """
    
    print("\n" + "="*80)
    print("RESEARCH 3: Options Data Sources")
    print("="*80 + "\n")
    
    researcher = GPTResearcher(query=query, report_type="research_report")
    await researcher.conduct_research()
    report = await researcher.write_report()
    
    return {
        "topic": "Options Data",
        "report": report,
        "sources": researcher.get_source_urls()
    }


async def research_alternative_data():
    """Research alternative data sources for alpha generation."""
    
    query = """
    What alternative data sources can enhance machine learning trading models 
    for predicting stock price movements?
    
    Focus on:
    1. Social media sentiment (Reddit, Twitter/X, StockTwits)
    2. Insider trading activity and SEC Form 4 filings
    3. Institutional holdings changes (13F filings)
    4. Google Trends and search volume data
    5. Satellite imagery and geolocation data (parking lots, shipping)
    6. Web scraping techniques (product reviews, job postings, etc.)
    7. Credit card data and consumer spending patterns
    8. Accessibility for individual developers (free or affordable)
    
    Provide specific data sources, APIs, and implementation approaches.
    """
    
    print("\n" + "="*80)
    print("RESEARCH 4: Alternative Data Sources")
    print("="*80 + "\n")
    
    researcher = GPTResearcher(query=query, report_type="research_report")
    await researcher.conduct_research()
    report = await researcher.write_report()
    
    return {
        "topic": "Alternative Data",
        "report": report,
        "sources": researcher.get_source_urls()
    }


async def research_feature_engineering():
    """Research feature engineering best practices for financial ML."""
    
    query = """
    What are the best practices for feature engineering in financial machine learning 
    models, especially for predicting stock price movements?
    
    Focus on:
    1. Marcos Lopez de Prado's techniques (Advances in Financial ML)
    2. Feature importance and selection methods
    3. Dealing with non-stationarity in financial time series
    4. Regime-based features and conditional models
    5. Interaction terms and polynomial features
    6. Temporal features (seasonality, day-of-week effects)
    7. Market microstructure features
    8. Avoiding data leakage and survivorship bias
    
    Emphasize practical, proven techniques with academic backing.
    """
    
    print("\n" + "="*80)
    print("RESEARCH 5: Feature Engineering Best Practices")
    print("="*80 + "\n")
    
    researcher = GPTResearcher(query=query, report_type="research_report")
    await researcher.conduct_research()
    report = await researcher.write_report()
    
    return {
        "topic": "Feature Engineering",
        "report": report,
        "sources": researcher.get_source_urls()
    }


async def main():
    """Run all research tasks and save comprehensive report."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE RESEARCH: Predictive Features for ML Trading")
    print("="*80)
    print("Running 5 deep research queries using GPT-Researcher...")
    print("Estimated time: 10-15 minutes")
    print("="*80 + "\n")
    
    results = []
    
    # Run all research tasks
    try:
        results.append(await research_ml_trading_features())
    except Exception as e:
        print(f"Error in Research 1: {e}")
    
    try:
        results.append(await research_historical_news_sentiment())
    except Exception as e:
        print(f"Error in Research 2: {e}")
    
    try:
        results.append(await research_options_data())
    except Exception as e:
        print(f"Error in Research 3: {e}")
    
    try:
        results.append(await research_alternative_data())
    except Exception as e:
        print(f"Error in Research 4: {e}")
    
    try:
        results.append(await research_feature_engineering())
    except Exception as e:
        print(f"Error in Research 5: {e}")
    
    # Save comprehensive report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/ml_models/feature_research_{timestamp}.json"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "research_count": len(results),
            "results": results
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("RESEARCH COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_file}")
    print(f"\nGenerated {len(results)} comprehensive research reports")
    
    # Also save markdown version
    md_file = output_file.replace('.json', '.md')
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# Comprehensive Research: Predictive Features for ML Trading\n\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write("---\n\n")
        
        for result in results:
            f.write(f"## {result['topic']}\n\n")
            f.write(result['report'])
            f.write("\n\n### Sources\n\n")
            for url in result.get('sources', []):
                f.write(f"- {url}\n")
            f.write("\n---\n\n")
    
    print(f"Markdown report saved to: {md_file}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
