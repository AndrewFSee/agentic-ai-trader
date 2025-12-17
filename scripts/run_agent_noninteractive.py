from datetime import datetime
import sys
import os

# Ensure project root is on sys.path so top-level modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analyze_trade_agent import load_vectorstore, analyze_trade_agent


if __name__ == '__main__':
    vectordb = load_vectorstore()
    idea = "long or short recommendation"
    symbol = "NVDA"
    print(f"Running non-interactive agent for {symbol} with idea: {idea}")
    try:
        analysis = analyze_trade_agent(trading_idea=idea, symbol=symbol, vectordb=vectordb)
    except Exception as e:
        print("Agent raised an exception:", repr(e))
        raise

    print("Analysis length:", len(analysis) if analysis else 0)
    print("Preview:\n", (analysis[:1000] + '...') if analysis and len(analysis) > 1000 else analysis)

    out_path = f"research_reports/nvda_agent_analysis_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.md"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(analysis or "")
    print(f"Saved analysis to {out_path}")
