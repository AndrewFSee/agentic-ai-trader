# Runner: fetch fundamentals, fred macro, then run company research integrating fundamentals
import os, sys, json
sys.path.append(os.getcwd())
from research_tools import get_tool_function

OUT_DIR = os.path.join(os.getcwd(), 'research_reports')
os.makedirs(OUT_DIR, exist_ok=True)

state = {'tool_results': {}}

# 1) Fetch fundamentals
try:
    fn_fund = get_tool_function('fetch_fundamentals')
    fn_fund(state, {'symbol': 'NVDA'})
    fund = state['tool_results'].get('fetch_fundamentals')
    with open(os.path.join(OUT_DIR, 'nvda_fundamentals.json'), 'w', encoding='utf-8') as f:
        json.dump(fund, f, indent=2)
    print('Saved fundamentals -> research_reports/nvda_fundamentals.json')
except Exception as e:
    print('Fundamentals tool failed:', e)

# 2) Fetch FRED macro indicators
try:
    fn_fred = get_tool_function('fred_macro_indicators')
    fn_fred(state, {})
    fred = state['tool_results'].get('fred_macro_indicators')
    with open(os.path.join(OUT_DIR, 'nvda_fred.json'), 'w', encoding='utf-8') as f:
        json.dump(fred, f, indent=2)
    print('Saved FRED indicators -> research_reports/nvda_fred.json')
except Exception as e:
    print('FRED tool failed:', e)

# 3) Run company research (this will call GPT-Researcher)
try:
    fn_comp = get_tool_function('gpt_researcher_company_analysis')
    fn_comp(state, {'symbol': 'NVDA'})
    comp = state['tool_results'].get('gpt_researcher_company_analysis')
    # Save report text if available
    report_text = ''
    if isinstance(comp, dict):
        report_text = comp.get('report') or comp.get('research_result') or ''
    else:
        report_text = str(comp)

    if report_text:
        with open(os.path.join(OUT_DIR, 'nvda_company_research_with_fundamentals.md'), 'w', encoding='utf-8') as f:
            f.write(report_text)
        print('Saved company research report -> research_reports/nvda_company_research_with_fundamentals.md')
    else:
        print('No report text returned from company analysis; saving raw object')
        with open(os.path.join(OUT_DIR, 'nvda_company_research_raw.json'), 'w', encoding='utf-8') as f:
            json.dump(comp, f, indent=2)

except Exception as e:
    print('Company research tool failed:', e)

# 4) Save full state for debugging
with open(os.path.join(OUT_DIR, 'nvda_full_state.json'), 'w', encoding='utf-8') as f:
    json.dump(state, f, indent=2)

print('Done. State and outputs saved under research_reports/')
