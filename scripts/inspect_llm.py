# inspect_llm.py
# Small helper to apply importlib.metadata shim (like analyze_trade_agent.py) and inspect ChatOpenAI
try:
    import importlib.metadata as _stdlib_meta
    import importlib_metadata as _backport_meta
    _orig_version_fn = _stdlib_meta.version
    def _safe_version(name: str):
        try:
            return _orig_version_fn(name)
        except Exception:
            try:
                return _backport_meta.version(name)
            except Exception:
                return None
    _stdlib_meta.version = _safe_version
except Exception:
    pass

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-5.1')
print('ChatOpenAI type:', type(llm))
print('callable?', callable(llm))
print('public methods:')
for m in sorted([x for x in dir(llm) if not x.startswith('_')]):
    print(' ', m)

# Try some candidate call interfaces (safely)
from langchain.schema import HumanMessage
msgs = [HumanMessage(content='hello')]
for method in ('__call__','predict_messages','predict','generate','chat','arun'):
    if hasattr(llm, method):
        print('\nFound method', method)
        try:
            fn = getattr(llm, method)
            out = fn(msgs)
            print('  -> returned type:', type(out))
            if hasattr(out, 'content'):
                print('  -> content attr:', getattr(out, 'content'))
        except Exception as e:
            print('  -> call failed:', e)
