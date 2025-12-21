"""Check production configuration."""
import os
from dotenv import load_dotenv

load_dotenv()

print("Production Configuration Check:")
print(f"  EDGE_SCORER_ENABLED: {os.getenv('EDGE_SCORER_ENABLED', 'NOT SET')}")
print(f"  EDGE_TRACE_LOGGING: {os.getenv('EDGE_TRACE_LOGGING', 'NOT SET')}")
print(f"  RERANKER_ENABLED: {os.getenv('RERANKER_ENABLED', 'NOT SET')}")
print(f"  RERANKER_SELECTIVE_MODE: {os.getenv('RERANKER_SELECTIVE_MODE', 'NOT SET')}")
