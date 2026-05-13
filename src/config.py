DRAFT_MODEL_ID = "google/gemma-3-1b-it"
TARGET_MODEL_ID = "google/gemma-3-12b-it"
PRECISION_CHOICES = ("bf16", "8bit", "4bit", "cpu")
DEFAULT_PRECISION = "bf16"
DRAFT_DEVICE_MAP = "auto"
TARGET_DEVICE_MAP = "auto"
DEFAULT_DRAFT_K = 5
DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.9
BENCHMARK_DEFAULT_DRAFT_K = 1
BENCHMARK_DEFAULT_TEMPERATURE = 0.0
BENCHMARK_PROMPTS = [
    "Explain how transformers work in machine learning.",
    "Write a Python implementation of merge sort with comments.",
    "Write a short story about a robot learning to paint.",
    "What are the tradeoffs between SQL and NoSQL databases?",
]
