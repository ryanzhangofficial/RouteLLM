import os
from routellm.controller import Controller

import litellm
litellm._turn_on_debug()

# os.environ["OPENAI_API_KEY"] = "sk-mhwFzxnx33jc0ke6MNu9eZNdNtG_hVr4_P7nqoFr0KT3BlbkFJFbbM3l7MYebIcrL6QHuqbgoMlTgjtV7WE7PkpkZQoA"
api_key = os.environ["HUGGINGFACEHUB_API_TOKEN"]

client = Controller(
    routers=["mf"],  # Use the Meta-Filtering router for routing decisions
    strong_model="huggingface/meta-llama/Llama-3.2-3B",  # Strong model (8B)
    weak_model="huggingface/meta-llama/Llama-3.2-1B",  # Weak model (1B),
#     api_base="https://api-inference.huggingface.co/models",
    api_key=api_key,
    progress_bar=True,
)

# > python -m routellm.calibrate_threshold --routers mf --strong-model-pct 0.5 --config   config.example.yaml
# For 50.0% strong model calls for mf, threshold = 0.11593

response = client.chat.completions.create(
  # This tells RouteLLM to use the MF router with a cost threshold of 0.11593
  model="router-mf-0.11593",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)