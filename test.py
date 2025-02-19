import os
from routellm.controller import Controller

# OPENAI_key = os.environ["OPENAI_API_KEY"]
api_key = os.environ["HUGGINGFACEHUB_API_TOKEN"]

client = Controller(
    routers=["mf"],  # Use the Meta-Filtering router for routing decisions
    strong_model="meta-llama/Llama-3.3-70B-Instruct",  # Strong model (70B)
    weak_model="meta-llama/Llama-3.2-1B-Instruct",  # Weak model (1B),
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

print(response)