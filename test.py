import os
import pandas as pd
import re
from routellm.controller import Controller

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# OPENAI_key = os.environ["OPENAI_API_KEY"]
api_key = os.environ["HUGGINGFACEHUB_API_TOKEN"]

client = Controller(
    routers=["mf"],  # Use the Meta-Filtering router for routing decisions
    strong_model="meta-llama/Llama-3.1-8B-Instruct",  # Strong model (70B)
    weak_model="meta-llama/Llama-3.2-1B-Instruct",    # Weak model (1B)
    #     api_base="https://api-inference.huggingface.co/models",
    api_key=api_key,
    progress_bar=True,
)

# > python -m routellm.calibrate_threshold --routers mf --strong-model-pct 0.5 --config config.example.yaml
# For 50.0% strong model calls for mf, threshold = 0.11593

# response = client.chat.completions.create(
#   # This tells RouteLLM to use the MF router with a cost threshold of 0.11593
#   model="router-mf-0.11593",
#   messages=[
#     {"role": "user", "content": "Give a description of the Three Kingdoms period in China."}
#   ]
# )
#
# print(response)

df = pd.read_csv("boolq_downsample.csv")

def get_completion(row):
    """
    Build the prompt from input_text (question) + doc (passage),
    then ask for a strictly 'true' or 'false' answer on the first line.
    """
    question = row["input_text"]  
    passage = row["doc"]         

    prompt = (
        f"Question: {question}\n"
        f"Passage: {passage}\n"
        "Respond with ONLY the word 'true' or 'false' on the first line:\n"
    )
    messages = [{"role": "user", "content": prompt}]

    # Get the completion using the specified router threshold.
    # Now, completions.create returns both the response and the routed model.
    response, routed_model = client.chat.completions.create(
        model="router-mf-0.11593",
        messages=messages
    )

    answer_text = response[0].outputs[0].text.strip().lower()
    match = re.search(r'\b(true|false)\b', answer_text)
    if match:
        first_token = match.group(0)
        predicted = True if first_token == "true" else False
    else:
        predicted = None

    return predicted, answer_text, routed_model

# df_subset = df.head()
df_subset = df  

# Apply the function to each row (axis=1) and create new columns for:
# predicted answer, raw response, and routed_model.
df_subset[["predicted", "raw_response", "routed_model"]] = df_subset.apply(
    lambda row: pd.Series(get_completion(row)), axis=1
)

# Create a "correct" column by comparing predicted T/F to 'target'
# In this dataset, 'target' is an int: 1 means True, 0 means False.
df_subset["correct"] = df_subset.apply(
    lambda row: (row["predicted"] == (row["target"] == 1)) if row["predicted"] is not None else False,
    axis=1
)

print(df_subset)

final_df = df_subset[["doc", "input_text", "predicted", "raw_response", "routed_model", "correct"]]
final_df.to_csv("routellm_boolq.csv", index=False)
