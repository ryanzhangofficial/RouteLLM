import torch
from huggingface_hub import PyTorchModelHubMixin
from transformers import BertModel, BertTokenizer

# Simplified MODEL_IDS with only Llama 3 series models
MODEL_IDS = {
    "meta-llama/Llama-3.2-1B": 0,
    "meta-llama/Llama-3.2-3B": 1,
    "meta-llama/Llama-3.3-70B-Instruct": 2,
}

class BertEmbeddingModel:
    """
    Uses a BERT model from Hugging Face transformers to compute sentence embeddings
    via average pooling over token embeddings.
    """
    def __init__(self, model_name="bert-base-uncased", device="cpu"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        self.model.eval()  # Set to evaluation mode

    def encode(self, text, convert_to_tensor=True):
        # Wrap a single string in a list
        if isinstance(text, str):
            text = [text]
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Obtain the last hidden state: (batch_size, sequence_length, hidden_size)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        # Compute weighted average using the attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        return embeddings if convert_to_tensor else embeddings.cpu().numpy()

class MFModel(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, dim, num_models, text_dim, num_classes, use_proj, device="cpu"):
        """
        Args:
            dim: output dimension for each model's embedding (e.g., 128)
            num_models: total number of models
            text_dim: expected dimension of text embedding as used during training (e.g. 1536)
            num_classes: number of classes (typically 1 for win rate regression)
            use_proj: whether to apply an additional projection layer
            device: device to run the model on
        """
        super().__init__()
        self._name = "TextMF"
        self.use_proj = use_proj
        self.P = torch.nn.Embedding(num_models, dim)
        self.device = device

        # Use our BERT-based embedding model (returns 768-dim vectors)
        self.bert_model = BertEmbeddingModel(model_name="bert-base-uncased", device=self.device)

        # If the checkpoint expects a different text_dim than 768, add an upsampling layer.
        if text_dim != 768:
            self.up_proj = torch.nn.Linear(768, text_dim, bias=False)
        else:
            self.up_proj = None

        if self.use_proj:
            self.text_proj = torch.nn.Sequential(
                torch.nn.Linear(text_dim, dim, bias=False)
            )
        else:
            # Without projection, the text_dim must equal the model embedding dimension.
            assert text_dim == dim, f"text_dim {text_dim} must be equal to dim {dim} if not using projection"

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(dim, num_classes, bias=False)
        )

    def get_device(self):
        return self.P.weight.device

    def forward(self, model_id, prompt):
        # Process model_id
        model_id = torch.tensor(model_id, dtype=torch.long).to(self.get_device())
        model_embed = self.P(model_id)
        model_embed = torch.nn.functional.normalize(model_embed, p=2, dim=1)

        # Obtain prompt embedding from BERT
        prompt_embed = self.bert_model.encode(prompt, convert_to_tensor=True)
        prompt_embed = prompt_embed.to(self.get_device())

        # Upsample if needed (e.g., 768 -> 1536)
        if self.up_proj is not None:
            prompt_embed = self.up_proj(prompt_embed)

        if self.use_proj:
            prompt_embed = self.text_proj(prompt_embed)

        return self.classifier(model_embed * prompt_embed).squeeze()

    @torch.no_grad()
    def pred_win_rate(self, model_a, model_b, prompt):
        logits = self.forward([model_a, model_b], prompt)
        winrate = torch.sigmoid(logits[0] - logits[1]).item()
        return winrate

    def load(self, path):
        self.load_state_dict(torch.load(path))
