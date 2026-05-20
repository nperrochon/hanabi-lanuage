from sentence_transformers import SentenceTransformer
import torch


class HanabiTextEmbedder:
    def __init__(
        self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda"
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts):
        """
        texts: list[str]
        returns: torch.Tensor [batch, embedding_dim]
        """
        emb = self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb
