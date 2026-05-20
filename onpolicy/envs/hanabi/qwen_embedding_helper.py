import hashlib
import numpy as np
import torch
from collections import OrderedDict
from sentence_transformers import SentenceTransformer


class QwenEmbeddingClient:
    def __init__(
        self,
        model_name="Qwen/Qwen3-Embedding-0.6B",
        device=None,
        normalize=True,
        output_dim=1024,
        cache_size=50000,
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.output_dim = output_dim

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[QWEN EMBED] loading {model_name} on {device}", flush=True)

        self.model = SentenceTransformer(model_name, device=device)
        self.model.eval()

        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.llm_feature_dim = output_dim

        print("[QWEN EMBED] loaded", flush=True)

    def _cache_key(self, text):
        payload = {
            "model": self.model_name,
            "normalize": self.normalize,
            "text": text,
        }
        return hashlib.sha256(str(payload).encode("utf-8")).hexdigest()

    @torch.no_grad()
    def embed_text(self, text):
        key = self._cache_key(text)

        if key in self.cache:
            self.cache_hits += 1
            emb = self.cache.pop(key)
            self.cache[key] = emb
            return emb.copy()

        self.cache_misses += 1

        emb = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )[0].astype(np.float32)

        # Qwen3-Embedding-0.6B should be 1024 by default.
        # This protects you if SentenceTransformer returns a different dim.
        if emb.shape[0] != self.output_dim:
            raise ValueError(
                f"Embedding dim mismatch: got {emb.shape[0]}, expected {self.output_dim}. "
                f"Check --llm_embedding_dim or model_name={self.model_name}"
            )

        self.cache[key] = emb
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return emb.copy()

    def verify(self):
        test = self.embed_text("test hanabi state")
        print(f"[QWEN EMBED] verify embedding shape = {test.shape}", flush=True)
        assert test.shape[0] == self.llm_feature_dim

    def get_action_from_context(self, llm_context, num_moves, hint_annotations=None):
        return self.get_embedding_from_context(llm_context, num_moves, hint_annotations)

    def get_embedding_from_context(self, llm_context, num_moves, hint_annotations=None):
        (
            text_obs,
            condensed_obs,
            legal_moves_dicts,
            legal_move_uids,
            game_info,
            hint_annotations_from_context,
        ) = llm_context

        if hint_annotations is None:
            hint_annotations = hint_annotations_from_context

        text = self._build_embedding_text(
            text_obs=text_obs,
            legal_moves_dicts=legal_moves_dicts,
            game_info=game_info,
            hint_annotations=hint_annotations,
        )

        vector = self.embed_text(text)

        metadata = {
            "mode": "qwen_embedding",
            "embedding_dim": vector.shape[0],
            "embedding_text": text,
        }

        return None, vector, metadata

    def _build_embedding_text(
        self,
        text_obs,
        legal_moves_dicts,
        game_info,
        hint_annotations=None,
    ):
        lines = []

        lines.append(
            "Instruction: Embed this Hanabi game state for a reinforcement learning policy. "
            "The embedding should represent strategic factors such as playable cards, "
            "dangerous discards, useful hints, information tokens, life tokens, and legal actions."
        )
        lines.append("")
        lines.append("Game info:")
        lines.append(str(game_info))
        lines.append("")
        lines.append("Observation:")
        lines.append(text_obs)
        lines.append("")
        lines.append("Legal moves:")

        for i, move in enumerate(legal_moves_dicts):
            extra = ""
            if hint_annotations and i in hint_annotations:
                extra = f" | hint_effect: {hint_annotations[i]}"
            lines.append(f"{i}: {move}{extra}")

        return "\n".join(lines)
