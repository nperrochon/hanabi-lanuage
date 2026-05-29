import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


class RemoteQwenEmbeddingClient:
    """
    Lightweight proxy used inside HanabiEnv subprocess workers.

    Important:
      - Does NOT import torch.
      - Does NOT import sentence_transformers.
      - Does NOT own a CUDA model.
      - Only talks to the Qwen embedding server over localhost HTTP.
    """

    def __init__(
        self,
        host="127.0.0.1",
        port=8765,
        timeout=120.0,
        max_retries=30,
        retry_sleep=2.0,
    ):
        self.host = host
        self.port = int(port)
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep

        self.cache_hits = 0
        self.cache_misses = 0

        health = self.verify()
        self.llm_feature_dim = int(health["embedding_dim"])

        print(
            f"[QWEN CLIENT] connected to {self.base_url}, dim={self.llm_feature_dim}",
            flush=True,
        )

    def _get_json(self, path):
        with urlopen(self.base_url + path, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post_json(self, path, payload):
        body = json.dumps(payload).encode("utf-8")
        req = Request(self.base_url + path, data=body, method="POST")
        req.add_header("Content-Type", "application/json")

        with urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def verify(self):
        last_err = None

        for attempt in range(1, self.max_retries + 1):
            try:
                health = self._get_json("/health")
                if health.get("ok"):
                    return health
                last_err = RuntimeError(f"Bad health payload: {health}")
            except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as e:
                last_err = e

            print(
                f"[QWEN CLIENT] waiting for server {self.base_url} "
                f"attempt {attempt}/{self.max_retries}: {last_err}",
                flush=True,
            )
            time.sleep(self.retry_sleep)

        raise RuntimeError(f"Could not connect to Qwen embedding server: {last_err}")

    def embed_text(self, text):
        url = f"{self.base_url}/embed"
        data = json.dumps({"text": text}).encode("utf-8")
        req = Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )

        with urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            # Convert the JSON list back to a numpy array for the RL code
            return np.array(result["embedding"])

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
            "mode": "qwen_embedding_server",
            "embedding_dim": int(vector.shape[0]),
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
        lines = [
            (
                "Instruction: Embed this Hanabi game state for a reinforcement learning policy. "
                "Represent strategic factors such as playable cards, dangerous discards, "
                "useful hints, information tokens, life tokens, and legal actions."
            ),
            "",
            "Game info:",
            str(game_info),
            "",
            "Observation:",
            text_obs,
            "",
            "Legal moves:",
        ]

        for i, move in enumerate(legal_moves_dicts):
            extra = ""
            if hint_annotations and i in hint_annotations:
                extra = f" | hint_effect: {hint_annotations[i]}"
            lines.append(f"{i}: {move}{extra}")

        return "\n".join(lines)
