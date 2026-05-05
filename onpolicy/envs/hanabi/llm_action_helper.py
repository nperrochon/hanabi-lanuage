"""Client for prompting a local Ollama LLM for Hanabi action recommendations.
Outputs a vector that can be appended to the MAPPO observation for the PPO algorithm.

Requires Ollama running locally (ollama serve) and a model pulled (e.g. ollama pull qwen2:7b).
No API tokens, no cloud dependencies.

Usage with MAPPO:
  In the Hanabi env or runner, when it's an agent's turn:
    1. llm_context = env.get_llm_context()  # from HanabiEnv
    2. llm_vector = get_llm_obs_vector(llm_context, num_moves=env.num_moves())
       # default model qwen2:7b; override with model_name="llama3.2" etc.
    3. obs_for_policy = np.concatenate([obs, llm_vector], axis=-1)
  Extend observation_space when using LLM: obs_dim becomes obs_dim + env.num_moves().
"""

from __future__ import absolute_import, division

import json
import re
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import hashlib
from collections import OrderedDict


def build_hanabi_prompt(
    text_observation: str,
    legal_moves_dicts: List[Dict[str, Any]],
    legal_move_uids: List[int],
    game_info: Dict[str, Any],
    num_legal_descriptions: int = 12,
) -> str:
    """Build a text prompt for the Hanabi game state suitable for an LLM.

    Args:
        text_observation: Human-readable game state from pyhanabi (e.g. obs.text_observation()).
        legal_moves_dicts: List of legal moves as dicts (e.g. from move.to_dict()).
        legal_move_uids: List of move UIDs corresponding to legal_moves_dicts (for mapping back).
        game_info: Dict with fireworks, information_tokens, life_tokens, deck_size.
        num_legal_descriptions: Max number of legal moves to list in the prompt (to avoid overflow).

    Returns:
        A single string prompt for the model.
    """
    fireworks = game_info.get("fireworks", {})
    info_tokens = game_info.get("information_tokens", 0)
    life_tokens = game_info.get("life_tokens", 0)
    deck_size = game_info.get("deck_size", 0)

    prompt_parts_long = [
        "You are playing Hanabi, a cooperative card game. You see the following game state.",
        "",
                "## Short Hanabi rules",
        "- Cards are played to the fireworks piles in order: rank 1, then 2, then 3, etc. for each color.",
        "- A PLAY move succeeds only if the card is exactly the next rank needed for its color; otherwise you lose a life token.",
        "- If all life tokens are lost the game ends immediately.",
        "- REVEAL_COLOR and REVEAL_RANK consume an information token; you cannot hint if there are zero information tokens.",
        "- A good hint makes it easier for your teammate to know which card is safely playable soon, or which card is definitely not useful (can be discarded).",
        "- DISCARD returns one information token but permanently removes that card from the game.",
        "",
        "## Strategy reminder",
        "Your goal is to maximize the final fireworks score and avoid losing life tokens.",
        "Prefer PLAY when you can infer with high confidence that a card is safely playable (the next needed rank for its color).",
        "Otherwise, use REVEAL_COLOR or REVEAL_RANK hints that give new, useful information; avoid repeating hints that your teammate already knows.",
        "Cluing colors can be very helpful: reveal the color of cards that are likely playable now or will be needed soon, rather than giving redundant hints.",
        "",
        "## Game state",
        text_observation.strip(),
        "",
        "## Game summary",
        "Fireworks: " + str(fireworks),
        "Information tokens: " + str(info_tokens),
        "Life tokens: " + str(life_tokens),
        "Cards left in deck: " + str(deck_size),
        "",
        "## Legal moves (choose exactly one)",
    ]

    prompt_parts = [
        "Role: Hanabi AI (Cooperative). Goal: Maximize score, minimize life loss.",
        "",
        "## Rules & Strategy",
        "- Play sequence: 1-2-3-4-5 per color. Failed PLAY = -1 life. 0 life = Game Over.",
        "- Hint (Rank/Color) costs 1 info token. Discard adds 1 info token.",
        "- Priority: 1. PLAY (high confidence) | 2. HINT (new/useful info) | 3. DISCARD.",
        "- Strategy: Hint cards that are playable soon; avoid redundant info.",
        "",
        "## State",
        text_observation.strip(),
        f"Fireworks: {fireworks} | Info: {info_tokens} | Lives: {life_tokens} | Deck: {deck_size}",
        "",
        "## Select One Legal Move:"
    ]


    # Describe each legal move with an index for the model to refer to
    move_descriptions = []
    for i, move_dict in enumerate(legal_moves_dicts[:num_legal_descriptions]):
        at = move_dict.get("action_type", "")
        if at == "PLAY":
            move_descriptions.append("  {}: PLAY card at hand index {}".format(i, move_dict.get("card_index", "?")))
        elif at == "DISCARD":
            move_descriptions.append("  {}: DISCARD card at hand index {}".format(i, move_dict.get("card_index", "?")))
        elif at == "REVEAL_COLOR":
            move_descriptions.append("  {}: REVEAL_COLOR {} to teammate (offset {})".format(
                i, move_dict.get("color", "?"), move_dict.get("target_offset", "?")))
        elif at == "REVEAL_RANK":
            move_descriptions.append("  {}: REVEAL_RANK {} to teammate (offset {})".format(
                i, move_dict.get("rank", "?"), move_dict.get("target_offset", "?")))
        else:
            move_descriptions.append("  {}: {}".format(i, move_dict))

    prompt_parts.append("\n".join(move_descriptions))
    prompt_parts.append("")
    prompt_parts.append("Reply with ONLY a single integer: the index of your chosen move (0 to {}). No other text.".format(
        min(len(legal_moves_dicts), num_legal_descriptions) - 1))

    return "\n".join(prompt_parts)


def parse_llm_action_response(
    response: str,
    num_legal: int,
) -> Optional[int]:
    """Parse the LLM's text response to a legal move index in [0, num_legal-1].

    Tries, in order: "ACTION: N", then first number in range, then last number in range
    (many LLMs put the answer at the end: "I choose 2" or "The best move is 2.").
    """
    if not response or num_legal <= 0:
        print("No response or no legal moves")
        return None
    response = response.strip()
    # 1) Explicit "ACTION: N"
    match = re.search(r"ACTION\s*:\s*(\d+)", response, re.IGNORECASE)
    if match:
        idx = int(match.group(1))
        if 0 <= idx < num_legal:
            return idx
    # 2) All numbers in range, take first then last
    numbers = [int(m.group(1)) for m in re.finditer(r"\b(\d+)\b", response)]
    for idx in numbers:
        if 0 <= idx < num_legal:
            return idx
    for idx in reversed(numbers):
        if 0 <= idx < num_legal:
            return idx
    print("No valid move found")
    print("Response: ", response)
    print("Numbers: ", numbers)
    print("Num legal: ", num_legal)
    return None


def llm_context_to_action_vector(
    num_moves: int,
    suggested_move_uid: Optional[int] = None,
    llm_context: Optional[Tuple] = None,
) -> np.ndarray:
    """Produce a vector of shape (num_moves,) that can be appended to the MAPPO observation.

    If suggested_move_uid is provided and valid, the vector is one-hot at that move index.
    Otherwise (no context, or invalid), the vector is zeros.

    Args:
        num_moves: Total number of moves in the game (e.g. env.num_moves()).
        suggested_move_uid: Move UID suggested by the LLM (0 <= suggested_move_uid < num_moves).
        llm_context: Unused; reserved for future extensions (e.g. soft logits).

    Returns:
        np.ndarray of shape (num_moves,) and dtype np.float32.
    """
    out = np.zeros(num_moves, dtype=np.float32)
    if suggested_move_uid is not None and 0 <= suggested_move_uid < num_moves:
        out[suggested_move_uid] = 1.0
    return out


class HanabiOllamaClient:
    """Client for querying a local Ollama server for Hanabi action recommendations.

    Requires Ollama running (ollama serve) and a model pulled (e.g. ollama pull qwen2:7b).
    """
    CACHE_MISS = object()

    def __init__(
        self,
        model_name: str = "qwen2:7b",
        base_url: str = "http://localhost:11434",
        max_new_tokens: int = 4,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: float = 60.0,
        cache_size: int = 50000,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout

        # LRU cache: prompt/settings hash -> suggested move UID or None
        self.cache_size = cache_size
        self._cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def _make_cache_key(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str):
        if key not in self._cache:
            self.cache_misses += 1
            return HanabiOllamaClient.CACHE_MISS

        self.cache_hits += 1
        value = self._cache.pop(key)
        self._cache[key] = value  # mark as recently used
        return value

    def _cache_put(self, key: str, value):
        self._cache[key] = value
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def verify(self) -> None:
        """Verify Ollama is reachable and the model exists. Call at startup; raises if not ready."""
        from urllib.request import Request, urlopen
        from urllib.error import URLError, HTTPError

        url = "{}/api/generate".format(self.base_url)
        payload = {
            "model": self.model_name,
            "prompt": "1",
            "stream": False,
            "options": {"num_predict": 1},
        }
        body = json.dumps(payload).encode("utf-8")
        req = Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urlopen(req, timeout=min(10.0, self.timeout)) as resp:
                json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 404:
                hint = "Use an Ollama tag, not a HuggingFace ID. " if ("/" in self.model_name or self.model_name != self.model_name.lower()) else ""
                raise RuntimeError(
                    "Ollama model '{}' not found. {}".format(
                        self.model_name, hint
                    )
                ) from e
            raise RuntimeError(
                "Ollama request failed (HTTP {}). Is Ollama running? Try: ollama serve".format(e.code)
            ) from e
        except URLError as e:
            raise RuntimeError(
                "Cannot reach Ollama at {}. Is it running? Try: ollama serve".format(self.base_url)
            ) from e

    def get_action_from_context(
        self,
        llm_context: Tuple[str, List[Dict], List[int], Dict],
        num_moves: int,
    ) -> Tuple[Optional[int], np.ndarray]:
        text_obs, legal_moves_dicts, legal_move_uids, game_info = llm_context

        if not legal_moves_dicts or not legal_move_uids:
            return None, llm_context_to_action_vector(num_moves, None)

        prompt = build_hanabi_prompt(
            text_obs,
            legal_moves_dicts,
            legal_move_uids,
            game_info,
        )

        cache_key = self._make_cache_key(prompt)
        cached_suggested_uid = self._cache_get(cache_key)

        if cached_suggested_uid is not HanabiOllamaClient..CACHE_MISS:
            return cached_suggested_uid, llm_context_to_action_vector(
                num_moves,
                cached_suggested_uid,
            )

        # Important: cache miss is the only place Ollama is called.
        response = self._generate(prompt)

        num_legal = len(legal_move_uids)
        idx = parse_llm_action_response(response, num_legal)

        if idx is not None:
            suggested_uid = legal_move_uids[idx]
        else:
            suggested_uid = None

        self._cache_put(cache_key, suggested_uid)

        return suggested_uid, llm_context_to_action_vector(num_moves, suggested_uid)

    def _generate(self, prompt: str) -> str:
        from urllib.request import Request, urlopen
        from urllib.error import URLError, HTTPError

        url = "{}/api/generate".format(self.base_url)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "72h",
            "options": {
                "num_predict": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        req = Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 404:
                # Model not found in Ollama (e.g. wrong tag; Ollama uses "tinyllama" not "TinyLlama/...")
                return ""
            raise RuntimeError(
                "Ollama request failed (HTTP {}). Is Ollama running? For 404: model not found - try 'ollama pull {}' (use Ollama tag, e.g. tinyllama not TinyLlama/...).".format(
                    e.code, self.model_name
                )
            ) from e
        except URLError as e:
            return ""  # e.g. connection refused when Ollama not running
        return data.get("response", "").strip()

def get_llm_obs_vector(
    llm_context: Optional[Tuple],
    num_moves: int,
    client: Optional["HanabiOllamaClient"] = None,
) -> np.ndarray:
    """Return the (num_moves,) observation vector from LLM context using local Ollama.

    Caller must pass a HanabiOllamaClient when llm_context is not None.

    Args:
        llm_context: From HanabiEnv.get_llm_context(), or None (returns zeros).
        num_moves: env.num_moves().
        client: HanabiOllamaClient to use; required when llm_context is not None.

    Returns:
        np.ndarray of shape (num_moves,) and dtype np.float32.
    """
    if llm_context is None:
        raise ValueError("llm_context is required when using get_llm_obs_vector")
    if client is None:
        raise TypeError("client is required when llm_context is not None")
    _, vector = client.get_action_from_context(llm_context, num_moves)
    return vector
