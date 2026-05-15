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

print("[BERT] about to import torch", flush=True)
import torch

print("[BERT] imported torch", flush=True)

print("[BERT] about to import transformers", flush=True)
from transformers import AutoTokenizer, AutoModel

print("[BERT] imported transformers", flush=True)

print("[BERT] imported torch/transformers", flush=True)

import json
import re
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import hashlib
from collections import OrderedDict


_GLOBAL_TEXT_ANALYSIS_ENCODER = None


def preload_text_analysis_encoder(
    model_name: str = "prajjwal1/bert-tiny",
    device: str = "cpu",
) -> "TextAnalysisEncoder":
    global _GLOBAL_TEXT_ANALYSIS_ENCODER

    if _GLOBAL_TEXT_ANALYSIS_ENCODER is None:
        print(
            "[BERT PRELOAD] loading TextAnalysisEncoder in parent process", flush=True
        )
        _GLOBAL_TEXT_ANALYSIS_ENCODER = TextAnalysisEncoder(
            model_name=model_name,
            device=device,
        )
        print(
            f"[BERT PRELOAD] loaded encoder dim={_GLOBAL_TEXT_ANALYSIS_ENCODER.output_dim}",
            flush=True,
        )

    return _GLOBAL_TEXT_ANALYSIS_ENCODER


def get_text_analysis_encoder() -> "TextAnalysisEncoder":
    global _GLOBAL_TEXT_ANALYSIS_ENCODER

    if _GLOBAL_TEXT_ANALYSIS_ENCODER is None:
        # Fallback path. Ideally this should not happen in SubprocVecEnv.
        print(
            "[BERT WARNING] encoder was not preloaded; loading in this process",
            flush=True,
        )
        _GLOBAL_TEXT_ANALYSIS_ENCODER = TextAnalysisEncoder()

    return _GLOBAL_TEXT_ANALYSIS_ENCODER


def build_hanabi_prompt(
    text_observation: str,
    legal_moves_dicts: List[Dict[str, Any]],
    legal_move_uids: List[int],
    game_info: Dict[str, Any],
    num_legal_descriptions: int = 100,
    hint_annotations: Optional[List[str]] = None,
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

    prompt_parts_one_hot = [
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
        "## Legal moves (choose exactly one). Think briefly, then output exactly: ACTION: <index>",
    ]

    prompt_parts_short = [
        "Role: Hanabi AI (Cooperative). Goal: Maximize score, minimize life loss.",
        "",
        "## Rules & Strategy",
        "- Play sequence: 1-2-3-4-5 per color. Failed PLAY = -1 life. 0 life = Game Over.",
        "- Hint (Rank/Color) costs 1 info token. Discard adds 1 info token.",
        "- Priority: 1. PLAY (high confidence) | 2. HINT (new/useful info) | 3. DISCARD.",
        "- Strategy: Hint cards that are playable soon; avoid redundant info.",
        "## State Key",
        "Format: [PlausibleColors : PlausibleRanks]. Example: [RY:1] means card is Red or Yellow, and definitely Rank 1.",
        "",
        "## State",
        text_observation.strip(),
        f"Fireworks: {fireworks} | Info: {info_tokens} | Lives: {life_tokens} | Deck: {deck_size}",
        "",
        "## Select One Legal Move:",
    ]

    prompt_parts_condensed_obs = [
        "You are playing Hanabi, a cooperative card game. You see the following game state.",
        "",
        "## State Legend",
        "Format: [PlausibleColors : PlausibleRanks].",
        "Example: [RG:12] means the card is Red or Green, and Rank 1 or 2.",
        "Example: [B:3] means the card is definitely Blue Rank 3.",
        "",
        "## Short Hanabi rules",
        # ... (keep existing rules)
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
        "## Game summary",
        "Fireworks: " + str(fireworks),
        "Information tokens: " + str(info_tokens),
        "Life tokens: " + str(life_tokens),
        "",
        "## Game state",
        text_observation.strip(),
        "",
        "## Legal moves (choose exactly one)",
    ]

    prompt_parts = [
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
        "When scoring hint moves:",
        "- A hint is useful if it tells the teammate about a card that is playable now.",
        "- A card is playable now if its rank equals fireworks[color] + 1.",
        "- Prefer hints that touch at least one currently playable teammate card.",
        "- Prefer hints that identify fewer irrelevant non-playable cards.",
        "- Do not give a high score to a hint just because it names a color/rank present in the teammate hand.",
        "- If a color/rank hint touches both playable and non-playable cards, score it lower than a hint that mostly identifies playable cards.",
        "",
    ]

    prompt_parts.append(
        f"""
        ## Output format

        Score every legal move index from 0 to {len(legal_moves_dicts) - 1}.

        Use scores from -2 to 2:
        - 2 = clearly good
        - 1 = reasonable
        - 0 = uncertain / neutral
        - -1 = risky
        - -2 = clearly bad

        Scoring rules:
        - Only set "best" when exactly one move has the highest positive score.
        - Do not set "best" to a move with score 0 or negative.
        - If there is no clearly good move, or if multiple moves are tied for best, set "best": null.
        - Give at most two moves score 1.
        - Most moves should be 0 or negative.
        - Do NOT give every PLAY or every HINT the same score.
        - Blind PLAY moves should be -2 unless the acting player's own knowledge makes the card definitely playable.
        - Blind DISCARD moves should be -1 or -2 unless information tokens are 0.
        - If information tokens are available and no play is certainly safe, the best move should usually be a useful hint.
        - A useful hint should identify a teammate card that is playable now or soon.

        Before scoring hints:
        1. List mentally which visible teammate cards are currently playable.
        2. A visible teammate card is playable if rank == fireworks[color] + 1.
        3. Hints that identify currently playable cards should usually receive score 2 or 1.
        4. Hints that identify only non-playable cards should receive 0 or negative.
        5. If two hints are equally useful, either score both 1 and set "best": null, or choose the one identifying fewer irrelevant cards as score 2.

        Return JSON only in this exact schema:
        {{"scores": {{"<legal_idx>": <score>}}, "best": <legal_idx>}}

        The "scores" object must include every legal move index from 0 to {len(legal_moves_dicts) - 1}.
        Do not copy the placeholder text literally. Replace placeholders with actual indices and scores.
        Do not prefer a move because of its index number. Legal indices have no strategic meaning.
        Evaluate the action description and game state only.
        """
    )

    prompt_parts.append(
        f"""
        ## Game state
        {text_observation.strip()}
        ## Game summary
        Fireworks: {str(fireworks)}
        Information tokens: {str(info_tokens)}
        Life tokens: {str(life_tokens)}
        Cards left in deck: {str(deck_size)}
        ## Legal moves:
        """
    )

    # Describe each legal move with an index for the model to refer to
    move_descriptions = []
    for i, move_dict in enumerate(legal_moves_dicts[:num_legal_descriptions]):
        at = move_dict.get("action_type", "")

        if at == "PLAY":
            line = "  {}: PLAY card at hand index {}".format(
                i, move_dict.get("card_index", "?")
            )

        elif at == "DISCARD":
            line = "  {}: DISCARD card at hand index {}".format(
                i, move_dict.get("card_index", "?")
            )

        elif at == "REVEAL_COLOR":
            line = "  {}: REVEAL_COLOR {} to teammate (offset {})".format(
                i, move_dict.get("color", "?"), move_dict.get("target_offset", "?")
            )

        elif at == "REVEAL_RANK":
            rank_val = move_dict.get("rank", "?")
            if isinstance(rank_val, int):
                rank_val += 1

            line = "  {}: REVEAL_RANK {} to teammate (offset {})".format(
                i, rank_val, move_dict.get("target_offset", "?")
            )

        else:
            line = "  {}: {}".format(i, move_dict)

        ann = None
        if hint_annotations is not None:
            if isinstance(hint_annotations, dict):
                ann = hint_annotations.get(i)
            elif i < len(hint_annotations):
                ann = hint_annotations[i]

        if ann:
            line += f" -> {ann}"

        move_descriptions.append(line)

    prompt_parts.append("\n".join(move_descriptions))
    prompt_parts.append("")
    # prompt_parts.append(
    #     "Reply with ONLY a single integer: the index of your chosen move (0 to {}). No other text.".format(
    #         min(len(legal_moves_dicts), num_legal_descriptions) - 1
    #     )
    # )

    return "\n".join(prompt_parts)


def build_hanabi_analysis_prompt(
    text_observation: str,
    legal_moves_dicts: List[Dict[str, Any]],
    legal_move_uids: List[int],
    game_info: Dict[str, Any],
    num_legal_descriptions: int = 100,
    hint_annotations: Optional[List[str]] = None,
) -> str:
    fireworks = game_info.get("fireworks", {})
    info_tokens = game_info.get("information_tokens", 0)
    life_tokens = game_info.get("life_tokens", 0)
    deck_size = game_info.get("deck_size", 0)

    move_descriptions = []
    for i, move_dict in enumerate(legal_moves_dicts[:num_legal_descriptions]):
        at = move_dict.get("action_type", "")

        if at == "PLAY":
            line = f"{i}: PLAY card at hand index {move_dict.get('card_index', '?')}"
        elif at == "DISCARD":
            line = f"{i}: DISCARD card at hand index {move_dict.get('card_index', '?')}"
        elif at == "REVEAL_COLOR":
            line = (
                f"{i}: REVEAL_COLOR {move_dict.get('color', '?')} "
                f"to teammate offset {move_dict.get('target_offset', '?')}"
            )
        elif at == "REVEAL_RANK":
            rank_val = move_dict.get("rank", "?")
            if isinstance(rank_val, int):
                rank_val += 1
            line = (
                f"{i}: REVEAL_RANK {rank_val} "
                f"to teammate offset {move_dict.get('target_offset', '?')}"
            )
        else:
            line = f"{i}: {move_dict}"

        ann = None
        if hint_annotations is not None:
            if isinstance(hint_annotations, dict):
                ann = hint_annotations.get(i)
            elif i < len(hint_annotations):
                ann = hint_annotations[i]

        if ann:
            line += f" -> {ann}"

        move_descriptions.append(line)

    return f"""
        You are analyzing a Hanabi game state for a reinforcement learning agent.

        Do NOT choose a move.
        Do NOT score legal moves.
        Do NOT output JSON.

        Give compact strategic information that may help the RL policy.

        Use exactly this format:

        SAFE_PLAY: yes/no/uncertain
        HINT_USEFUL: yes/no/uncertain
        DISCARD_RISK: low/medium/high
        STATE_RISK: low/medium/high
        ANALYSIS: one short sentence

        Game summary:
        Fireworks: {fireworks}
        Information tokens: {info_tokens}
        Life tokens: {life_tokens}
        Cards left in deck: {deck_size}

        Game state:
        {text_observation.strip()}

        Legal moves:
        {chr(10).join(move_descriptions)}
        """.strip()


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

    # print("No valid move found")
    # print("Response: ", response)
    # print("Numbers: ", numbers)
    # print("Num legal: ", num_legal)
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


def llm_context_to_score_vector(
    num_moves: int,
    legal_move_uids: List[int],
    scores_by_legal_idx: Dict[int, float],
    normalize: bool = True,
) -> np.ndarray:
    """
    Produce a vector of shape (num_moves,).

    Non-legal moves stay 0.
    Legal moves get LLM scores mapped by UID.

    If normalize=True:
      raw score [-2, 2] becomes [-1, 1]
    """
    out = np.zeros(num_moves, dtype=np.float32)

    for legal_idx, score in scores_by_legal_idx.items():
        if 0 <= legal_idx < len(legal_move_uids):
            uid = legal_move_uids[legal_idx]
            if 0 <= uid < num_moves:
                val = float(score)
                if normalize:
                    val = val / 2.0
                out[uid] = val

    return out


def scores_to_softmax_vector(
    num_moves: int,
    legal_move_uids: List[int],
    scores_by_legal_idx: Dict[int, float],
    temperature: float = 1.0,
) -> np.ndarray:
    out = np.zeros(num_moves, dtype=np.float32)

    if not legal_move_uids:
        return out

    scores = np.array(
        [scores_by_legal_idx.get(i, 0.0) for i in range(len(legal_move_uids))],
        dtype=np.float32,
    )

    scores = scores / max(temperature, 1e-6)
    scores = scores - np.max(scores)
    probs = np.exp(scores)
    probs = probs / max(np.sum(probs), 1e-8)

    for uid, prob in zip(legal_move_uids, probs):
        if 0 <= uid < num_moves:
            out[uid] = prob

    return out


def parse_llm_score_response(
    response: str, num_legal: int
) -> Tuple[Dict[int, float], Optional[int]]:
    """
    Parse LLM JSON score response:
      {"scores": {"0": -2, "1": 1, ...}, "best": 1}

    Returns:
      scores_by_legal_idx: dict mapping legal move index -> score in [-2, 2]
      best_idx: optional legal move index
    """
    if not response or num_legal <= 0:
        return {}, None

    response = response.strip()

    # Try to extract JSON object even if model adds text.
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return {}, parse_llm_action_response(response, num_legal)

    try:
        data = json.loads(match.group(0))
    except Exception:
        return {}, parse_llm_action_response(response, num_legal)

    raw_scores = data.get("scores", {})
    scores = {}

    if isinstance(raw_scores, dict):
        for k, v in raw_scores.items():
            try:
                idx = int(k)
                score = float(v)
            except Exception:
                continue

            if 0 <= idx < num_legal:
                scores[idx] = max(-2.0, min(2.0, score))

    best_idx = None
    try:
        b = int(data.get("best"))
        if 0 <= b < num_legal:
            best_idx = b
    except Exception:
        pass

    # Always choose the highest-scored legal move.
    # This avoids cases where the LLM says best=0 but gives move 0 a worse score.
    if scores:
        max_score = max(scores.values())
        best_candidates = [i for i, s in scores.items() if s == max_score]

        # Only trust LLM if there is a meaningful positive preference.
        if len(best_candidates) == 1 and max_score >= 1.0:
            best_idx = best_candidates[0]
        else:
            best_idx = None

    return scores, best_idx


# class HanabiOllamaClient:
#     """Client for querying a local Ollama server for Hanabi action recommendations.

#     Requires Ollama running (ollama serve) and a model pulled (e.g. ollama pull qwen2:7b).
#     """

#     CACHE_MISS = object()

#     def __init__(
#         self,
#         model_name: str = "qwen2:7b",
#         base_url: str = "http://localhost:11434",
#         max_new_tokens: int = 256,
#         temperature: float = 0.0,
#         top_p: float = 1.0,
#         timeout: float = 60.0,
#         cache_size: int = 50000,
#     ):
#         self.model_name = model_name
#         self.base_url = base_url.rstrip("/")
#         self.max_new_tokens = max_new_tokens
#         self.temperature = temperature
#         self.top_p = top_p
#         self.timeout = timeout

#         # LRU cache: prompt/settings hash -> suggested move UID or None
#         self.cache_size = cache_size
#         self._cache = OrderedDict()
#         self.cache_hits = 0
#         self.cache_misses = 0

#     def _make_cache_key(self, prompt: str) -> str:
#         payload = {
#             "model": self.model_name,
#             "prompt": prompt,
#             "max_new_tokens": self.max_new_tokens,
#             "temperature": self.temperature,
#             "top_p": self.top_p,
#         }
#         serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
#         return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

#     def _cache_get(self, key: str):
#         if key not in self._cache:
#             self.cache_misses += 1
#             return HanabiOllamaClient.CACHE_MISS

#         self.cache_hits += 1
#         value = self._cache.pop(key)
#         self._cache[key] = value  # mark as recently used
#         return value

#     def _cache_put(self, key: str, value):
#         self._cache[key] = value
#         if len(self._cache) > self.cache_size:
#             self._cache.popitem(last=False)

#     def verify(self) -> None:
#         """Verify Ollama is reachable and the model exists with robust retries."""
#         from urllib.request import Request, urlopen
#         from urllib.error import URLError, HTTPError
#         import time

#         url = "{}/api/generate".format(self.base_url)
#         payload = {
#             "model": self.model_name,
#             "prompt": "1",
#             "stream": False,
#             "options": {"num_predict": 1},
#         }
#         body = json.dumps(payload).encode("utf-8")

#         # Robust retry loop
#         max_retries = 10
#         for attempt in range(max_retries):
#             try:
#                 req = Request(url, data=body, method="POST")
#                 req.add_header("Content-Type", "application/json")
#                 # High timeout for cluster environments
#                 with urlopen(req, timeout=60.0) as resp:
#                     json.loads(resp.read().decode("utf-8"))
#                     return  # Success!
#             except Exception as e:
#                 if attempt < max_retries - 1:
#                     wait = (attempt + 1) * 5
#                     print(
#                         f"Ollama verify attempt {attempt+1} failed. Retrying in {wait}s... Error: {e}"
#                     )
#                     time.sleep(wait)
#                 else:
#                     raise RuntimeError(
#                         f"Could not connect to Ollama after {max_retries} attempts: {e}"
#                     )

#     def get_action_from_context(
#         self,
#         llm_context: Tuple[str, List[Dict], List[int], Dict],
#         num_moves: int,
#         hint_annotations: Optional[List[str]] = None,
#     ) -> Tuple[Optional[int], np.ndarray]:
#         text_obs, condensed_obs, legal_moves_dicts, legal_move_uids, game_info = (
#             llm_context
#         )

#         if not legal_moves_dicts or not legal_move_uids:
#             return None, np.zeros(num_moves, dtype=np.float32), {}

#         prompt = build_hanabi_prompt(
#             condensed_obs,
#             legal_moves_dicts,
#             legal_move_uids,
#             game_info,
#             hint_annotations=hint_annotations,
#         )

#         if not hasattr(self, "_printed_first_prompt"):
#             print("\n" + "=" * 100, flush=True)
#             print("[FIRST LLM PROMPT SENT]", flush=True)
#             print(prompt[:8000], flush=True)
#             print("=" * 100 + "\n", flush=True)
#             self._printed_first_prompt = True

#         cache_key = self._make_cache_key(prompt)
#         cached_suggested_uid = self._cache_get(cache_key)

#         if cached_suggested_uid is not HanabiOllamaClient.CACHE_MISS:
#             return cached_suggested_uid, llm_context_to_action_vector(
#                 num_moves,
#                 cached_suggested_uid,
#             )

#         # Important: cache miss is the only place Ollama is called.
#         response = self._generate(prompt)

#         num_legal = len(legal_move_uids)
#         idx = parse_llm_action_response(response, num_legal)

#         if idx is not None:
#             suggested_uid = legal_move_uids[idx]
#         else:
#             suggested_uid = None

#         self._cache_put(cache_key, suggested_uid)

#         return suggested_uid, llm_context_to_action_vector(num_moves, suggested_uid)

#     def _generate(self, prompt: str) -> str:
#         from urllib.request import Request, urlopen
#         from urllib.error import URLError, HTTPError

#         url = "{}/api/generate".format(self.base_url)
#         payload = {
#             "model": self.model_name,
#             "prompt": prompt,
#             "stream": False,
#             "keep_alive": "72h",
#             "options": {
#                 "num_predict": self.max_new_tokens,
#                 "temperature": self.temperature,
#                 "top_p": self.top_p,
#             },
#         }
#         body = json.dumps(payload).encode("utf-8")
#         req = Request(url, data=body, method="POST")
#         req.add_header("Content-Type", "application/json")
#         try:
#             with urlopen(req, timeout=self.timeout) as resp:
#                 data = json.loads(resp.read().decode("utf-8"))
#         except HTTPError as e:
#             if e.code == 404:
#                 # Model not found in Ollama (e.g. wrong tag; Ollama uses "tinyllama" not "TinyLlama/...")
#                 return ""
#             raise RuntimeError(
#                 "Ollama request failed (HTTP {}). Is Ollama running? For 404: model not found - try 'ollama pull {}' (use Ollama tag, e.g. tinyllama not TinyLlama/...).".format(
#                     e.code, self.model_name
#                 )
#             ) from e
#         except URLError as e:
#             return ""  # e.g. connection refused when Ollama not running
#         return data.get("response", "").strip()


# def get_llm_obs_vector(
#     llm_context: Optional[Tuple],
#     num_moves: int,
#     client: Optional["HanabiOllamaClient"] = None,
# ) -> np.ndarray:
#     """Return the (num_moves,) observation vector from LLM context using local Ollama.

#     Caller must pass a HanabiOllamaClient when llm_context is not None.

#     Args:
#         llm_context: From HanabiEnv.get_llm_context(), or None (returns zeros).
#         num_moves: env.num_moves().
#         client: HanabiOllamaClient to use; required when llm_context is not None.

#     Returns:
#         np.ndarray of shape (num_moves,) and dtype np.float32.
#     """
#     if llm_context is None:
#         raise ValueError("llm_context is required when using get_llm_obs_vector")
#     if client is None:
#         raise TypeError("client is required when llm_context is not None")
#     _, vector, _ = client.get_action_from_context(llm_context, num_moves)
#     return vector


class HanabiVLLMClient:
    """Client for querying a local vLLM OpenAI-compatible server.

    Start server first, e.g.:
      vllm serve Qwen/Qwen2.5-1.5B-Instruct \
        --host 127.0.0.1 \
        --port 8000 \
        --generation-config vllm \
        --enable-prefix-caching
    """

    CACHE_MISS = object()

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        llm_vector_mode: str = "softmax",
        base_url: str = "http://127.0.0.1:8000/v1",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: float = 60.0,
        cache_size: int = 50000,
        analysis_encoder=None,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.llm_vector_mode = llm_vector_mode
        self.cache_size = cache_size
        self._cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.analysis_encoder = analysis_encoder

        if self.llm_vector_mode == "analysis_embedding":
            # if self.analysis_encoder is None:
            #     self.analysis_encoder = TextAnalysisEncoder()
            self.llm_feature_dim = 128  # self.analysis_encoder.output_dim
        else:
            self.llm_feature_dim = None

    def _make_cache_key(self, prompt: str) -> str:
        payload = {
            "backend": "vllm",
            "llm_vector_mode": self.llm_vector_mode,
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
            return HanabiVLLMClient.CACHE_MISS

        self.cache_hits += 1
        value = self._cache.pop(key)
        self._cache[key] = value
        return value

    def _cache_put(self, key: str, value):
        self._cache[key] = value
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def verify(self) -> None:
        """Verify vLLM is reachable."""
        from urllib.request import Request, urlopen
        import time

        url = f"{self.base_url}/models"

        max_retries = 30
        for attempt in range(max_retries):
            try:
                req = Request(url, method="GET")
                with urlopen(req, timeout=10.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                model_ids = [m.get("id") for m in data.get("data", [])]
                print(f"vLLM reachable. Available models: {model_ids}")

                if self.model_name not in model_ids:
                    print(
                        f"Warning: requested model {self.model_name} not listed. "
                        f"This may still be okay if vLLM aliases the served model."
                    )
                return

            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 5
                    print(
                        f"vLLM verify attempt {attempt + 1} failed. "
                        f"Retrying in {wait}s... Error: {e}"
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Could not connect to vLLM after {max_retries} attempts at {url}: {e}"
                    ) from e

    def get_action_from_context(
        self,
        llm_context: Tuple[str, List[Dict], List[int], Dict],
        num_moves: int,
        hint_annotations: Optional[List[str]] = None,
    ) -> Tuple[Optional[int], np.ndarray]:
        if len(llm_context) == 6:
            (
                text_obs,
                condensed_obs,
                legal_moves_dicts,
                legal_move_uids,
                game_info,
                context_hint_annotations,
            ) = llm_context

            if hint_annotations is None:
                hint_annotations = context_hint_annotations

        else:
            (
                text_obs,
                condensed_obs,
                legal_moves_dicts,
                legal_move_uids,
                game_info,
            ) = llm_context

        if not legal_moves_dicts or not legal_move_uids:
            return None, np.zeros(num_moves, dtype=np.float32), {}

        if self.llm_vector_mode == "analysis_embedding":
            prompt = build_hanabi_analysis_prompt(
                text_obs,
                legal_moves_dicts,
                legal_move_uids,
                game_info,
                hint_annotations=hint_annotations,
            )
        else:
            prompt = build_hanabi_prompt(
                text_obs,
                legal_moves_dicts,
                legal_move_uids,
                game_info,
                hint_annotations=hint_annotations,
            )

        if not hasattr(self, "_printed_first_prompt"):
            print("\n" + "=" * 100, flush=True)
            print("[FIRST LLM PROMPT SENT]", flush=True)
            print(prompt[:8000], flush=True)
            print("=" * 100 + "\n", flush=True)
            self._printed_first_prompt = True

        cache_key = self._make_cache_key(prompt)

        cached = self._cache_get(cache_key)

        if cached is not HanabiVLLMClient.CACHE_MISS:
            suggested_uid, llm_vector, scores_by_legal_idx = cached
            return suggested_uid, llm_vector, scores_by_legal_idx

        #         cached_suggested_uid = self._cache_get(cache_key)
        # if cached_suggested_uid is not HanabiVLLMClient.CACHE_MISS:
        #     return cached_suggested_uid, llm_context_to_action_vector(
        #         num_moves,
        #         cached_suggested_uid,
        #     )

        # response = self._generate(prompt)

        # num_legal = len(legal_move_uids)
        # idx = parse_llm_action_response(response, num_legal)

        # if idx is not None:
        #     suggested_uid = legal_move_uids[idx]
        # else:
        #     suggested_uid = None

        # self._cache_put(cache_key, suggested_uid)

        # return suggested_uid, llm_context_to_action_vector(num_moves, suggested_uid)
        response = self._generate(prompt)

        num_legal = len(legal_move_uids)

        if self.llm_vector_mode == "analysis_embedding":
            analysis_text = response.strip()
            if self.analysis_encoder is None:
                import os
                import random
                import time
                from filelock import FileLock

                lock_path = "/data/class/mae93/nperroch/huggingface/bert_tiny_load.lock"
                os.makedirs(os.path.dirname(lock_path), exist_ok=True)

                delay = random.uniform(0, 10)
                print(f"[LLM] sleeping {delay:.2f}s before BERT load", flush=True)
                time.sleep(delay)

                print("[LLM] waiting for BERT load lock", flush=True)
                with FileLock(lock_path, timeout=600):
                    print("[LLM] acquired BERT load lock", flush=True)

                    if self.analysis_encoder is None:
                        print("[LLM] lazy-loading TextAnalysisEncoder", flush=True)
                        self.analysis_encoder = get_text_analysis_encoder()
                        print("[LLM] loaded TextAnalysisEncoder", flush=True)

            llm_vector = self.analysis_encoder.encode(analysis_text)

            suggested_uid = None
            metadata = {"analysis_text": analysis_text}

            self._cache_put(cache_key, (suggested_uid, llm_vector, metadata))

            if not hasattr(self, "_debug_analysis_print_count"):
                self._debug_analysis_print_count = 0

            if self._debug_analysis_print_count < 30:
                print("[LLM ANALYSIS]", analysis_text[:1000], flush=True)
                print("[LLM ANALYSIS VECTOR SHAPE]", llm_vector.shape, flush=True)
                print(
                    "[LLM ANALYSIS VECTOR NORM]",
                    float(np.linalg.norm(llm_vector)),
                    flush=True,
                )
                self._debug_analysis_print_count += 1

            return suggested_uid, llm_vector, metadata

        num_legal = len(legal_move_uids)
        scores_by_legal_idx, best_idx = parse_llm_score_response(response, num_legal)
        if not scores_by_legal_idx and best_idx is not None:
            scores_by_legal_idx = {i: 0.0 for i in range(num_legal)}
            scores_by_legal_idx[best_idx] = 2.0

        if best_idx is not None:
            suggested_uid = legal_move_uids[best_idx]
        else:
            suggested_uid = None

        # llm_vector = llm_context_to_score_vector(
        #     num_moves=num_moves,
        #     legal_move_uids=legal_move_uids,
        #     scores_by_legal_idx=scores_by_legal_idx,
        #     normalize=False,
        # )
        has_positive_score = (
            bool(scores_by_legal_idx) and max(scores_by_legal_idx.values()) >= 1.0
        )

        if self.llm_vector_mode == "one_hot":
            if best_idx is not None:
                suggested_uid = legal_move_uids[best_idx]
                llm_vector = llm_context_to_action_vector(num_moves, suggested_uid)
            else:
                suggested_uid = None
                llm_vector = np.zeros(num_moves, dtype=np.float32)

        elif self.llm_vector_mode == "softmax":
            suggested_uid = legal_move_uids[best_idx] if best_idx is not None else None

            if has_positive_score:
                llm_vector = scores_to_softmax_vector(
                    num_moves=num_moves,
                    legal_move_uids=legal_move_uids,
                    scores_by_legal_idx=scores_by_legal_idx,
                    temperature=1.5,
                )
            else:
                llm_vector = np.zeros(num_moves, dtype=np.float32)

        else:
            raise ValueError(f"Unknown llm_vector_mode: {self.llm_vector_mode}")
        # llm_vector = scores_to_softmax_vector(
        #     num_moves=num_moves,
        #     legal_move_uids=legal_move_uids,
        #     scores_by_legal_idx=scores_by_legal_idx,
        #     temperature=0.7,
        # )

        self._cache_put(cache_key, (suggested_uid, llm_vector, scores_by_legal_idx))

        if not hasattr(self, "_debug_score_print_count"):
            self._debug_score_print_count = 0

        if self._debug_score_print_count < 30:
            print("[LLM RAW RESPONSE]", response[:1000], flush=True)
            print("[LLM PARSED SCORES]", scores_by_legal_idx, flush=True)
            print("[LLM BEST IDX]", best_idx, "UID:", suggested_uid, flush=True)
            self._debug_score_print_count += 1

        return suggested_uid, llm_vector, scores_by_legal_idx

    def _generate(self, prompt: str) -> str:
        from urllib.request import Request, urlopen
        from urllib.error import URLError, HTTPError

        url = f"{self.base_url}/chat/completions"

        if self.llm_vector_mode == "analysis_embedding":
            system_content = (
                "You are a concise Hanabi strategic analyst. "
                "Do not choose an action. Do not output JSON."
            )
        else:
            system_content = (
                "You are a Hanabi move evaluator. "
                "Return JSON only. No prose. No markdown."
            )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_content,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
        }

        body = json.dumps(payload).encode("utf-8")
        req = Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            with urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
        except HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            print(f"[vLLM HTTP ERROR] {e.code}: {error_body}", flush=True)
            return ""
        except URLError as e:
            print(f"[vLLM CONNECTION ERROR] {e}", flush=True)
            return ""
        except Exception as e:
            print(f"[vLLM UNKNOWN ERROR] {type(e).__name__}: {e}", flush=True)
            return ""

        choices = data.get("choices", [])
        if not choices:
            print("[vLLM NO CHOICES]", data, flush=True)
            return ""

        text = choices[0].get("message", {}).get("content", "").strip()

        if not hasattr(self, "_debug_response_count"):
            self._debug_response_count = 0

        if self._debug_response_count < 20:
            print("[vLLM CHAT RAW DATA]", data, flush=True)
            print("[vLLM CHAT TEXT]", repr(text), flush=True)

        self._debug_response_count += 1

        return text

    def _generate_old(self, prompt: str) -> str:
        from urllib.request import Request, urlopen
        from urllib.error import URLError, HTTPError

        url = f"{self.base_url}/completions"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
        }

        body = json.dumps(payload).encode("utf-8")
        req = Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            with urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            print(f"vLLM HTTP error {e.code}: {error_body}")
            return ""
        except URLError as e:
            print(f"vLLM connection error: {e}")
            return ""

        choices = data.get("choices", [])
        if not choices:
            print("vLLM response had no choices:", data)
            return ""

        return choices[0].get("text", "").strip()


class TextAnalysisEncoder:
    def __init__(
        self,
        model_name: str = "prajjwal1/bert-tiny",
        device: str = "cpu",
        max_length: int = 128,
    ):

        self.torch = torch
        self.device = device
        self.max_length = max_length

        print("[BERT] loading tokenizer", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
        )
        print("[BERT] loaded tokenizer", flush=True)

        print("[BERT] loading model", flush=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=True,
        ).to(device)
        print("[BERT] loaded model", flush=True)

        self.model.eval()
        self.output_dim = self.model.config.hidden_size
        print(f"[BERT] output_dim={self.output_dim}", flush=True)

    def encode(self, text: str) -> np.ndarray:
        torch = self.torch

        if text is None:
            text = ""

        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)

            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)

            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            vec = pooled.squeeze(0).detach().cpu().numpy().astype(np.float32)

        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm

        return vec
