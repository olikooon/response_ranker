"""
Supported datasets - allenai/soda: 1.5M social dialogues grounded in commonsense knowledge (Gordon et al., 2022)
blended_skill_talk - Facebook BST: 7k conversations blending persona, knowledge & empathy (Smith et al., 2020)
hh_rlhf - Anthropic HH-RLHF: human-assistant dialogues used for RLHF training (Bai et al., 2022)

Negative sampling strategy:
  Distractors = gold responses from OTHER conversations in the same split,
  matching the Ubuntu Corpus v2 / DSTC7 evaluation protocol.
"""

import re
import random
from typing import List, Tuple
from data_loader import ConversationExample


def _sample_distractors(all_gold: List[str], own_index: int, n: int) -> List[str]:
    pool = [g for i, g in enumerate(all_gold) if i != own_index]
    return random.sample(pool, min(n, len(pool)))


# ─────────────────────────────────────────────────────────────────────────────
# SODA  (allenai/soda)
# ─────────────────────────────────────────────────────────────────────────────

def load_soda(
    split: str = "test",
    max_examples: int = 300,
    num_distractors: int = 9,
    min_turns: int = 3,
    seed: int = 42,
) -> List[ConversationExample]:
    """
    Load SODA from HuggingFace (allenai/soda).
    Each conversation contributes one example:
      context = all utterances except the last, joined with ' | '
      gold    = last utterance

    """
    from datasets import load_dataset

    random.seed(seed)
    print(f"  Downloading 'allenai/soda' ({split} split) from HuggingFace...")
    ds = load_dataset("allenai/soda", split=split)

    convs = [ex["dialogue"] for ex in ds if len(ex["dialogue"]) >= min_turns]
    random.shuffle(convs)
    if max_examples:
        convs = convs[:max_examples]

    all_gold = [turns[-1].strip() for turns in convs]

    examples = []
    for i, turns in enumerate(convs):
        context = " | ".join(t.strip() for t in turns[:-1])
        gold = turns[-1].strip()
        distractors = _sample_distractors(all_gold, i, num_distractors)
        examples.append(ConversationExample(context=context, gold=gold, distractors=distractors))

    return examples


# ─────────────────────────────────────────────────────────────────────────────
# Blended Skill Talk  (Facebook AI, 2020)
# ─────────────────────────────────────────────────────────────────────────────

def load_blended_skill_talk(
    split: str = "test",
    max_examples: int = 300,
    num_distractors: int = 9,
    seed: int = 42,
) -> List[ConversationExample]:
    """
    Load BlendedSkillTalk from HuggingFace.
    Uses previous_utterance as context and the first free_message as gold.
    """
    from datasets import load_dataset

    random.seed(seed)
    print(f"  Downloading 'blended_skill_talk' ({split} split) from HuggingFace...")
    ds = load_dataset("blended_skill_talk", split=split)

    raw: List[Tuple[List[str], str]] = []
    for ex in ds:
        prev = ex.get("previous_utterance", [])
        free = ex.get("free_messages", [])
        if prev and free:
            raw.append((prev, free[0]))

    random.shuffle(raw)
    if max_examples:
        raw = raw[:max_examples]

    all_gold = [gold.strip() for _, gold in raw]

    examples = []
    for i, (ctx_turns, gold) in enumerate(raw):
        context = " | ".join(t.strip() for t in ctx_turns)
        gold = gold.strip()
        distractors = _sample_distractors(all_gold, i, num_distractors)
        examples.append(ConversationExample(context=context, gold=gold, distractors=distractors))

    return examples


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic HH-RLHF
# ─────────────────────────────────────────────────────────────────────────────

def _parse_hh_turns(text: str) -> List[str]:
    """
    Parse Anthropic HH-RLHF format:
      '\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: ...'
    Returns a flat list of turn strings (alternating H/A).
    """
    parts = re.split(r"\n\nHuman:\s*|\n\nAssistant:\s*", text.strip())
    return [p.strip() for p in parts if p.strip()]


def load_hh_rlhf(
    split: str = "test",
    max_examples: int = 300,
    num_distractors: int = 9,
    min_turns: int = 3,
    seed: int = 42,
) -> List[ConversationExample]:
    """
    Load Anthropic HH-RLHF from HuggingFace.

    Uses the 'chosen' field (preferred response trajectory).
    Parses the H/A turn format, uses all turns except the last as context,
    and the last assistant turn as gold.
    """
    from datasets import load_dataset

    random.seed(seed)
    print(f"  Downloading 'Anthropic/hh-rlhf' ({split} split) from HuggingFace...")
    ds = load_dataset("Anthropic/hh-rlhf", split=split)

    raw: List[Tuple[List[str], str]] = []
    for ex in ds:
        turns = _parse_hh_turns(ex["chosen"])
        if len(turns) < min_turns:
            continue
        # Last turn is always an assistant response in chosen trajectories
        raw.append((turns[:-1], turns[-1]))

    random.shuffle(raw)
    if max_examples:
        raw = raw[:max_examples]

    all_gold = [gold for _, gold in raw]

    examples = []
    for i, (ctx_turns, gold) in enumerate(raw):
        context = " | ".join(ctx_turns)
        distractors = _sample_distractors(all_gold, i, num_distractors)
        examples.append(ConversationExample(context=context, gold=gold, distractors=distractors))

    return examples


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    "soda": load_soda,
    "blended_skill_talk": load_blended_skill_talk,
    "hh_rlhf": load_hh_rlhf,
}

DATASET_INFO = {
    "soda":
        "allenai/soda — 1.5M social dialogues grounded in commonsense knowledge (Kim et al., 2022)",
    "blended_skill_talk":
        "facebook/bst — 7k conversations blending persona, knowledge & empathy (Smith et al., 2020)",
    "hh_rlhf":
        "Anthropic/hh-rlhf — human-assistant dialogues with RLHF preference labels (Bai et al., 2022)",
}


def load_real_dataset(
    name: str,
    split: str = "test",
    max_examples: int = 300,
    num_distractors: int = 9,
    seed: int = 42,
) -> List[ConversationExample]:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name](
        split=split,
        max_examples=max_examples,
        num_distractors=num_distractors,
        seed=seed,
    )
