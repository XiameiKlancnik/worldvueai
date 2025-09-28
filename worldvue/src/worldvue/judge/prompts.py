"""Prompts for LLM style judging."""

from typing import Tuple

STYLE_AXES = [
    'one_sidedness',
    'hype',
    'sourcing',
    'fight_vs_fix',
    'certain_vs_caution'
]

AXIS_DESCRIPTIONS = {
    'one_sidedness': {
        'name': 'One-Sidedness',
        'description': 'How much the article presents only one perspective vs multiple viewpoints',
        'low': 'Presents multiple perspectives fairly',
        'high': 'Presents only one viewpoint'
    },
    'hype': {
        'name': 'Sensationalism/Hype',
        'description': 'Use of dramatic language vs measured tone',
        'low': 'Measured, calm tone',
        'high': 'Dramatic, sensationalized language'
    },
    'sourcing': {
        'name': 'Source Quality',
        'description': 'Quality and transparency of sources',
        'low': 'Poor or no sources cited',
        'high': 'High-quality, transparent sources'
    },
    'fight_vs_fix': {
        'name': 'Fight vs Fix',
        'description': 'Focus on conflict vs solutions',
        'low': 'Focus on solutions and cooperation',
        'high': 'Focus on conflict and division'
    },
    'certain_vs_caution': {
        'name': 'Certainty vs Caution',
        'description': 'Presents claims as certain vs acknowledging uncertainty',
        'low': 'Acknowledges uncertainty and nuance',
        'high': 'Presents claims as absolute certainties'
    }
}


def get_judge_prompt(axis: str, text_a: str, text_b: str,
                    cluster_summary: str) -> Tuple[str, str]:
    """
    Generate judge prompts for a specific axis.

    Returns:
        (system_prompt, user_prompt)
    """
    axis_info = AXIS_DESCRIPTIONS[axis]

    system_prompt = f"""You are an expert media analyst with decisive judgment skills.

You will compare two articles on {axis_info['name']}: {axis_info['description']}

Your task: Determine which article scores HIGHER on this axis:
- Low: {axis_info['low']}
- High: {axis_info['high']}

CRITICAL RULES:
1. You MUST choose A or B - ties are NOT allowed
2. Even tiny differences matter - pick the relatively higher one
3. Focus ONLY on writing style, not political stance or accuracy
4. Use the full article content, not just headlines
5. Provide specific evidence quotes that justify your choice

If articles seem equal, look for subtle differences in:
- Word choice intensity
- Sentence structure complexity
- Evidence presentation style
- Tone nuances

RESPOND WITH VALID JSON ONLY - NO OTHER TEXT BEFORE OR AFTER:
{{
  "winner": "A",
  "confidence": 0.8,
  "evidence_a": "Quote from A",
  "evidence_b": "Quote from B"
}}

Do not include explanations, comments, or any text outside the JSON object."""

    user_prompt = f"""Topic: {cluster_summary}

Article A:
{text_a}

Article B:
{text_b}

Which article scores HIGHER on {axis_info['name']}? You must pick A or B."""

    return system_prompt, user_prompt


def get_multi_axis_judge_prompt(text_a: str, text_b: str, cluster_summary: str) -> Tuple[str, str]:
    """
    Generate judge prompts for ALL style axes in a single call.

    Returns:
        (system_prompt, user_prompt)
    """

    # Build concise axis descriptions
    axes_desc = []
    for axis in STYLE_AXES:
        info = AXIS_DESCRIPTIONS[axis]
        axes_desc.append(f"• {axis}: {info['description']} (Low: {info['low']} | High: {info['high']})")

    system_prompt = f"""Expert media analyst - compare articles on 5 style axes simultaneously.

AXES TO EVALUATE:
{chr(10).join(axes_desc)}

CRITICAL RULES:
1. You MUST choose A or B for each axis - NO TIES allowed
2. Focus ONLY on writing style, not political content or accuracy
3. Even small differences matter - pick the relatively higher one
4. If articles are in foreign languages, mentally translate them to English first, then compare styles
5. Provide brief evidence quotes in English (max 10 words each)

For near-equal articles, look for subtle style differences in word choice, tone, evidence presentation.

RESPOND WITH VALID JSON ONLY - NO OTHER TEXT BEFORE OR AFTER:
{{
  "one_sidedness": {{"winner": "A", "confidence": 0.8, "evidence_a": "quote", "evidence_b": "quote"}},
  "hype": {{"winner": "B", "confidence": 0.7, "evidence_a": "quote", "evidence_b": "quote"}},
  "sourcing": {{"winner": "A", "confidence": 0.9, "evidence_a": "quote", "evidence_b": "quote"}},
  "fight_vs_fix": {{"winner": "B", "confidence": 0.6, "evidence_a": "quote", "evidence_b": "quote"}},
  "certain_vs_caution": {{"winner": "A", "confidence": 0.8, "evidence_a": "quote", "evidence_b": "quote"}}
}}

Do not include explanations, comments, or any text outside the JSON object."""

    user_prompt = f"""Topic: {cluster_summary}

A: {text_a[:1500]}

B: {text_b[:1500]}

Compare all 5 axes. Pick A or B for each."""

    return system_prompt, user_prompt


def get_validation_prompt(text: str, check_type: str) -> Tuple[str, str]:
    """
    Generate prompts for validation checks.

    Args:
        text: Article text
        check_type: 'paraphrase' or 'entity_swap'
    """
    if check_type == 'paraphrase':
        system_prompt = """You are checking if two texts are paraphrases of each other.

Output JSON only:
{
  "is_paraphrase": true or false,
  "confidence": 0.5 to 1.0
}"""

        user_prompt = f"""Are these two texts paraphrases of the same article?

Text 1:
{text}

Text 2:
[Paraphrased version would go here]"""

    elif check_type == 'entity_swap':
        system_prompt = """You are checking if entity swaps affect style judgments.

The same article is presented twice, but with different entity names.
Style judgments should be identical regardless of entity names.

Output JSON only:
{
  "styles_match": true or false,
  "confidence": 0.5 to 1.0
}"""

        user_prompt = f"""Compare style (not content) of these versions:

Version 1:
{text}

Version 2:
[Entity-swapped version would go here]"""

    else:
        raise ValueError(f"Unknown check type: {check_type}")

    return system_prompt, user_prompt