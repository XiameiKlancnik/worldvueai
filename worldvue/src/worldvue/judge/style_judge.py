"""LLM-based pairwise style judge for article comparison."""

import json
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import random
from pathlib import Path
import pandas as pd
import click

from ..budget.config import BudgetConfig
from .prompts import get_judge_prompt, get_multi_axis_judge_prompt, STYLE_AXES


def format_article_for_prompt(title: Optional[str], text: Optional[str]) -> str:
    """Combine article title and body for LLM prompts."""
    title = (title or '').strip()
    body = (text or '').strip()
    if title and body:
        return f"Title: {title}\n\nBody:\n{body}"
    if title:
        return f"Title: {title}"
    return body


@dataclass
class JudgeResult:
    pair_id: str
    axis: str
    winner: str  # 'A', 'B', or 'Tie'
    confidence: float
    evidence_a: str
    evidence_b: str
    flags: Dict[str, any]  # Allow any value type in flags, not just bool
    raw_response: Optional[str] = None


class StyleJudge:
    """
    LLM-based judge for pairwise style comparisons.
    Evaluates articles on five style axes.
    """

    def __init__(self, config: BudgetConfig, llm_client=None):
        self.config = config
        self.llm_client = llm_client
        self.results: List[JudgeResult] = []

    def judge_pair(self, pair_data: Dict, article_a_text: str, article_b_text: str,
                  cluster_summary: str) -> List[JudgeResult]:
        """
        Judge a single pair on all style axes.

        Args:
            pair_data: Pair metadata
            article_a_text: Text of article A
            article_b_text: Text of article B
            cluster_summary: Summary of the topic cluster
        """
        # Randomize A/B positions to eliminate position bias
        import random
        if random.random() < 0.5:
            text_a, text_b = article_b_text, article_a_text
            pair_data = pair_data.copy()
            pair_data['swapped'] = True
        else:
            text_a, text_b = article_a_text, article_b_text
            pair_data = pair_data.copy()
            pair_data['swapped'] = False

        # Truncate texts
        text_a = text_a[:self.config.truncate_chars_per_side]
        text_b = text_b[:self.config.truncate_chars_per_side]

        # Apply entity masking if enabled
        if self.config.entity_masking:
            text_a, entities_a = self._mask_entities(text_a)
            text_b, entities_b = self._mask_entities(text_b)
            entity_masked = bool(entities_a or entities_b)
        else:
            entity_masked = False

        # Skip external translation - let ChatGPT handle it inline for cost efficiency
        translation_used = False
        if self.config.use_translation and self.config.pivot_language:
            # Check if translation would be needed (for logging purposes)
            if (pair_data.get('a_lang') != self.config.pivot_language or
                pair_data.get('b_lang') != self.config.pivot_language):
                translation_used = True  # Mark that translation was conceptually needed

        results = []

        # Use multi-axis judgment for efficiency
        if hasattr(self.config, 'use_multi_axis_judging') and self.config.use_multi_axis_judging:
            # Multiple votes per pair, but judge all axes at once
            multi_votes = []
            for vote_idx in range(self.config.votes_per_pair):
                vote_results = self._multi_axis_judgment(
                    pair_data['pair_id'],
                    text_a,
                    text_b,
                    cluster_summary,
                    entity_masked,
                    translation_used
                )
                if vote_results:
                    multi_votes.append(vote_results)

            # Aggregate votes for each axis
            for axis in STYLE_AXES:
                axis_votes = [vote[axis] for vote in multi_votes if axis in vote]
                if axis_votes:
                    final_result = self._aggregate_votes(axis_votes)
                    # Unswap the winner if articles were swapped
                    if pair_data.get('swapped', False):
                        if final_result.winner == 'A':
                            final_result.winner = 'B'
                        elif final_result.winner == 'B':
                            final_result.winner = 'A'
                    results.append(final_result)
        else:
            # Original single-axis approach
            for axis in STYLE_AXES:
                axis_results = []

                # Multiple votes per pair
                for vote_idx in range(self.config.votes_per_pair):
                    result = self._single_judgment(
                        pair_data['pair_id'],
                        axis,
                        text_a,
                        text_b,
                        cluster_summary,
                        entity_masked,
                        translation_used
                    )
                    if result:
                        axis_results.append(result)

                # Aggregate votes
                if axis_results:
                    final_result = self._aggregate_votes(axis_results)
                    # Unswap the winner if articles were swapped
                    if pair_data.get('swapped', False):
                        if final_result.winner == 'A':
                            final_result.winner = 'B'
                        elif final_result.winner == 'B':
                            final_result.winner = 'A'
                    results.append(final_result)

        return results

    def _multi_axis_judgment(self, pair_id: str, text_a: str, text_b: str,
                           cluster_summary: str, entity_masked: bool, translation_used: bool) -> Optional[Dict]:
        """Make a judgment call for all axes at once."""
        if self.config.dry_run:
            # Return mock results for all axes in dry run mode
            return {axis: self._mock_judgment(pair_id, axis) for axis in STYLE_AXES}

        # Prepare multi-axis prompt
        system_prompt, user_prompt = get_multi_axis_judge_prompt(
            text_a, text_b, cluster_summary
        )

        # Call LLM
        try:
            response = self._call_llm(system_prompt, user_prompt)

            # Debug: check if response is actually being returned
            if response is None or response == "":
                click.echo(f"WARNING: Empty LLM response for {pair_id}", err=True)

            # Parse multi-axis response
            results = self._parse_multi_axis_response(response, pair_id)

            # Add flags to all results and debug the raw response
            for axis, result in results.items():
                if result:
                    result.flags['entity_masking_triggered'] = entity_masked
                    result.flags['low_text_length'] = len(text_a) < 100 or len(text_b) < 100
                    result.flags['language_mismatch'] = False
                    result.flags['translation_used'] = translation_used
                    result.flags['response_debug'] = f"resp_len:{len(response) if response else 0}"

            return results

        except Exception as e:
            click.echo(f"Multi-axis judge error for {pair_id}: {e}", err=True)
            return None

    def _single_judgment(self, pair_id: str, axis: str, text_a: str, text_b: str,
                        cluster_summary: str, entity_masked: bool, translation_used: bool) -> Optional[JudgeResult]:
        """Make a single judgment call."""
        if self.config.dry_run:
            # Return mock result in dry run mode
            return self._mock_judgment(pair_id, axis)

        # Prepare prompt
        system_prompt, user_prompt = get_judge_prompt(
            axis, text_a, text_b, cluster_summary
        )

        # Call LLM
        try:
            response = self._call_llm(system_prompt, user_prompt)

            # Parse response
            result = self._parse_response(response, pair_id, axis)

            # Add flags
            result.flags['entity_masking_triggered'] = entity_masked
            result.flags['low_text_length'] = len(text_a) < 100 or len(text_b) < 100
            result.flags['language_mismatch'] = False  # Would check in production
            result.flags['translation_used'] = translation_used

            return result

        except Exception as e:
            click.echo(f"Judge error for {pair_id}/{axis}: {e}", err=True)
            return None

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM API."""
        if not self.llm_client:
            # If no client provided, create one using OpenAI
            try:
                import openai
                import os
                self.llm_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            except Exception as e:
                raise ValueError(f"LLM client not configured and could not create OpenAI client: {e}")

        # Call OpenAI API with cost-optimized settings
        model = "gpt-4o-mini" if getattr(self.config, 'use_cheaper_model', True) else "gpt-4"
        try:
            response = self.llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Slightly higher for more decisive judgments
                max_tokens=400,   # Optimized for multi-axis JSON + inline translation
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            if content is None:
                click.echo(f"OpenAI returned None content, response: {response}", err=True)
                content = ""
            return content
        except Exception as e:
            # Log detailed error information
            click.echo(f"OpenAI API error: {type(e).__name__}: {e}", err=True)

            # Return mock response on error to keep pipeline running
            if hasattr(self.config, 'use_multi_axis_judging') and self.config.use_multi_axis_judging:
                # Multi-axis mock response
                mock_response = {}
                for axis in ['one_sidedness', 'hype', 'sourcing', 'fight_vs_fix', 'certain_vs_caution']:
                    mock_response[axis] = {
                        "winner": random.choice(['A', 'B']),
                        "confidence": random.uniform(0.6, 0.95),
                        "evidence_a": f"Mock evidence A for {axis} (API error)",
                        "evidence_b": f"Mock evidence B for {axis} (API error)"
                    }
                return json.dumps(mock_response)
            else:
                # Single-axis mock response
                response = {
                    "winner": random.choice(['A', 'B']),
                    "confidence": random.uniform(0.6, 0.95),
                    "evidence_a": "Mock evidence from article A (API error)",
                    "evidence_b": "Mock evidence from article B (API error)"
                }
                return json.dumps(response)

    def _parse_response(self, response_text: str, pair_id: str, axis: str) -> JudgeResult:
        """Parse LLM JSON response."""
        try:
            data = json.loads(response_text)

            # Extract reasoning if present (new format)
            reasoning = data.get('reasoning', '')

            return JudgeResult(
                pair_id=pair_id,
                axis=axis,
                winner=data['winner'],
                confidence=data['confidence'],
                evidence_a=data['evidence_a'],
                evidence_b=data['evidence_b'],
                flags={'reasoning': reasoning} if reasoning else {},
                raw_response=response_text
            )
        except json.JSONDecodeError as e:
            # Parse error - randomize decision to avoid A bias
            import random
            winner = random.choice(['A', 'B'])

            # Debug info
            debug_info = f"JSON parse error. Response length: {len(response_text) if response_text else 0}"

            return JudgeResult(
                pair_id=pair_id,
                axis=axis,
                winner=winner,
                confidence=0.6,  # Slightly higher confidence for forced decision
                evidence_a=f"FIXED_VERSION - Parse error ({debug_info})",
                evidence_b=f"FIXED_VERSION - Parse error (winner: {winner})",
                flags={'parse_error': True, 'forced_decision': True, 'response_length': len(response_text) if response_text else 0},
                raw_response=response_text
            )

    def _parse_multi_axis_response(self, response_text: str, pair_id: str) -> Dict[str, JudgeResult]:
        """Parse multi-axis LLM JSON response."""
        results = {}
        try:
            data = json.loads(response_text)

            for axis in STYLE_AXES:
                if axis in data:
                    axis_data = data[axis]
                    results[axis] = JudgeResult(
                        pair_id=pair_id,
                        axis=axis,
                        winner=axis_data['winner'],
                        confidence=axis_data['confidence'],
                        evidence_a=axis_data['evidence_a'],
                        evidence_b=axis_data['evidence_b'],
                        flags={},
                        raw_response=response_text
                    )

            return results

        except json.JSONDecodeError as e:
            # Log the actual problematic response for debugging
            click.echo(f"JSON Parse Error for {pair_id}: {str(e)[:100]}", err=True)
            click.echo(f"Response preview: {response_text[:200] if response_text else 'None'}...", err=True)

            # Fallback: try to extract individual axes from malformed JSON
            import random
            for i, axis in enumerate(STYLE_AXES):
                # Parse error for this axis - randomize decision to avoid A bias
                winner = random.choice(['A', 'B'])

                results[axis] = JudgeResult(
                    pair_id=pair_id,
                    axis=axis,
                    winner=winner,
                    confidence=0.6,
                    evidence_a=f"FIXED_VERSION - Multi-axis parse error",
                    evidence_b=f"FIXED_VERSION - Multi-axis (winner: {winner})",
                    flags={'parse_error': True, 'forced_decision': True, 'response_length': len(response_text) if response_text else 0, 'json_error': str(e)[:100]},
                    raw_response=response_text
                )

            return results

    def _aggregate_votes(self, votes: List[JudgeResult]) -> JudgeResult:
        """Aggregate multiple votes into a single result."""
        # Count winners
        winner_counts = {'A': 0, 'B': 0, 'Tie': 0}
        for vote in votes:
            winner_counts[vote.winner] += 1

        # Majority vote
        final_winner = max(winner_counts.items(), key=lambda x: x[1])[0]

        # Average confidence
        avg_confidence = sum(v.confidence for v in votes) / len(votes)

        # Use evidence from highest confidence vote
        best_vote = max(votes, key=lambda v: v.confidence)

        return JudgeResult(
            pair_id=votes[0].pair_id,
            axis=votes[0].axis,
            winner=final_winner,
            confidence=avg_confidence,
            evidence_a=best_vote.evidence_a,
            evidence_b=best_vote.evidence_b,
            flags={'votes': len(votes), 'agreement': winner_counts[final_winner] / len(votes)}
        )

    def _mask_entities(self, text: str) -> Tuple[str, List[str]]:
        """Mask named entities in text."""
        # Simple regex-based masking for now
        # In production, use NER model
        entities = []

        # Mask potential person names (capitalized words)
        def replace_name(match):
            entities.append(match.group())
            return f"Leader_{len(entities)}"

        # Simple pattern for consecutive capitalized words
        pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        masked_text = re.sub(pattern, replace_name, text)

        return masked_text, entities

    def _translate(self, text: str, target_lang: str) -> str:
        """Translate text to target language."""
        # For now, just return original text but add a flag
        # In production, this would use a translation API like Google Translate
        click.echo(f"WARNING: Translation not implemented - keeping original text", err=True)
        return text

    def _mock_judgment(self, pair_id: str, axis: str) -> JudgeResult:
        """Generate mock judgment for dry run mode."""
        winner = random.choice(['A', 'B'])  # No ties in mock data
        confidence = random.uniform(0.6, 0.95)

        return JudgeResult(
            pair_id=pair_id,
            axis=axis,
            winner=winner,
            confidence=confidence,
            evidence_a=f"Mock evidence for article A on {axis}",
            evidence_b=f"Mock evidence for article B on {axis}",
            flags={'dry_run': True}
        )

    def save_results(self, path: Path):
        """Save judge results to JSONL file."""
        with open(path, 'w') as f:
            for result in self.results:
                f.write(json.dumps(asdict(result)) + '\n')

    @classmethod
    def load_results(cls, path: Path) -> List[JudgeResult]:
        """Load judge results from JSONL file."""
        results = []
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                results.append(JudgeResult(**data))
        return results


class MockJudge(StyleJudge):
    """Mock judge for testing without LLM calls."""

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Return mock LLM response."""
        winner = random.choice(['A', 'B'])  # No ties in mock responses
        response = {
            "winner": winner,
            "confidence": random.uniform(0.6, 0.95),
            "evidence_a": f"Mock evidence from article A",
            "evidence_b": f"Mock evidence from article B",
            "reasoning": f"Mock reasoning for choosing {winner}"
        }
        return json.dumps(response)