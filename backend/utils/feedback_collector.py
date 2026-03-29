"""Feedback collector for continuous RL improvement.

Automatically logs model interactions and implicit quality signals to PostgreSQL.
The collected data is later exported as DPO preference pairs for LoRA fine-tuning.
"""
import logging
import json
from datetime import datetime, timezone
from utils.db_connect import get_pool

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Stores interaction data for future DPO training."""

    async def log_interaction(
        self,
        username: str,
        prompt: str,
        response: str,
        node_type: str,
        intent_parse_success: bool = True,
        quality_signal: str = "neutral",
        metadata: dict = None,
    ):
        """Log a successful model interaction.
        
        Args:
            username: The user who triggered the interaction.
            prompt: The full prompt sent to the model.
            response: The model's raw response text.
            node_type: 'reasoning' or 'generation'.
            intent_parse_success: Whether the response was valid JSON (reasoning only).
            quality_signal: 'positive', 'negative', or 'neutral'.
            metadata: Additional context (category, fact, etc.).
        """
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO feedback_logs 
                        (username, prompt, response, node_type, 
                         intent_parse_success, response_quality_signal, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    username,
                    prompt[:4096],  # Truncate very long prompts
                    response[:4096],
                    node_type,
                    intent_parse_success,
                    quality_signal,
                    json.dumps(metadata or {}),
                )
        except Exception as e:
            # Never let feedback logging break the main flow
            logger.warning(f"[Feedback] Failed to log interaction: {e}")

    async def log_implicit_negative(
        self,
        username: str,
        prompt: str,
        response: str,
        node_type: str,
        reason: str,
    ):
        """Log a failed/poor interaction as a negative signal.
        
        Common reasons:
        - 'json_parse_failure': reasoning response wasn't valid JSON
        - 'user_re_asked': user repeated the same question
        - 'user_corrected': user explicitly corrected the bot
        """
        await self.log_interaction(
            username=username,
            prompt=prompt,
            response=response,
            node_type=node_type,
            intent_parse_success=False if reason == "json_parse_failure" else True,
            quality_signal="negative",
            metadata={"failure_reason": reason},
        )

    async def export_dpo_pairs(self, output_path: str, min_interactions: int = 500):
        """Export collected feedback as DPO preference pairs.
        
        Groups by prompt similarity and pairs positive/negative signals
        to create {prompt, chosen, rejected} triples for DPO training.
        
        Args:
            output_path: Path to write JSONL file.
            min_interactions: Minimum interactions required before export.
        
        Returns:
            Number of pairs exported, or 0 if not enough data.
        """
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                # Check if we have enough data
                count = await conn.fetchval("SELECT count(*) FROM feedback_logs")
                if count < min_interactions:
                    logger.info(
                        f"[Feedback] Only {count}/{min_interactions} interactions. "
                        f"Skipping DPO export."
                    )
                    return 0

                # Get positive/negative pairs for reasoning node
                # Positive = JSON parsed successfully, Negative = parse failure
                pairs = await conn.fetch(
                    """
                    WITH positives AS (
                        SELECT prompt, response
                        FROM feedback_logs
                        WHERE response_quality_signal = 'positive'
                           OR (response_quality_signal = 'neutral' 
                               AND intent_parse_success = true)
                        ORDER BY created_at DESC
                        LIMIT 5000
                    ),
                    negatives AS (
                        SELECT prompt, response
                        FROM feedback_logs
                        WHERE response_quality_signal = 'negative'
                        ORDER BY created_at DESC
                        LIMIT 5000
                    )
                    SELECT p.prompt, p.response AS chosen, n.response AS rejected
                    FROM positives p
                    JOIN negatives n ON p.node_type = n.node_type
                    LIMIT 10000
                    """
                )

                pair_count = 0
                with open(output_path, "w", encoding="utf-8") as f:
                    for row in pairs:
                        pair = {
                            "prompt": row["prompt"],
                            "chosen": row["chosen"],
                            "rejected": row["rejected"],
                        }
                        f.write(json.dumps(pair) + "\n")
                        pair_count += 1

                logger.info(f"[Feedback] Exported {pair_count} DPO pairs to {output_path}")
                return pair_count

        except Exception as e:
            logger.error(f"[Feedback] Export failed: {e}")
            return 0


# Singleton instance
feedback_collector = FeedbackCollector()
