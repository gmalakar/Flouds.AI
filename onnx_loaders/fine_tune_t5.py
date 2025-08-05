# =============================================================================
# File: fine_tune_t5.py
# Date: 2025-08-01
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

#!/usr/bin/env python3
"""
Fine-tune T5 for sentence embeddings using sentence-transformers.
"""

from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader


def fine_tune_t5_embeddings(model_name: str, output_path: str):
    """Fine-tune T5 model for better sentence embeddings."""

    # Load the model
    model = SentenceTransformer(model_name)

    # Create training examples (replace with your data)
    train_examples = [
        InputExample(
            texts=["This is a positive example", "This is also positive"], label=1.0
        ),
        InputExample(texts=["This is negative", "This is also negative"], label=1.0),
        InputExample(texts=["Positive text", "Negative text"], label=0.0),
    ]

    # Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # Define loss function
    train_loss = losses.CosineSimilarityLoss(model)

    # Fine-tune the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=100,
        output_path=output_path,
    )

    print(f"Fine-tuned model saved to: {output_path}")


if __name__ == "__main__":
    fine_tune_t5_embeddings(
        model_name="sentence-transformers/sentence-t5-base",
        output_path="./fine_tuned_t5",
    )
