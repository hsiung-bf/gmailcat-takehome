#!/usr/bin/env python
"""
Test script to evaluate classification without requiring OpenAI API.
Tests candidate generation (semantic search) phase only.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.email_categorizer.data_managers.message_manager import MessageManager
from src.email_categorizer.data_managers.faiss_manager import FaissManager
from src.email_categorizer.types import Category

def test_candidate_generation():
    """Test how well semantic search finds relevant candidates."""

    print("=" * 70)
    print("Testing Candidate Generation (Semantic Search)")
    print("=" * 70)

    # Initialize managers
    message_manager = MessageManager()
    faiss_manager = FaissManager()

    # Get all messages
    all_messages = message_manager.get_all_messages()
    print(f"\n📧 Total messages in database: {len(all_messages)}")

    # Test categories
    test_categories = [
        Category(
            name="Travel Receipts",
            description="Flight tickets and travel receipts from airlines like Delta, United, Southwest",
            slug="travel-receipts",
            keywords=["flight", "airline", "ticket", "travel", "delta", "united", "southwest"]
        ),
        Category(
            name="Shopping Orders",
            description="Online shopping orders and shipping confirmations from Amazon, Target, Best Buy",
            slug="shopping-orders",
            keywords=["order", "shipping", "amazon", "purchase", "delivery"]
        ),
        Category(
            name="Health Appointments",
            description="Medical appointment reminders and health-related notifications",
            slug="health-appointments",
            keywords=["appointment", "medical", "doctor", "health", "reminder"]
        ),
        Category(
            name="Tech Newsletters",
            description="Technology newsletters and AI research updates from Substack and tech publications",
            slug="tech-newsletters",
            keywords=["newsletter", "ai", "technology", "substack", "research"]
        )
    ]

    for category in test_categories:
        print(f"\n{'=' * 70}")
        print(f"Category: {category.name}")
        print(f"Description: {category.description}")
        print(f"{'=' * 70}")

        # Get semantic search candidates
        candidates = faiss_manager.search_similar(category, limit=20)

        print(f"\n🔍 Top {len(candidates)} candidates by semantic similarity:\n")

        for i, (msg_id, score) in enumerate(candidates[:15], 1):
            messages_dict = message_manager.get_messages_by_ids([msg_id])
            if msg_id in messages_dict:
                msg = messages_dict[msg_id]
                print(f"{i:2d}. Score: {score:.4f}")
                print(f"    From: {msg.sender[:50]}")
                print(f"    Subject: {msg.subject[:60]}")
                print()

        if len(candidates) > 15:
            print(f"    ... and {len(candidates) - 15} more candidates\n")

    print("=" * 70)
    print("✅ Test Complete!")
    print("=" * 70)

if __name__ == "__main__":
    test_candidate_generation()
