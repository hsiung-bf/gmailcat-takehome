#!/usr/bin/env python
"""Test classification on new diverse email dataset."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.email_categorizer.data_managers.message_manager import MessageManager
from src.email_categorizer.data_managers.faiss_manager import FaissManager
from src.email_categorizer.types import Category

def test_new_categories():
    """Test with diverse categories on new email dataset."""

    print("=" * 80)
    print("Testing Classification on New Diverse Email Dataset (57 emails)")
    print("=" * 80)

    message_manager = MessageManager()
    faiss_manager = FaissManager()

    all_messages = message_manager.get_all_messages()
    print(f"\n📧 Total messages in database: {len(all_messages)}")

    # Test with different category types
    test_categories = [
        Category(
            name="Job Applications",
            description="Job-related emails including interview invitations, offer letters, recruitment messages, and application updates",
            slug="job-applications",
            keywords=["job", "interview", "offer", "recruiting", "career", "position"]
        ),
        Category(
            name="Banking & Finance",
            description="Financial emails including credit card statements, payment alerts, investment updates, and banking notifications",
            slug="banking-finance",
            keywords=["payment", "statement", "bank", "credit", "transaction", "balance"]
        ),
        Category(
            name="Healthcare",
            description="Medical appointments, prescription notifications, lab results, health insurance, and wellness updates",
            slug="healthcare",
            keywords=["health", "medical", "doctor", "prescription", "appointment", "insurance"]
        ),
        Category(
            name="Online Learning",
            description="Educational content including online courses, assignments, certificates, academic papers, and learning platforms",
            slug="online-learning",
            keywords=["course", "learning", "education", "certificate", "assignment", "lecture"]
        ),
        Category(
            name="Shopping & Deliveries",
            description="E-commerce orders, shipping notifications, product recommendations, and retail store receipts",
            slug="shopping-deliveries",
            keywords=["order", "shipping", "delivery", "purchase", "product", "receipt"]
        ),
        Category(
            name="Streaming & Subscriptions",
            description="Subscription services, streaming platforms, renewal reminders, and membership updates",
            slug="subscriptions",
            keywords=["subscription", "renewal", "streaming", "membership", "plan", "payment"]
        ),
        Category(
            name="Social Events",
            description="Personal invitations, social media notifications, friend requests, and social activity updates",
            slug="social-events",
            keywords=["invite", "party", "event", "friend", "social", "celebration"]
        ),
    ]

    results_summary = []

    for category in test_categories:
        print(f"\n{'=' * 80}")
        print(f"📁 Category: {category.name}")
        print(f"📝 Description: {category.description}")
        print(f"{'=' * 80}")

        # Get semantic search candidates
        candidates = faiss_manager.search_similar(category, limit=15)

        print(f"\n🔍 Top {len(candidates)} candidates:\n")

        matches = []
        for i, (msg_id, score) in enumerate(candidates[:10], 1):
            messages_dict = message_manager.get_messages_by_ids([msg_id])
            if msg_id in messages_dict:
                msg = messages_dict[msg_id]
                matches.append({
                    'score': score,
                    'subject': msg.subject,
                    'sender': msg.sender
                })
                print(f"{i:2d}. [{score:.4f}] {msg.sender[:35]:<35} | {msg.subject[:45]}")

        # Calculate summary stats
        high_conf = len([c for c in candidates if c[1] >= 0.45])
        med_conf = len([c for c in candidates if 0.35 <= c[1] < 0.45])
        low_conf = len([c for c in candidates if c[1] < 0.35])

        results_summary.append({
            'category': category.name,
            'total_candidates': len(candidates),
            'high_confidence': high_conf,
            'medium_confidence': med_conf,
            'low_confidence': low_conf,
            'top_score': candidates[0][1] if candidates else 0,
            'matches': matches[:5]
        })

    # Print summary
    print(f"\n\n{'=' * 80}")
    print("📊 CLASSIFICATION SUMMARY")
    print(f"{'=' * 80}\n")

    for result in results_summary:
        print(f"Category: {result['category']}")
        print(f"  Top Score: {result['top_score']:.4f}")
        print(f"  Candidates: {result['total_candidates']} "
              f"(High: {result['high_confidence']}, "
              f"Med: {result['medium_confidence']}, "
              f"Low: {result['low_confidence']})")
        print(f"  Top Matches:")
        for match in result['matches']:
            print(f"    - [{match['score']:.3f}] {match['subject'][:60]}")
        print()

    print(f"{'=' * 80}")
    print("✅ Classification Test Complete!")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    test_new_categories()
