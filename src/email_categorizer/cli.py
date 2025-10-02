"""
CLI interface for email categorization.

Interactive interface for creating categories and classifying emails.
"""

import os
# Suppress Hugging Face tokenizers parallelism warnings - must be set before any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import click
import sys
import threading
import time
from dotenv import load_dotenv
from .types import Category
from .orchestrator import EmailCategorizationOrchestrator
import traceback

load_dotenv()


def show_main_menu():
    """Display the main menu and get user choice."""
    click.echo("\n" + "="*50)
    click.echo("Supported Actions")
    click.echo("="*50)
    click.echo("1. Create a new category")
    click.echo("2. List all categories")
    click.echo("3. View emails in a category")
    click.echo("4. Delete a category")
    click.echo("5. Delete all categories")
    click.echo("6. Exit")
    click.echo("="*50)
    
    while True:
        choice = click.prompt("Choose an option (1-6)", type=int)
        if 1 <= choice <= 6:
            return choice
        click.echo("❌ Invalid choice. Please enter 1-6.")


def create_category(orchestrator) -> Category:
    """Interactive category creation."""
    click.echo("\n📝 Creating a new category...")
    
    name = click.prompt("Category name")
    description = click.prompt("Category description")
    
    try:
        category = orchestrator.create_category(name, description)
        click.echo(f"\n✅ Created category: {category.name}")
        click.echo(f"🏷️  Slug: {category.slug}")
        return category
    except ValueError as e:
        click.echo(f"❌ Error: {e}")
        return None


def list_categories_interactive(orchestrator):
    """Interactive category listing."""
    click.echo("\n📋 All Categories")
    
    categories = orchestrator.get_all_categories()
    
    if not categories:
        click.echo("No categories found. Create one to get started!")
        return
    
    for i, category in enumerate(categories, 1):
        click.echo(f"\n{i}. {category.name}")
        click.echo(f"   Slug: {category.slug}")
        click.echo(f"   Description: {category.description}")
        
        email_previews = orchestrator.get_emails_in_category(category.slug)
        click.echo(f"   Emails in category: {len(email_previews)}")


def classify_emails_for_category(orchestrator, category=None):
    """Interactive email classification."""
    if category is None:
        # Let user select a category
        categories = orchestrator.get_all_categories()
        
        if not categories:
            click.echo("❌ No categories found. Create one first!")
            return
        
        click.echo("\nAvailable categories:")
        for i, cat in enumerate(categories, 1):
            click.echo(f"{i}. {cat.name} ({cat.slug})")
        
        choice = click.prompt("\nSelect a category number", type=int)
        
        if choice < 1 or choice > len(categories):
            click.echo("❌ Invalid choice")
            return
        
        category = categories[choice - 1]
    
    messages = orchestrator.load_messages()
    
    # Run classification via orchestrator
    click.echo(f"Classifying {len(messages)} emails for '{category.name}'...")
    try:
        results, summary = orchestrator.classify_emails(messages, category)
        
        click.echo(f"\n✅ Classification complete!")
        click.echo(f"📊 Results: {summary['in_category']} in category, {summary['not_in_category']} not in category")
        
        if summary['classification_failure'] > 0:
            click.echo(f"⚠️  {summary['classification_failure']} classifications failed")
        
        # TODO: Show sample results
        
    except Exception as e:
        click.echo(f"\n❌ Classification failed: {e}")
        traceback.print_exc()
        return


def view_emails_in_category(orchestrator):
    """Interactive email viewing."""
    click.echo("\n📧 View emails in a category")
    
    # Get all categories
    categories = orchestrator.get_all_categories()
    
    if not categories:
        click.echo("❌ No categories found. Create one first!")
        return
    
    # Show categories
    click.echo("\nAvailable categories:")
    for i, category in enumerate(categories, 1):
        click.echo(f"{i}. {category.name} ({category.slug})")
    
    # Get user choice
    choice = click.prompt("\nSelect a category number", type=int)
    
    if choice < 1 or choice > len(categories):
        click.echo("❌ Invalid choice")
        return
    
    selected_category = categories[choice - 1]
    
    # Get emails in category
    try:
        email_previews = orchestrator.get_emails_in_category(selected_category.slug)
        
        if not email_previews:
            click.echo(f"\nNo emails found in category '{selected_category.name}'")
            click.echo("Run classification for this category first.")
            return
        
        click.echo(f"\n📬 Found {len(email_previews)} emails in '{selected_category.name}':")
        click.echo("="*50)
        
        for i, preview in enumerate(email_previews[:20], 1):  # Show first 20
            click.echo(f"\n{i}.")
            click.echo(preview)
            click.echo("-"*50)
        
        if len(email_previews) > 20:
            click.echo(f"\n... and {len(email_previews) - 20} more emails")
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")


def delete_category_interactive(orchestrator):
    """Interactive category deletion."""
    click.echo("\n🗑️  Delete a category")
    categories = orchestrator.get_all_categories()
    
    if not categories:
        click.echo("❌ No categories found.")
        return
    
    # Show categories
    click.echo("\nAvailable categories:")
    for i, category in enumerate(categories, 1):
        click.echo(f"{i}. {category.name} ({category.slug})")
    
    # Get user choice
    choice = click.prompt("\nSelect a category number to delete", type=int)
    
    if choice < 1 or choice > len(categories):
        click.echo("❌ Invalid choice")
        return
    
    selected_category = categories[choice - 1]
    
    # Confirm deletion
    if not click.confirm(f"\n⚠️  Are you sure you want to delete '{selected_category.name}'?"):
        click.echo("Deletion cancelled.")
        return
    
    # Delete
    if orchestrator.delete_category(selected_category.slug):
        click.echo(f"✅ Category '{selected_category.name}' deleted successfully!")
    else:
        click.echo(f"❌ Failed to delete category.")


def delete_all_categories_interactive(orchestrator):
    """Interactive deletion of all categories."""
    click.echo("\n🗑️  Delete ALL categories")
    
    categories = orchestrator.get_all_categories()
    
    if not categories:
        click.echo("❌ No categories found.")
        return
    
    click.echo(f"\n⚠️  WARNING: This will delete all categories!")
    
    # Show what will be deleted
    click.echo("\nCategories to be deleted:")
    for i, category in enumerate(categories, 1):
        click.echo(f"  {i}. {category.name}")
    
    # Double confirmation
    if not click.confirm(f"\nAre you sure you want to delete ALL categories?", default=False):
        click.echo("Deletion cancelled.")
        return
    
    # Delete all
    count = orchestrator.delete_all_categories()
    click.echo(f"\n✅ Successfully deleted {count} categories!")


@click.command()
@click.option('--test', is_flag=True, help='Test mode: limit to 5 emails')
def main(test):
    """Interactive email categorization CLI."""
    click.echo("🚀 Welcome to Email Categorizer!")
    
    if test:
        click.echo("🧪 Running in TEST MODE (limited to 5 emails)")
    
    orchestrator = EmailCategorizationOrchestrator(test_mode=test)
    
    # Check if ingestion has been run
    if not orchestrator.has_messages():
        click.echo("⚠️  No processed messages found. Please run ingestion first:")
        click.echo("   python -m src.email_categorizer.ingestion.ingestion sample-messages.jsonl")
        return

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        click.echo("❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        return
    
    messages = orchestrator.load_messages()
    click.echo(f"📧 Found {len(messages)} emails in your inbox")
    
    while True:
        choice = show_main_menu()
        
        if choice == 1:
            category = create_category(orchestrator)
            if category:
                classify_emails_for_category(orchestrator, category)
        elif choice == 2:
            list_categories_interactive(orchestrator)
        elif choice == 3:
            view_emails_in_category(orchestrator)
        elif choice == 4:
            delete_category_interactive(orchestrator)
        elif choice == 5:
            delete_all_categories_interactive(orchestrator)
        elif choice == 6:
            click.echo("\n👋 Goodbye!")
            break
        
        if choice != 6:
            click.pause() # Pause before showing menu again


if __name__ == '__main__':
    main() 