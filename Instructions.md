# Run ingestion (must be run first)
python -m src.email_categorizer.ingestion.ingestion sample-messages.jsonl

# Run CLI
python -m src.email_categorizer.cli