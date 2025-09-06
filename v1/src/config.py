"""
Configuration file for GitHub repository extraction.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# BigQuery Configuration
BIGQUERY_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
BIGQUERY_DATASET = 'bigquery-public-data.github_repos'
BIGQUERY_TABLE = 'sample_repos'  # Main table for repository data (2022 snapshot)
BIGQUERY_LOCATION = os.getenv('BIGQUERY_LOCATION', 'US')

# Extraction Parameters
MIN_VIEWS = 100
MAX_REPOS = None  # Set to a number to limit extraction for testing
CHUNK_SIZE = 10000  # Process data in chunks if needed

# Output Configuration
OUTPUT_DIR = 'data'
CSV_FILENAME = 'repositories.csv'
JSON_FILENAME = 'sample_repos.json'
LOG_FILENAME = 'extraction_log.txt'
SUMMARY_FILENAME = 'extraction_summary.txt'

# Repository Filtering
FILTER_FORKS = True  # Exclude forks
FILTER_ARCHIVED = True  # Exclude archived repositories
FILTER_DISABLED = True  # Exclude disabled repositories
FILTER_VISIBILITY = 'public'  # Only public repositories

# Data Schema - Columns to extract
REPOSITORY_COLUMNS = [
    'repo_name',
    'repo_owner', 
    'description',
    'language',
    'stars',
    'forks',
    'created_at',
    'updated_at',
    'homepage',
    'size',
    'has_wiki',
    'has_issues',
    'has_downloads',
    'has_pages',
    'default_branch',
    'license',
    'topics',
    'archived',
    'disabled',
    'visibility'
]

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
