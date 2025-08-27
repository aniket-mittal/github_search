# GitHub Repository Search Algorithm

This project extracts and indexes GitHub repositories from Google BigQuery's public dataset to enable efficient repository search.

## Features

- Extracts repositories with >10 stars
- Filters out forks and duplicates
- Collects repository metadata (name, description, language, website, etc.)
- No commit history - only current repository state
- Clean, indexed data for search algorithms

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Google Cloud Setup

1. **Install Google Cloud CLI:**
   ```bash
   # macOS (using Homebrew)
   brew install google-cloud-sdk
   
   # Or download from: https://cloud.google.com/sdk/docs/install
   ```

2. **Authenticate with Google Cloud:**
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

3. **Set your project (optional but recommended):**
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```

### 3. Environment Configuration

Create a `.env` file in the root directory:
```bash
# Optional: Set specific project ID
GOOGLE_CLOUD_PROJECT=your-project-id

# Optional: Set BigQuery dataset location
BIGQUERY_LOCATION=US
```

## Usage

### Extract Repository Data

```bash
python src/extract_repos.py
```

This will:
- Query the BigQuery `bigquery-public-data.github_repos` dataset
- Filter repositories based on criteria (>10 stars, no forks)
- Extract metadata and save to `data/repositories.csv`

### Data Output

The script generates:
- `data/repositories.csv` - Main repository dataset
- `data/extraction_log.txt` - Processing log
- `data/sample_repos.json` - Sample of extracted data for verification

## Data Schema

The extracted data includes:
- `repo_name`: Repository name
- `repo_owner`: Repository owner
- `description`: Repository description
- `language`: Primary programming language
- `stars`: Number of stars
- `forks`: Number of forks
- `created_at`: Creation date
- `updated_at`: Last update date
- `homepage`: Repository homepage URL
- `topics`: Repository topics/tags
- `size`: Repository size in KB
- `has_wiki`: Whether repository has wiki
- `has_issues`: Whether repository has issues enabled
- `has_downloads`: Whether repository has downloads enabled

## Next Steps

After running the extraction:
1. Review the extracted data in `data/repositories.csv`
2. Implement search indexing algorithms
3. Build search API endpoints
4. Create web interface for repository search

## Troubleshooting

### Common Issues

1. **Authentication Error:**
   - Ensure you've run `gcloud auth application-default login`
   - Check that your account has BigQuery access

2. **BigQuery Quota Exceeded:**
   - The public dataset has usage limits
   - Consider running during off-peak hours

3. **Memory Issues:**
   - The script processes data in chunks
   - Adjust chunk size in the script if needed

## License

MIT License - see LICENSE file for details.
