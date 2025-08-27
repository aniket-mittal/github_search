#!/usr/bin/env python3
"""
GitHub Repository Data Extraction Script

This script extracts repository data from Google BigQuery's public GitHub dataset
based on specific criteria: >10 stars, no forks, unique repositories.
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from google.cloud import bigquery
from google.auth.exceptions import DefaultCredentialsError
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/extraction_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GitHubRepoExtractor:
    """Extracts GitHub repository data from BigQuery."""
    
    def __init__(self):
        """Initialize the extractor with BigQuery client."""
        try:
            # Get project from environment variables
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            location = os.getenv('BIGQUERY_LOCATION', 'US')
            
            # Check for service account key file
            key_file = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if not key_file:
                # Look for the key file in current directory
                key_file = '*.json'
                if os.path.exists(key_file):
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_file
                    logger.info(f"Using service account key: {key_file}")
                else:
                    logger.warning("No service account key found, using default credentials")
            
            if project_id:
                logger.info(f"Using Google Cloud project: {project_id}")
                self.client = bigquery.Client(project=project_id, location=location)
            else:
                logger.warning("No GOOGLE_CLOUD_PROJECT set, using default project")
                self.client = bigquery.Client()
                
            logger.info("BigQuery client initialized successfully")
        except DefaultCredentialsError:
            logger.error("Google Cloud credentials not found. Please set GOOGLE_APPLICATION_CREDENTIALS or run 'gcloud auth application-default login'")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {e}")
            raise
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the GitHub dataset."""
        try:
            dataset_ref = self.client.dataset('github_repos', project='bigquery-public-data')
            dataset = self.client.get_dataset(dataset_ref)
            
            # List available tables
            tables = list(self.client.list_tables(dataset))
            table_names = [table.table_id for table in tables]
            
            logger.info(f"Available tables in github_repos dataset: {table_names}")
            
            return {
                'dataset_id': dataset.dataset_id,
                'project_id': dataset.project,
                'tables': table_names,
                'description': dataset.description
            }
        except Exception as e:
            logger.error(f"Failed to get dataset info: {e}")
            raise
    
    def get_repositories_table_schema(self) -> List[Dict[str, Any]]:
        """Get the schema of the repositories table."""
        try:
            query = """
            SELECT column_name, data_type, is_nullable
            FROM `bigquery-public-data.github_repos.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = 'sample_repos'
            ORDER BY ordinal_position
            """
            
            query_job = self.client.query(query)
            results = query_job.result()
            
            schema = []
            for row in results:
                schema.append({
                    'column_name': row.column_name,
                    'data_type': row.data_type,
                    'is_nullable': row.is_nullable
                })
            
            logger.info(f"Retrieved schema for repositories table: {len(schema)} columns")
            return schema
            
        except Exception as e:
            logger.error(f"Failed to get table schema: {e}")
            raise
    
    def extract_repositories(self, min_views: int = 100, limit: int = None) -> pd.DataFrame:
        """
        Extract repositories with their actual source code content.
        
        Args:
            min_views: Minimum number of views required
            limit: Maximum number of repositories to extract (None for all)
        
        Returns:
            DataFrame with repository data including source code
        """
        try:
            # First get popular repositories from sample_repos
            popular_repos_query = f"""
            SELECT DISTINCT
                repo_name,
                watch_count as views
            FROM `bigquery-public-data.github_repos.sample_repos`
            WHERE 
                watch_count >= {min_views}
                AND repo_name IS NOT NULL
            """
            
            if limit:
                popular_repos_query += f" LIMIT {limit}"
            
            logger.info(f"Getting popular repositories with {min_views}+ views...")
            popular_repos_job = self.client.query(popular_repos_query)
            popular_repos_df = popular_repos_job.to_dataframe()
            
            logger.info(f"Found {len(popular_repos_df)} popular repositories")
            
            # Get ALL source code content for these repositories (in batches to avoid memory overflow)
            logger.info("Extracting ALL source code content...")
            
            # Get unique repository names
            repo_names = popular_repos_df['repo_name'].unique()
            
            # Process repositories in batches to avoid memory issues
            batch_size = 50
            all_content = []
            
            # Create code files directory early
            code_dir = os.path.join('data', 'code_files')
            os.makedirs(code_dir, exist_ok=True)
            logger.info(f"Created code files directory: {code_dir}")
            
            for i in range(0, len(repo_names), batch_size):
                batch = repo_names[i:i + batch_size]
                batch_repos = "', '".join(batch)
                
                content_query = f"""
                SELECT 
                    c.sample_repo_name as repo_name,
                    c.sample_path as path,
                    c.content,
                    c.size,
                    c.binary,
                    c.copies,
                    c.sample_mode,
                    c.sample_symlink_target
                FROM `bigquery-public-data.github_repos.sample_contents` c
                WHERE c.sample_repo_name IN ('{batch_repos}')
                """
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(repo_names) + batch_size - 1)//batch_size} ({len(batch)} repos)")
                content_job = self.client.query(content_query)
                batch_content = content_job.to_dataframe()
                
                if not batch_content.empty:
                    all_content.append(batch_content)
                    logger.info(f"  Batch {i//batch_size + 1}: {len(batch_content)} files")
                    
                    # Save this batch immediately to disk
                    self._save_batch_to_disk(batch_content, code_dir, i//batch_size + 1)
                
                # Small delay to avoid overwhelming BigQuery
                import time
                time.sleep(0.1)
            
            if all_content:
                content_df = pd.concat(all_content, ignore_index=True)
                logger.info(f"Extracted content for {len(content_df)} total files")
                
                # Merge with popular repos to get views
                final_df = content_df.merge(popular_repos_df, on='repo_name', how='inner')
                
                # Clean and process the data
                final_df = self._clean_dataframe(final_df)
                
                return final_df
            else:
                logger.warning("No content found for popular repositories")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to extract repositories: {e}")
            raise
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and process the extracted DataFrame."""
        try:
            # Remove duplicates based on repo_name and path (unique files)
            initial_count = len(df)
            df = df.drop_duplicates(subset=['repo_name', 'path'])
            logger.info(f"Removed {initial_count - len(df)} duplicate files")
            
            # Extract owner from repo_name (format: "owner/repo")
            df['repo_owner'] = df['repo_name'].apply(lambda x: x.split('/')[0] if '/' in str(x) else 'Unknown')
            df['repo_name_clean'] = df['repo_name'].apply(lambda x: x.split('/')[1] if '/' in str(x) else x)
            
            logger.info(f"Extracted owners from repository names")
            
            # Ensure numeric columns are properly typed
            numeric_columns = ['views', 'size', 'copies', 'sample_mode']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Keep all files - no filtering for search engine
            logger.info(f"Keeping all {len(df)} files for comprehensive code search")
            
            # Sort by views descending, then by file size
            if 'views' in df.columns:
                df = df.sort_values(['views', 'size'], ascending=[False, True])
            
            logger.info("DataFrame cleaning completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Failed to clean DataFrame: {e}")
            raise
    
    def _save_batch_to_disk(self, batch_df: pd.DataFrame, code_dir: str, batch_num: int):
        """Save a batch of files to disk immediately."""
        try:
            logger.info(f"  Saving batch {batch_num} files to disk...")
            
            # Group by repository for organized storage
            for repo_name, repo_files in batch_df.groupby('repo_name'):
                # Create safe directory name
                safe_repo_name = repo_name.replace('/', '_').replace('\\', '_')
                repo_dir = os.path.join(code_dir, safe_repo_name)
                os.makedirs(repo_dir, exist_ok=True)
                
                # Save each file
                for _, file_row in repo_files.iterrows():
                    # Create safe filename
                    safe_path = file_row['path'].replace('/', '_').replace('\\', '_')
                    if not safe_path:
                        safe_path = 'root_file'
                    
                    # Add file extension based on path
                    if '.' not in safe_path:
                        safe_path += '.txt'
                    
                    file_path = os.path.join(repo_dir, safe_path)
                    
                    try:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(str(file_row['content']))
                    except Exception as e:
                        logger.warning(f"    Could not save {file_path}: {e}")
                        # Try with different encoding
                        try:
                            with open(file_path, 'w', encoding='latin-1') as f:
                                f.write(str(file_row['content']))
                        except Exception as e2:
                            logger.error(f"    Failed to save {file_path} with any encoding: {e2}")
            
            logger.info(f"  Batch {batch_num} saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save batch {batch_num} to disk: {e}")
    
    def save_data(self, df: pd.DataFrame, output_dir: str = 'data') -> Dict[str, str]:
        """Save the extracted data to various formats."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save as CSV (metadata only, no content)
            csv_path = os.path.join(output_dir, 'repositories.csv')
            metadata_df = df[['repo_name', 'repo_owner', 'repo_name_clean', 'path', 'size', 'views']].copy()
            metadata_df.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(df)} repository metadata to {csv_path}")
            
            # Save sample as JSON for verification
            sample_size = min(100, len(df))
            sample_df = df.head(sample_size)
            json_path = os.path.join(output_dir, 'sample_repos.json')
            
            # Convert to JSON-serializable format
            sample_data = []
            for _, row in sample_df.iterrows():
                sample_data.append({
                    'repo_name': row['repo_name'],
                    'repo_owner': row['repo_owner'],
                    'repo_name_clean': row['repo_name_clean'],
                    'path': row['path'],
                    'content_preview': row['content'][:200] + '...' if len(str(row['content'])) > 200 else row['content'],
                    'size': int(row['size']),
                    'views': int(row['views'])
                })
            
            with open(json_path, 'w') as f:
                json.dump(sample_data, f, indent=2)
            
            logger.info(f"Saved sample data to {json_path}")
            
            # Code files already saved incrementally during processing
            code_dir = os.path.join(output_dir, 'code_files')
            logger.info(f"Code files already saved to {code_dir} during processing")
            
            # Save summary statistics
            summary_path = os.path.join(output_dir, 'extraction_summary.txt')
            with open(summary_path, 'w') as f:
                f.write(f"GitHub Repository Source Code Extraction Summary\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Data Source: BigQuery sample_repos + sample_contents tables (2022 snapshot)\n")
                f.write(f"Total files: {len(df)}\n")
                f.write(f"Unique repositories: {df['repo_name'].nunique()}\n")
                f.write(f"Unique owners: {df['repo_owner'].nunique()}\n")
                f.write(f"Repositories with 100+ views: {len(df[df['views'] >= 100]):,}\n")
                f.write(f"Total views: {df['views'].sum():,}\n")
                f.write(f"Average views: {df['views'].mean():.2f}\n")
                f.write(f"Total code size: {df['size'].sum():,} bytes\n")
                f.write(f"Average file size: {df['size'].mean():.2f} bytes\n")
                f.write(f"Top owners by file count:\n")
                
                # Show top 10 owners
                top_owners = df['repo_owner'].value_counts().head(10)
                for owner, count in top_owners.items():
                    f.write(f"  {owner}: {count} files\n")
                
                f.write(f"\nNote: Source code content extracted from popular repositories\n")
                f.write(f"Code files saved to: {code_dir}\n")
            
            logger.info(f"Saved summary to {summary_path}")
            
            return {
                'csv': csv_path,
                'json': json_path,
                'code_files': code_dir,
                'summary': summary_path
            }
            
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            raise

def main():
    """Main execution function."""
    try:
        logger.info("Starting GitHub repository extraction process")
        
        # Initialize extractor
        extractor = GitHubRepoExtractor()
        
        # Get dataset information
        dataset_info = extractor.get_dataset_info()
        logger.info(f"Dataset: {dataset_info['dataset_id']} in project {dataset_info['project_id']}")
        
        # Schema info not needed for extraction
        logger.info("Proceeding with extraction...")
        
        # Extract repositories
        logger.info("Extracting repositories with 100+ views and their source code...")
        df = extractor.extract_repositories(min_views=100)
        
        # Save data
        output_files = extractor.save_data(df)
        
        logger.info("Extraction completed successfully!")
        logger.info(f"Output files: {list(output_files.values())}")
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"EXTRACTION COMPLETED SUCCESSFULLY!")
        print(f"{'='*50}")
        print(f"Total files extracted: {len(df):,}")
        print(f"Unique repositories: {df['repo_name'].nunique():,}")
        print(f"Unique owners: {df['repo_owner'].nunique():,}")
        print(f"Repositories with 100+ views: {len(df[df['views'] >= 100]):,}")
        print(f"Total views: {df['views'].sum():,}")
        print(f"Average views: {df['views'].mean():.2f}")
        print(f"Total code size: {df['size'].sum():,} bytes")
        print(f"Average file size: {df['size'].mean():.2f} bytes")
        print(f"Output files saved to 'data/' directory")
        print(f"{'='*50}")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        print(f"ERROR: {e}")
        print("Please check the logs in 'data/extraction_log.txt'")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
