"""
Hyperliquid Raw Fills S3 Downloader
-----------------------------------

This script downloads archived raw fill data from the public AWS S3 bucket 
`hl-mainnet-node-data` used by Hyperliquid. These files are organized in 
`hourly/yyyymmdd/` folders by block timestamp.

Key Features:
-------------
1. Automatically lists all available files for each day between START_DATE and END_DATE.
2. Uses boto3's paginator to efficiently list S3 contents.
3. Downloads files in parallel using ThreadPoolExecutor.
4. Skips any files that already exist locally.
5. Displays progress and download summary using tqdm.

Use Case:
---------
You can use this script to bulk download historical on-chain trade data for offline 
processing, OHLCV generation, or orderbook reconstruction.

Requirements:
-------------
- boto3
- tqdm

Set AWS credentials in your environment or via AWS CLI configuration if needed.
Public access to this bucket requires `RequestPayer='requester'`.
"""

import boto3
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from tqdm import tqdm # Import tqdm

# --- Configuration for the first run ---
S3_BUCKET = 'hl-mainnet-node-data'
S3_PREFIX = 'node_fills_by_block/hourly'
LOCAL_DATA_DIR = './hyperliquid_data/raw_fills'
START_DATE = datetime(2025, 3, 22)
END_DATE = datetime(2025, 8, 22)
MAX_WORKERS = 16

def download_s3_object(s3_client, bucket, key, local_path):
    """Downloads a single S3 object and handles the RequestPayer parameter"""
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(
            Bucket=bucket,
            Key=key,
            Filename=local_path,
            ExtraArgs={'RequestPayer': 'requester'}
        )
        # We'll let the progress bar handle success messages, but we return the key for counting
        return key, None
    except Exception as e:
        # Return the key and the error for summary
        return key, e

def main():
    """Main function to list S3 objects and download them in parallel"""
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')

    print("--- Phase 1: Listing all files to download ---")
    all_keys_to_download = []
    current_date = START_DATE
    while current_date <= END_DATE:
        date_str = current_date.strftime('%Y%m%d')
        prefix_for_day = f"{S3_PREFIX}/{date_str}/"
        
        print(f"Listing objects for prefix: {prefix_for_day}", end="") # Use end="" for same-line printing
        
        try:
            keys_found_for_day = []
            pages = paginator.paginate(
                Bucket=S3_BUCKET,
                Prefix=prefix_for_day,
                RequestPayer='requester'
            )
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        keys_found_for_day.append(obj['Key'])
            
            # --- Added: Report files found per day ---
            print(f" -> Found {len(keys_found_for_day)} files.")
            all_keys_to_download.extend(keys_found_for_day)
            
        except Exception as e:
            print(f" -> Could not list objects for {prefix_for_day}: {e}")
            
        current_date += timedelta(days=1)

    print(f"\nTotal files found across all dates: {len(all_keys_to_download)}")

    if not all_keys_to_download:
        print("No files to download. Exiting.")
        return

    print("\n--- Phase 2: Preparing download tasks (checking for existing files) ---")
    tasks_to_submit = []
    skipped_count = 0
    for key in all_keys_to_download:
        local_path = os.path.join(LOCAL_DATA_DIR, key)
        if os.path.exists(local_path):
            skipped_count += 1
            continue
        tasks_to_submit.append((key, local_path))
    
    print(f"Skipping {skipped_count} files that already exist locally.")
    print(f"Preparing to download {len(tasks_to_submit)} new files.")

    if not tasks_to_submit:
        print("No new files to download. Exiting.")
        return

    print("\n--- Phase 3: Downloading files in parallel ---")
    successful_downloads = []
    failed_downloads = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Prepare futures
        futures = {executor.submit(download_s3_object, s3_client, S3_BUCKET, key, path): key for key, path in tasks_to_submit}
        
        # --- Added: tqdm progress bar ---
        for future in tqdm(as_completed(futures), total=len(tasks_to_submit), desc="Downloading"):
            key, error = future.result()
            if error:
                failed_downloads.append((key, error))
            else:
                successful_downloads.append(key)

    # --- Added: Final Summary ---
    print("\n--- Download Complete: Summary ---")
    print(f"Successfully downloaded: {len(successful_downloads)} files")
    print(f"Skipped (already existed): {skipped_count} files")
    print(f"Failed to download: {len(failed_downloads)} files")
    
    if failed_downloads:
        print("\nFailed files:")
        for key, error in failed_downloads:
            print(f" - {key}: {error}")


if __name__ == '__main__':
    main()


"""
Example Usage:
--------------
1. Ensure you have installed the required libraries:

   pip install boto3 tqdm

2. Configure AWS credentials (if needed):

   aws configure

   Note: This bucket requires requester pays, so no billing credentials are needed 
   unless you exceed public access limits.

3. Run the script:

   python s3_hyperliquid_downloader.py

   It will scan all files between START_DATE and END_DATE,
   skip files that are already present locally, and download only the missing ones.

Output:
-------
All downloaded files will be saved under the directory:

   ./hyperliquid_data/raw_fills/node_fills_by_block/hourly/YYYYMMDD/

Each file represents hourly raw fill data for a specific block timestamp.
"""