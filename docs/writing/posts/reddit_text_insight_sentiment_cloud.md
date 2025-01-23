---
draft: false


date: 2025-01-23
slug: Reddit AI Pulse (Cloud)

categories:
  - AI
  - Data Engineering
  - LLMs
  - end-to-end pipeline
  - cloud
  - GCP
  - ML
authors:
  - sulmank
---


# From Reddit to Insights: Building an AI-Powered Data Pipeline with Gemini (Cloud)

## Introduction

!!! abstract "Purpose"
    In this blog post, I document the process of building an AI-driven, cloud data pipeline to automate this task. Using Google’s Gemini AI, the pipeline collects, processes, and synthesizes discussions from AI-related subreddits into structured daily reports. The system is designed to filter out irrelevant or harmful content, ensuring the extracted insights are both meaningful and actionable.

    Check out the [project GitHub repository](https://github.com/SulmanK/reddit_ai_pulse_cloud_public) for the full code and detailed documentation and [Web Application](https://reddit-text-insight-and-sentiment-website-local.vercel.app/).

    <!-- more -->

!!! info "Problem Statement"
    The field of artificial intelligence and machine learning evolves at an unprecedented pace, with new breakthroughs, trends, and discussions emerging daily. Platforms like Reddit host vibrant AI-focused communities that are rich with valuable insights. However, manually monitoring multiple subreddits to extract meaningful information is both time-consuming and inefficient.


### **What You’ll Learn**
- **Leveraging Google Cloud Storage (GCS) for scalable data lake**: How to use GCS for storing raw data, ML artifacts, and daily results.
- **Optimizing data warehousing with BigQuery and dbt**: Best practices for structuring, managing, and transforming data efficiently in BigQuery.
- **Applying AI for sentiment analysis and summarization**: Techniques to extract concise insights from unstructured Reddit discussions using BART and RoBERTa models.
- **Utilizing Google’s Gemini AI for content analysis**: Insights into leveraging advanced AI models to categorize and interpret data.
- **Orchestrating data workflows with Airflow on a single VM**: How to manage complex data pipelines using Airflow in a cloud environment.
- **Implementing cost-effective cloud solutions**: Strategies for managing costs through scheduled VM start/stop, preemptible instances, and resource optimization.


## System Architecture: Modular and Scalable Design

Our pipeline is designed with modularity and scalability in mind, comprising six main layers. Below is a high-level overview of how the components interact:

![Pipeline Architecture Diagram](./img/reddit_ai_pulse_cloud_pipeline.png)

The diagram above illustrates the flow of data through our system, from collection to presentation. Each layer has specific responsibilities and communicates with adjacent layers through well-defined interfaces.

---

### 1. Data Collection Layer
- **Reddit API Integration**:
    - Fetch posts and comments from AI-focused subreddits
    - Store raw data in GCS buckets
- **Text Preprocessing**:
    - Clean and standardize text data
    - Prepare for ML processing
- **BigQuery Integration**: 
    - Load data into staging tables
    - Manage processing state
    - Incremental processing

### 2. Storage Layer
- **Google Cloud Storage (GCS)**:
    - Raw data storage
    - ML model artifacts
    - Daily results
- **BigQuery**:
    - Structured data storage
    - Efficient query processing
    - Cost-optimized table partitioning
- **DBT Transformations**:
    -  Stage posts and comments for further processing.
    -  Clean and standardize data structures.
    -  Manage processing state for each pipeline run.
    -  Automate cleanup upon pipeline completion.

### 3. Processing Layer
- **Text Summarization**: BART for concise summaries
- **Sentiment Analysis**: RoBERTa model
- **Insight Generation**: Google's Gemini AI
- **MLflow Integration**: Track experiments and models

### 4. Orchestration Layer
- **Single VM Deployment**:
    - e2-standard-2 instance
    - Docker container orchestration
    - Automated start/stop
- **Airflow DAGs**:
    - Main pipeline workflow
    - GitHub sync automation
    - Metric collection

### 5. Observability Layer
- **Cloud Monitoring**:
    - VM health checks
    - Cost monitoring
    - Alert policies
- **Local Monitoring**:
    - Prometheus metrics
    - Grafana dashboards
    - StatsD integration

### 6. Cost Management Layer
- **VM Lifecycle**:
    - Scheduled start/stop
    - Preemptible instances
    - Resource optimization


---

This modular design ensures adaptability, maintainability, and scalability, enabling seamless interaction between components and the efficient transformation of Reddit data into actionable insights.


### Deep Dive: Key Components

#### Resource Management
Resource management is done using Terraform, which enables consistent, efficient, and automated provisioning of cloud resources across multiple platforms.

##### Resource Creation 
`build_res.sh` is a shell script that builds the resources using Terraform.
```bash
# =============================================================================
# build_res.sh
# -----------------------------------------------------------------------------
# Main deployment script for the Reddit AI Pulse Cloud infrastructure.
# This script sets up and configures all necessary GCP resources for the project.
#
# The script performs the following operations:
# 1. Loads environment configuration
# 2. Generates terraform.tfvars from environment variables
# 3. Sets up Docker and Artifact Registry
# 4. Builds and pushes Cloud Run images
# 5. Initializes and applies Terraform configuration
# 6. Updates Cloud Run URLs in .env file
# 7. Manages service account credentials
# 8. Uploads secrets to GitHub
# 9. Configures Git repository settings
#
# The script includes error handling, logging, and automatic backup
# of sensitive files. It can optionally commit changes to Git.
#
# Usage:
#   ./build_res.sh
#
# Requirements:
#   - .env file with required configuration
#   - Authenticated gcloud CLI
#   - Docker installed and configured
#   - GitHub CLI installed and authenticated
#   - Terraform installed
# =============================================================================
```
Once the resources are created, secrets are securely uploaded to Google Cloud Storage (GCS) using GitHub Actions via the `upload-secrets.yml` workflow.

```yml
name: Upload Secrets to GCS

on:
  workflow_dispatch:  # Manual trigger
  repository_dispatch:
    types: [secrets_updated]  # Custom event type
  push:
    paths:
      - '.github/workflows/upload-secrets.yml'

jobs:
  upload-secrets:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write

    steps:
    - uses: actions/checkout@v4

    - id: 'auth'
      name: 'Authenticate to Google Cloud'
      uses: 'google-github-actions/auth@v2'
      with:
        credentials_json: '${{ secrets.GCP_SERVICE_ACCOUNT_KEY }}'

    - name: 'Set up Cloud SDK'
      uses: 'google-github-actions/setup-gcloud@v2'

    - name: 'Create secrets directory'
      run: mkdir -p secrets

    - name: 'Create .env file'
      run: |
        cat << EOF > secrets/.env
        # Docker Registry
        DOCKER_REGISTRY=${{ secrets.DOCKER_REGISTRY }}

        # GCP Project Configuration
        GCP_PROJECT_ID=${{ secrets.GCP_PROJECT_ID }}
        GCS_BUCKET_NAME=${{ secrets.GCS_BUCKET_NAME }}
        GCP_REGION=${{ secrets.GCP_REGION }}
        GCP_ZONE=${{ secrets.GCP_ZONE }}

        # Service Account & Authentication
        GOOGLE_APPLICATION_CREDENTIALS=/opt/airflow/credentials/service-account.json
        SA_EMAIL=${{ secrets.SA_EMAIL }}

        # VM Configuration
        VM_INSTANCE_NAME=${{ secrets.VM_INSTANCE_NAME }}
        VM_MACHINE_TYPE=${{ secrets.VM_MACHINE_TYPE }}

        # Network Configuration
        NETWORK_NAME=${{ secrets.NETWORK_NAME }}
        SUBNET_NAME=${{ secrets.SUBNET_NAME }}
        SUBNET_CIDR=${{ secrets.SUBNET_CIDR }}

        # Email Configuration
        ALERT_EMAIL=${{ secrets.ALERT_EMAIL }}
        ALERT_EMAIL_PASSWORD=${{ secrets.ALERT_EMAIL_PASSWORD }}

        # BigQuery Configuration
        BIGQUERY_DATASET_RAW=${{ secrets.BIGQUERY_DATASET_RAW }}
        BIGQUERY_DATASET_PROCESSED=${{ secrets.BIGQUERY_DATASET_PROCESSED }}

        # Reddit API Configuration
        REDDIT_CLIENT_ID=${{ secrets.REDDIT_CLIENT_ID }}
        REDDIT_CLIENT_SECRET=${{ secrets.REDDIT_CLIENT_SECRET }}
        REDDIT_USERNAME=${{ secrets.REDDIT_USERNAME }}
        REDDIT_PASSWORD=${{ secrets.REDDIT_PASSWORD }}
        REDDIT_USER_AGENT=${{ secrets.REDDIT_USER_AGENT }}

        # Gemini API Key
        GOOGLE_GEMINI_API_KEY=${{ secrets.GOOGLE_GEMINI_API_KEY }}

        # GitHub Configuration
        GH_PAT=${{ secrets.GH_PAT }}
        GH_OWNER=${{ secrets.GH_OWNER }}
        GH_REPO=${{ secrets.GH_REPO }}
        GH_WEBSITE_REPO=${{ secrets.GH_WEBSITE_REPO }}
        AUTO_COMMIT=${{ secrets.AUTO_COMMIT }}

        # Grafana credentials
        GF_SECURITY_ADMIN_USER=${{ secrets.GF_SECURITY_ADMIN_USER }}
        GF_SECURITY_ADMIN_PASSWORD=${{ secrets.GF_SECURITY_ADMIN_PASSWORD }}

        # Airflow PostgreSQL credentials
        AIRFLOW_DB_USER=${{ secrets.AIRFLOW_DB_USER }}
        AIRFLOW_DB_PASSWORD=${{ secrets.AIRFLOW_DB_PASSWORD }}
        AIRFLOW_DB_NAME=${{ secrets.AIRFLOW_DB_NAME }}

        # VM Function URLs
        STOP_VM_FUNCTION_URL=${{ secrets.STOP_VM_FUNCTION_URL }}
        START_VM_FUNCTION_URL=${{ secrets.START_VM_FUNCTION_URL }}

        # Airflow configuration
        AIRFLOW_UID=50000
        AIRFLOW_GID=0
        EOF

    - name: 'Create service account key file'
      run: |
        echo '${{ secrets.GCP_SERVICE_ACCOUNT_KEY }}' > secrets/service-account.json

    - name: 'Upload secrets to GCS'
      run: |
        gsutil cp secrets/.env gs://${{ secrets.GCS_BUCKET_NAME }}/secrets/.env
        gsutil cp secrets/service-account.json gs://${{ secrets.GCS_BUCKET_NAME }}/secrets/service-account.json 
```


##### Resource Deletion
`cleanup.sh` is a shell script that deletes the resources using Terraform.
```bash
# =============================================================================
# cleanup.sh
# -----------------------------------------------------------------------------
# A comprehensive cleanup script for the Reddit AI Pulse Cloud infrastructure.
# This script systematically removes all GCP resources created by the project.
#
# The script performs cleanup in the following order:
# 1. IAM (roles and bindings)
# 2. Cloud Scheduler jobs
# 3. Cloud Functions
# 4. GCS buckets and contents
# 5. BigQuery datasets and tables
# 6. Artifact Registry repositories
# 7. Cloud Run services
# 8. Service accounts
# 9. Compute Engine resources
# 10. VPC, subnets, and firewall rules
# 11. Monitoring resources (alerts, dashboards)
# 12. Terraform state
# 13. Log buckets
#
# The script includes error handling, retries for dependent resources,
# and creates detailed logs of the cleanup process in the resource_info directory.
#
# Usage:
#   ./cleanup.sh
#
# Requirements:
#   - GCP project configuration in .env file
#   - Authenticated gcloud CLI
#   - Required permissions to delete resources
# =============================================================================
```

#### Reddit Data Collection and Preprocessing
The foundation of our pipeline is reliable data collection and preprocessing. We utilize Python's Reddit API wrapper (PRAW) to fetch posts and comments from specified subreddits, with immediate text preprocessing for clean data storage.

##### 1. Data Collection with PRAW
The Reddit client implementation focuses on efficient and reliable data collection. Here's how we handle both posts and their comments:

```python
def fetch_and_save_posts(reddit, subreddit_name, db_utils, conn):
    """Fetches new posts from a subreddit since the last processed timestamp."""
    last_processed_utc = get_last_processed_timestamp(conn, subreddit_name)
    current_max_utc = 0
    
    subreddit = reddit.subreddit(subreddit_name)
    for submission in subreddit.hot(limit=20):
        # Skip already processed posts
        if submission.created_utc <= last_processed_utc:
            continue
        
        # Process post data
        post_data = {
            "subreddit": subreddit_name,
            "post_id": submission.id,
            "title": str(submission.title),
            "author": str(submission.author),
            "url": str(submission.url),
            "score": int(submission.score),
            "created_utc": submission.created_utc,
            "comments": []
        }
        
        # Handle comments if they exist
        if submission.num_comments > 0:
            try:
                submission.comments.replace_more(limit=0)  # Expand comment tree
                post_data["comments"] = [
                    {
                        "comment_id": comment.id,
                        "author": str(comment.author),
                        "body": str(comment.body),
                        "created_utc": comment.created_utc,
                    }
                    for comment in submission.comments.list()[:10]  # Top 10 comments
                    if comment.created_utc > last_processed_utc
                ]
            except Exception as e:
                logging.error(f"Error fetching comments for post {submission.id}: {e}")

        # Save to database
        db_utils.insert_raw_post_data(conn, post_data)
        current_max_utc = max(current_max_utc, submission.created_utc)
```

The raw data is stored in GCS as a JSON file.

```python
def save_to_gcs(storage_client: storage.Client, bucket_name: str, 
                subreddit_name: str, posts: List[Dict]) -> None:
    """Saves posts data to Google Cloud Storage."""
    try:
        # Get bucket
        bucket = storage_client.bucket(bucket_name)
        
        # Generate blob name with current timestamp
        batch_timestamp = datetime.utcnow()
        blob_name = get_gcs_blob_name(subreddit_name, batch_timestamp)
        blob = bucket.blob(blob_name)
        
        # Convert to newline-delimited JSON
        ndjson_data = '\n'.join(json.dumps(post) for post in posts)
        blob.upload_from_string(ndjson_data)
        
        logger.info(f"Saved {len(posts)} posts to GCS: {blob_name}")
    except Exception as e:
        logger.error(f"Error saving to GCS: {e}")
        raise
```

Our system is designed for efficient and reliable data processing. We use incremental data collection, fetching only new data based on timestamps. To manage the volume of Reddit data, we focus on the top 10 comments per post, allowing us to process approximately 1,000 comments daily in about 50 minutes. This can be easily scaled by adding more compute resources. We store comments in JSONB format, which provides flexibility for handling semi-structured data.

We've implemented robust error handling with retry mechanisms and transaction management to ensure data consistency. Batch processing is used to improve scalability and efficiency when dealing with large datasets. While stream processing could further enhance scalability, we opted for batch processing for this use case.

JSON was chosen for its ability to handle semi-structured data like Reddit comments, which often vary in format and content. By storing comments as JSON, the system accommodates diverse data structures without rigid schemas, while still allowing efficient querying and indexing for analytics.


##### 2. Data Preprocessing with PySpark
To ensure the quality of our analysis, we implement a series of preprocessing steps for Reddit posts using PySpark. The following code snippet demonstrates how we filter and clean post titles:

###### Posts Processing
The following code snippet demonstrates how we parse, filter, and clean posts:
```python
def preprocess_posts(df):
    """Preprocesses Reddit posts with content filtering and cleaning."""
    # Content filtering patterns
    profanity_pattern = r'(?i)\b(wordword2|word3|word4|word5)\b|' + \
                       r'(?i)\b(w\*\*d1|w\*rd2|w\*rd3|w\*\*d4)\b|' + \
                       r'(?i)w[^\w]?r[^\w]?d1|' + \
                       r'(?i)w[^\w]?r[^\w]?d2'

    # Filter and clean posts
    return df.filter(
        # Basic validation
        (F.col("title").isNotNull()) & 
        (F.col("title") != "") & 
        (F.col("title") != "[deleted]") &
        (F.col("author").isNotNull()) & 
        (F.col("author") != "[deleted]") &
        # Content filtering
        ~F.col("title").rlike(profanity_pattern)
    ).withColumn(
        # Clean and normalize text
        "title", F.regexp_replace("title", r'[\n\r\t]', ' ')
    ).withColumn(
        # Remove multiple spaces
        "title", F.regexp_replace("title", r'\s+', ' ')
    ).dropDuplicates(["post_id"])
```
Our post preprocessing involves several key steps: we filter out offensive content using regular expressions, validate the `title` and `author` fields, clean and normalize the text, and remove duplicate posts. This ensures our analysis is based on clean and relevant data.

###### Comments Processing

Similar to posts, we also preprocess Reddit comments to ensure data quality. The following code snippet demonstrates how we parse, filter, and clean comments stored in JSON format:

```python
def preprocess_comments(df):
    """Preprocesses Reddit comments with content filtering and cleaning."""
    # Parse and explode comments from JSONB
    comments_schema = ArrayType(StructType([
        StructField("body", StringType(), True),
        StructField("author", StringType(), True),
        StructField("comment_id", StringType(), True),
        StructField("created_utc", DoubleType(), True)
    ]))

    return df.withColumn(
        "comments_parsed",
        F.from_json(F.col("comments"), comments_schema)
    ).withColumn(
        "comment", F.explode("comments_parsed")
    ).select(
        "post_id",
        F.col("comment.body").alias("body"),
        F.col("comment.author").alias("author"),
        F.col("comment.comment_id").alias("comment_id"),
        F.to_timestamp(F.col("comment.created_utc")).alias("created_utc")
    ).filter(
        # Remove deleted/empty comments
        (F.col("body").isNotNull()) &
        (F.col("body") != "") &
        (F.col("body") != "[deleted]") &
        (F.col("author") != "[deleted]")
    ).withColumn(
        # Clean and normalize text
        "body", F.regexp_replace("body", r'[\n\r\t]', ' ')
    ).withColumn(
        # Remove multiple spaces
        "body", F.regexp_replace("body", r'\s+', ' ')
    ).dropDuplicates(["comment_id"])
```
Our comment preprocessing involves defining a schema for the JSON data, parsing and exploding the comments, selecting and aliasing relevant fields (including converting the timestamp), filtering out invalid comments, cleaning and normalizing the text, and removing duplicates.

###### Processed Data Schema
The cleaned data is stored in separate tables for posts and comments:

```terraform
# Dynamic processed posts tables for each subreddit
resource "google_bigquery_table" "processed_posts_tables" {
  for_each = toset(var.subreddits)
  
  dataset_id = google_bigquery_dataset.processed_data.dataset_id
  table_id   = "posts_${lower(each.value)}"
  project    = var.project

  deletion_protection = false

  schema = jsonencode([
    {
      name = "post_id",
      type = "STRING",
      mode = "REQUIRED"
    },
    {
      name = "subreddit",
      type = "STRING",
      mode = "REQUIRED"
    },
    {
      name = "title",
      type = "STRING",
      mode = "NULLABLE"
    },
    {
      name = "author",
      type = "STRING",
      mode = "NULLABLE"
    },
    {
      name = "url",
      type = "STRING",
      mode = "NULLABLE"
    },
    {
      name = "score",
      type = "INTEGER",
      mode = "NULLABLE"
    },
    {
      name = "created_utc",
      type = "TIMESTAMP",
      mode = "NULLABLE"
    },
    {
      name = "ingestion_timestamp",
      type = "TIMESTAMP",
      mode = "NULLABLE"
    }
  ])
}

# Dynamic processed comments tables for each subreddit
resource "google_bigquery_table" "processed_comments_tables" {
  for_each = toset(var.subreddits)
  
  dataset_id = google_bigquery_dataset.processed_data.dataset_id
  table_id   = "comments_${lower(each.value)}"
  project    = var.project

  deletion_protection = false

  schema = jsonencode([
    {
      name = "post_id",
      type = "STRING",
      mode = "REQUIRED"
    },
    {
      name = "comment_id",
      type = "STRING",
      mode = "REQUIRED"
    },
    {
      name = "author",
      type = "STRING",
      mode = "NULLABLE"
    },
    {
      name = "body",
      type = "STRING",
      mode = "NULLABLE"
    },
    {
      name = "created_utc",
      type = "TIMESTAMP",
      mode = "NULLABLE"
    }
  ])
}



```

###### Daily Summary Processing
After preprocessing, we generate daily summaries of the Reddit data using PySpark. The following code snippet demonstrates how we load, filter, and join posts and comments to create these summaries:

```python
def generate_daily_summaries(spark: SparkSession, project_id: str, dataset_id: str, subreddits: list, bucket_name: str):
    """
    Generates daily summaries of the processed data using PySpark.
    Implements incremental loading by only processing data since the last processed timestamp.
    """
    current_timestamp = datetime.now()
    last_processed = get_last_processed_timestamp(spark, project_id, dataset_id)
    total_summaries_added = 0
    
    if last_processed:
        logger.info(f"Processing summaries from {last_processed} to {current_timestamp}")
    else:
        logger.info("No previous summaries found. Processing all available data.")

    for subreddit in subreddits:
        logger.info(f"Processing daily summary for subreddit: {subreddit}")

        # Load posts and comments tables from BigQuery
        posts_df = spark.read.format('bigquery') \
            .option('table', f"{project_id}.{dataset_id}.posts_{subreddit.lower()}") \
            .load()

        posts_count = posts_df.count()
        logger.info(f"Found {posts_count} posts for {subreddit}")

        comments_df = spark.read.format('bigquery') \
            .option('table', f"{project_id}.{dataset_id}.comments_{subreddit.lower()}") \
            .load()

        comments_count = comments_df.count()
        logger.info(f"Found {comments_count} comments for {subreddit}")

        # Filter posts and comments based on created_utc timestamp
        if last_processed:
            daily_posts_df = posts_df.filter(
                (F.col("created_utc") > last_processed) &
                (F.col("created_utc") <= current_timestamp)
            )

            daily_comments_df = comments_df.filter(
                (F.col("created_utc") > last_processed) &
                (F.col("created_utc") <= current_timestamp)
            )
        else:
            daily_posts_df = posts_df.filter(F.col("created_utc") <= current_timestamp)
            daily_comments_df = comments_df.filter(F.col("created_utc") <= current_timestamp)

        filtered_posts_count = daily_posts_df.count()
        logger.info(f"Filtered to {filtered_posts_count} posts for processing")

        if filtered_posts_count == 0:
            logger.info(f"No new posts to summarize for {subreddit}")
            continue

        # Join posts and comments on post_id
        joined_df = daily_posts_df.alias("posts").join(
            daily_comments_df.alias("comments"), 
            "post_id", 
            "right"
        )
        
        # Prepare the daily summary
        daily_summary_df = joined_df.select(
            F.monotonically_increasing_id().alias("id"),
            F.col("subreddit"),
            F.col("posts.post_id").alias("post_id"),
            F.col("posts.score").alias("post_score"),
            F.col("posts.url").alias("post_url"),
            F.col("comments.comment_id").alias("comment_id"),
            F.to_date(F.col("comments.created_utc")).alias("summary_date"),
            F.current_timestamp().alias("processed_date"),
            F.lit(True).alias("needs_processing"),
            F.col("posts.title").alias("post_content"),
            F.col("comments.body").alias("comment_body")
        )

        # Filter out rows where required fields are null
        daily_summary_df = daily_summary_df.filter(
            (F.col("comment_body").isNotNull()) & 
            (F.col("comment_id").isNotNull()) &
            (F.col("post_id").isNotNull())
        )

        filtered_summaries_count = daily_summary_df.count()
        logger.info(f"Generated {filtered_summaries_count} summaries before deduplication")

        # Deduplication Logic
        existing_summary_df = spark.read.format('bigquery') \
            .option('table', f"{project_id}.{dataset_id}.daily_summary_data") \
            .load()

        # Perform a left anti-join to get only new records
        new_daily_summary_df = daily_summary_df.join(
            existing_summary_df,
            ["post_id", "comment_id"],  # Using composite key for deduplication
            "left_anti"
        )

        new_summaries_count = new_daily_summary_df.count()
        
        if new_summaries_count > 0:
            # Write the new daily summaries to BigQuery
            new_daily_summary_df.write.format('bigquery') \
                .option('table', f"{project_id}.{dataset_id}.daily_summary_data") \
                .option('temporaryGcsBucket', bucket_name) \
                .mode('append') \
                .save()
            
            total_summaries_added += new_summaries_count
            logger.info(f"Successfully added {new_summaries_count} new summaries for {subreddit}")
        else:
            logger.info(f"No new unique summaries to add for {subreddit}")
```
To create daily summaries, we load new posts and comments for each subreddit. We then join these datasets on post_id, select the necessary fields, and filter out any rows with null values. This function combines post and comment data, preparing it for further analysis.


The daily summary data is stored with the following schema:

```terraform
resource "google_bigquery_table" "daily_summary" {
  dataset_id = google_bigquery_dataset.processed_data.dataset_id
  table_id   = "daily_summary_data"
  project    = var.project

  deletion_protection = false

  schema = jsonencode([
    {
      name = "id",
      type = "INTEGER",
      mode = "REQUIRED"
    },
    {
      name = "subreddit",
      type = "STRING",
      mode = "REQUIRED"
    },
    {
      name = "post_id",
      type = "STRING",
      mode = "REQUIRED"
    },
    {
      name = "post_score",
      type = "INTEGER",
      mode = "NULLABLE"
    },
    {
      name = "post_url",
      type = "STRING",
      mode = "NULLABLE"
    },
    {
      name = "comment_id",
      type = "STRING",
      mode = "NULLABLE"
    },
    {
      name = "summary_date",
      type = "DATE",
      mode = "REQUIRED"
    },
    {
      name = "processed_date",
      type = "TIMESTAMP",
      mode = "REQUIRED"
    },
    {
      name = "needs_processing",
      type = "BOOLEAN",
      mode = "REQUIRED"
    },
    {
      name = "post_content",
      type = "STRING",
      mode = "NULLABLE"
    },
    {
      name = "comment_body",
      type = "STRING",
      mode = "NULLABLE"
    }
  ])
}
```

#### dbt
##### DBT Data Transformations
Our data transformations are managed using dbt, which allows us to structure our project with specific models for each stage of processing. This approach provides a clear, modular, and maintainable data pipeline.

```yaml
# dbt_project.yml configuration
name: 'dbt_reddit_summary_cloud'
version: '1.0.0'

models:
  dbt_reddit_summary_local:
    +materialized: table  # Default materialization
    current_summary_staging:
      +materialized: view  # Staging view
    joined_summary_analysis:
      +materialized: table  # Final analysis
    update_processing_status:
      +materialized: incremental

vars:
  subreddits: ['dataengineering', 'datascience', 'machinelearning', 'claudeai',
   'singularity', 'localllama', 'openai', 'stablediffusion']
```
This dbt project configuration includes a default table materialization, a staging view for intermediate data, a final analysis table, and an incremental model for updating processing status. Additionally, we define a list of subreddits that are processed by our pipeline.


The DBT workflow includes three main transformation models:

1. **Staging Model** (`current_summary_staging`):
```sql
{{
    config(
        materialized='view'
    )
}}

SELECT
    id,
    subreddit,
    post_id,
    post_score,
    post_url,
    comment_id,
    summary_date,
    processed_date,
    post_content,
    comment_body
FROM {{ source('summary_analytics', 'daily_summary_data') }} ds
WHERE comment_body IS NOT NULL
    AND needs_processing = TRUE
```
The current_summary_staging model is a view that selects key fields from the daily summary data, filtering out rows with null comment bodies and those that do not need processing. This view is materialized as a view for real-time filtering.


2. **Analysis Model** (`joined_summary_analysis`):
```sql
{{ config(
    materialized='table',
    unique_key='comment_id',
) }}

WITH validation_cte AS (
    SELECT 
        cs.*,
        ts.comment_summary,
        sa.sentiment_score,
        sa.sentiment_label,
        CASE 
            WHEN LENGTH(cs.comment_body) < 2 THEN 'Too short'
            WHEN LENGTH(cs.comment_body) > 10000 THEN 'Too long'
            ELSE 'Valid'
        END as comment_quality,
        CASE 
            WHEN ts.comment_summary IS NULL OR TRIM(ts.comment_summary) = '' THEN 'Missing'
            WHEN LENGTH(ts.comment_summary) > LENGTH(cs.comment_body) THEN 'Invalid'
            ELSE 'Valid'
        END as summary_quality
    FROM {{ source('summary_analytics', 'current_summary_staging') }} cs
    LEFT JOIN {{ source('summary_analytics', 'text_summary_results') }} ts 
        ON cs.comment_id = ts.comment_id
    LEFT JOIN {{ source('summary_analytics', 'sentiment_analysis_results') }} sa 
        ON cs.comment_id = sa.comment_id
)

SELECT 
    *,
    CURRENT_TIMESTAMP as processed_at,
    'dbt' as processed_by
FROM validation_cte
WHERE comment_quality = 'Valid' 
  AND summary_quality = 'Valid'
```
The joined_summary_analysis model joins the staging data with text summary and sentiment analysis results. It then validates the quality of comments and summaries, selecting only valid rows for further analysis. This model also adds processing metadata.


3. **Processing Status Model** (`update_processing_status`):
```sql
{{ config(
    materialized='incremental',
    unique_key=['comment_id', 'post_id'],
    post_hook=[
        """
        UPDATE {{ source('summary_analytics', 'daily_summary_data') }}
        SET needs_processing = FALSE
        WHERE comment_id IN (
            SELECT comment_id 
            FROM {{ this }}
        );
        """
    ]
) }}

-- Select all records that were attempted to be processed, including failed validations
WITH validation_cte AS (
    SELECT 
        comment_id,
        comment_body,
        post_id
    FROM {{ source('summary_analytics', 'current_summary_staging') }}
),
all_processed_records AS (
    SELECT 
        cs.comment_id,
        cs.post_id,
        v.comment_quality
    FROM {{ source('summary_analytics', 'current_summary_staging') }} cs
    LEFT JOIN (
        SELECT 
            comment_id,
            CASE 
                WHEN LENGTH(comment_body) < 2 THEN 'Too short'
                WHEN LENGTH(comment_body) > 10000 THEN 'Too long'
                ELSE 'Valid'
            END as comment_quality
        FROM validation_cte
    ) v ON cs.comment_id = v.comment_id
)

SELECT 
    comment_id,
    post_id
FROM all_processed_records
```
The update_processing_status model is an incremental model that updates the processing status of comments. It selects all records that were attempted to be processed, including those that failed validation, and uses a post-hook to mark them as processed in the source table. This ensures that comments are not reprocessed.



##### DBT Data Quality Tests
Our project implements comprehensive data quality tests using DBT's testing framework. These tests are defined in our `schema.yml`:

```yaml
version: 2

sources:
  - name: raw_data
    schema: raw_data
    tables:
      - name: raw_dataengineering
        columns: &raw_columns
          - name: post_id
            tests:
              - not_null
              - unique
              - is_text
          - name: subreddit
            tests:
              - not_null
              - is_text
              - accepted_values:
                  values: ["dataengineering", "datascience", "machinelearning", 
                          "claudeai", "singularity", "localllama", "openai", 
                          "stablediffusion"]
          - name: score
            tests:
              - not_null
              - is_int
              - dbt_utils.expression_is_true:
                  expression: "{{ column_name }} >= 0"
          - name: comments
            tests:
              - not_null
              - is_json
```

Our testing strategy includes generic tests for data presence and type validation, custom tests for specific data structures and rules, relationship tests for data integrity, and data quality metrics to monitor data health. These tests are integrated into our CI/CD pipeline for early detection of issues.

#### Text Processing Pipeline
Our cleaned data is analyzed using transformer models for text summarization and sentiment analysis, with MLflow for experiment tracking.

##### 1. Text Summarization
We use a fine-tuned BART model for generating concise summaries of Reddit comments:

```python
def create_summarizer():
    """Initialize the summarization model."""
    model_name = "philschmid/bart-large-cnn-samsum"
    local_model_path = "/models/models--philschmid--bart-large-cnn-samsum/snapshots/e49b3d60d923f12db22bdd363356f1a4c68532ad"

    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        local_model_path,
        torch_dtype=torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    return pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        framework="pt"
    )

def generate_summary(comment_body, summarizer):
    """Generate summary for a given comment."""
    if len(comment_body.split()) <= 59:
        return comment_body 
        
    try:
        summary = summarizer(
            comment_body, 
            max_length=60, 
            min_length=15, 
            do_sample=False, 
            truncation=True
        )
        if summary and isinstance(summary, list) and len(summary) > 0:
            return summary[0]['summary_text']
    except Exception as e:
        logging.error(f"Summarization failed: {e}")
    
    return ""
```

We use a fine-tuned BART-large model for dialogue summarization, which outperforms the standard model on Reddit comments. It's optimized for local machine use with efficient memory management.
##### 2. Sentiment Analysis
We employ a RoBERTa-based model fine-tuned for emotion detection:

```python
def initialize_emotion_analyzer():
    """Initialize the emotion analysis model."""
    model_name = "SamLowe/roberta-base-go_emotions"
    local_model_path = "/models/models--SamLowe--roberta-base-go_emotions/snapshots/58b6c5b44a7a12093f782442969019c7e2982299"

    model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer
    )

def perform_sentiment_analysis(text, emotion_analyzer):
    """Perform sentiment analysis on the given text."""
    try:
        if not text or text.strip() == "":
            return "Neutral", 0.0, "No content"
            
        truncated_text = truncate_text(text, max_length=500)
        emotion_output = emotion_analyzer(truncated_text)
        
        if not emotion_output:
            return "Neutral", 0.0, "Analysis failed"
            
        emotion_label = emotion_output[0]['label']
        emotion_score = emotion_output[0]['score']
        
        # Map emotions to sentiment categories
        sentiment_mapping = {
            'joy': 'Positive',
            'love': 'Positive',
            'admiration': 'Positive',
            'approval': 'Positive',
            'excitement': 'Positive',
            'gratitude': 'Positive',
            'optimism': 'Positive',
            'anger': 'Negative',
            'disappointment': 'Negative',
            'disgust': 'Negative',
            'fear': 'Negative',
            'sadness': 'Negative',
            'confusion': 'Neutral',
            'curiosity': 'Neutral',
            'surprise': 'Neutral'
        }
        
        sentiment_label = sentiment_mapping.get(emotion_label.lower(), 'Neutral')
        return sentiment_label, float(emotion_score), emotion_label
        
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {str(e)}")
        return "Neutral", 0.0, "Error in analysis"
```

Our sentiment analysis component uses a RoBERTa model fine-tuned for emotion detection, which we found to be more effective than standard sentiment analysis for Reddit comments. Key features include smart text truncation, mapping to sentiment categories (Positive, Negative, or Neutral), and comprehensive error handling.


##### 3. Pipeline Integration
Both components are integrated into the data pipeline with MLflow, which we use for experiment tracking, allowing us to monitor model performance and manage different model versions. We also use MLflow for model serving, enabling us to deploy our models.

```python
# MLflow configuration
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("reddit_sentiment_analysis_experiments")

with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_param("model_name", "SamLowe/roberta-base-go_emotions")
    
```



#### Gemini AI Analysis
We use Google's Gemini AI for advanced analysis, generating structured insights from Reddit discussions with a focus on content safety and consistent formatting.

##### 1. Analysis Configuration
The Gemini analysis is guided by a prompt template that instructs the model to filter content for safety, rank threads by score, summarize content, analyze emotions, extract top viewpoints, and embed links. The output is structured with a title, date, description, tags, overall ranking, and detailed thread analysis.

```python
def create_prompt_template():
    """Return the standard prompt template for Gemini analysis."""
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    return f"""
    Analyze the provided text files, which contain Reddit posts and comments.
    **Instructions:**
    1.  **Content Filtering:** Check for harassment, hate speech, or explicit material
    2.  **Ranking:** Rank threads by total "Score"
    3.  **Summarization:** Utilize "Summary" fields for thread overviews
    4.  **Emotional Analysis:** Analyze "Emotion Label" and "Emotion Score"
    5.  **Point of View Extraction:** Extract top 3 viewpoints per thread
    6.  **Links:** Embed URLs in thread titles using Markdown
    
    **Output Format:**
    ---
    title: "{{subreddit_name}} subreddit"
    date: "{current_date}"
    description: "Analysis of top discussions and trends"
    tags: ["tag1", "tag2", "tag3"]
    ---

    # Overall Ranking and Top Discussions
    [Ranked list of threads with scores and summaries]

    # Detailed Analysis by Thread 
    [Thread-by-thread analysis with summaries, emotions, and viewpoints]
    """
```
The Gemini analysis is guided by a prompt template that instructs the model to filter content for safety, rank threads by score, summarize content, analyze emotions, extract top viewpoints, and embed links. The output is structured with a title, date, description, tags, overall ranking, and detailed thread analysis.


##### 2. Data Processing
The system processes data subreddit by subreddit, formatting each post and its comments:

```python
def process_subreddit(model, cur, subreddit, output_dir):
    """Process a single subreddit's data."""
    # Fetch processed data
    cur.execute("""
        SELECT post_id, subreddit, post_score, post_url, comment_id,
               summary_date, post_content, comment_body, comment_summary,
               sentiment_score, sentiment_label
        FROM processed_data.joined_summary_analysis
        WHERE subreddit = %s
        ORDER BY post_score DESC
    """, (subreddit,))
    
    # Format data for analysis
    text_files = "".join(
        format_text_file(row) for row in cur.fetchall()
    )
    
    # Generate insights using Gemini
    response = model.generate_content(final_prompt + text_files)
    
    # Save formatted output
    output_file_path = os.path.join(output_dir, f"llm_{subreddit}.md")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(response.text)
```

##### 3. Output Formatting
The system ensures consistent formatting across all analyses:

```python
def get_formatted_subreddit_name(subreddit: str) -> str:
    """Returns a properly formatted subreddit name for display."""
    subreddit_formats = {
        "claudeai": "ClaudeAI",
        "dataengineering": "Data Engineering",
        "datascience": "Data Science",
        "localllama": "LocalLLaMA",
        "machinelearning": "Machine Learning",
        "openai": "OpenAI",
        "singularity": "Singularity",
        "stablediffusion": "Stable Diffusion"
    }
    return f"{subreddit_formats.get(subreddit.lower(), subreddit)} Subreddit"
```


##### 4. Pipeline Integration
The analysis is integrated into the main pipeline with organized output storage:

```python
def analyze_data():
    """
    Main function to analyze Reddit data using Google's Gemini model.
    
    Pipeline steps:
    1. Set up logging and output directory with current date
    2. Initialize Gemini model
    3. Connect to BigQuery and GCS
    4. Process each subreddit
    5. Clean generated markdown files
    6. Upload results to GCS
    """
    try:
        # 1. Set up output directory with year/month/day structure
        current_date = datetime.now()
        year = current_date.strftime('%Y')
        month = current_date.strftime('%m')
        day = current_date.strftime('%d')
        
        output_dir = os.path.join(CLOUD_DIR, 'results', year, month, day)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory set to {output_dir}")

        # 2. Initialize model
        genai.configure(api_key=GEMINI_CONFIG['GOOGLE_GEMINI_API_KEY'])
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        logger.info("Model loaded")

        # 3. Connect to BigQuery and GCS
        bq_client = bigquery.Client()
        storage_client = storage.Client()
        logger.info("BigQuery and GCS clients initialized")

        # 4. Process subreddits
        for subreddit in SUBREDDITS:
            process_subreddit(model, bq_client, subreddit, output_dir)
            
        # 5. Clean markdown files and upload to GCS
        for filename in os.listdir(output_dir):
            if filename.endswith('.md'):
                local_file_path = os.path.join(output_dir, filename)
                
                # Clean the markdown file
                clean_markdown_file(local_file_path)
                
                # Upload to GCS
                gcs_blob_path = get_gcs_path(year, month, day, filename)
                upload_to_gcs(local_file_path, gcs_blob_path, storage_client)
        
        logger.info("Analysis complete - all files processed and uploaded to GCS")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
```


#### Web Application
Our web application, built with Next.js 15, React 18, TypeScript, and MDX, provides a modern and user-friendly way to explore our insights. The application automatically syncs with our on-premise analysis results and deploys to Vercel using GitHub Actions, with build optimizations for performance.

##### Automated Deployment
The deployment process leverages GitHub Actions:

```yaml
# .github/workflows/deploy.yml
name: Deploy to Vercel

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

env:
  VERCEL_ORG_ID: ${{ secrets.VERCEL_ORG_ID }}
  VERCEL_PROJECT_ID: ${{ secrets.VERCEL_PROJECT_ID }}

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - name: Install dependencies
        run: npm ci
      - name: Install Vercel CLI
        run: npm install --global vercel@latest
      - name: Pull Vercel Environment Information
        run: vercel pull --yes --environment=production --token=${{ secrets.VERCEL_TOKEN }}
      - name: Build Project Artifacts
        run: vercel build --prod --token=${{ secrets.VERCEL_TOKEN }}
      - name: Deploy Project Artifacts to Vercel
        run: vercel deploy --prebuilt --prod --token=${{ secrets.VERCEL_TOKEN }}
```

#### Orchestrator Stack
Our pipeline is orchestrated using Apache Airflow, configured with a robust setup that ensures reliability, scalability, and observability. The orchestration layer manages the entire data pipeline from ingestion to analysis.

##### 1. Airflow Configuration
The Airflow setup is containerized using Docker Compose with several key components:

```yaml
x-common-env: &common-env
  AIRFLOW__CORE__EXECUTOR: CeleryExecutor
  AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://${AIRFLOW_DB_USER}:${AIRFLOW_DB_PASSWORD}@postgres/${AIRFLOW_DB_NAME}
  AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://${AIRFLOW_DB_USER}:${AIRFLOW_DB_PASSWORD}@postgres/${AIRFLOW_DB_NAME}
  AIRFLOW__CELERY__BROKER_URL: redis://:@redis:6379/0
  AIRFLOW__CORE__LOAD_EXAMPLES: false
  AIRFLOW__CORE__DAGS_FOLDER: /opt/airflow/airflow_project/dags
  AIRFLOW__LOGGING__BASE_LOG_FOLDER: /opt/airflow/airflow_project/logs
  AIRFLOW__CORE__PLUGINS_FOLDER: /opt/airflow/airflow_project/plugins
  AIRFLOW__LOGGING__LOGGING_LEVEL: INFO
  AIRFLOW__LOGGING__FAB_LOGGING_LEVEL: WARNING
  AIRFLOW__LOGGING__LOGGING_CONFIG_CLASS: airflow.config_templates.airflow_local_settings.DEFAULT_LOGGING_CONFIG
  AIRFLOW__LOGGING__DELETE_LOCAL_LOGS: "true"
  AIRFLOW__LOGGING__MAX_LOG_AGE_IN_DAYS: 7
  PYTHONPATH: /opt/airflow
  AIRFLOW__METRICS__STATSD_ON: "true"
  AIRFLOW__METRICS__STATSD_HOST: "statsd-exporter"
  AIRFLOW__METRICS__STATSD_PORT: "9125"
  AIRFLOW__METRICS__STATSD_PREFIX: "airflow"
  AIRFLOW__METRICS__STATSD_ALLOW_LIST: "*"
  AIRFLOW__METRICS__METRICS_ALLOW_LIST: "*"
  AIRFLOW__WEBSERVER__EXPOSE_METRICS: 'true'
  AIRFLOW__METRICS__STATSD_INTERVAL: "30"
  GOOGLE_APPLICATION_CREDENTIALS: ${GOOGLE_APPLICATION_CREDENTIALS}
  GH_PAT: ${GH_PAT}
  GH_OWNER: ${GH_OWNER}
  GH_REPO: ${GH_REPO}
  GH_WEBSITE_REPO: ${GH_WEBSITE_REPO}
  GCP_PROJECT_ID: ${GCP_PROJECT_ID}
  GCS_BUCKET_NAME: ${GCS_BUCKET_NAME}
  STOP_VM_FUNCTION_URL: ${STOP_VM_FUNCTION_URL}
  ALERT_EMAIL: ${ALERT_EMAIL}
  AIRFLOW__SMTP__SMTP_HOST: smtp.gmail.com
  AIRFLOW__SMTP__SMTP_USER: ${ALERT_EMAIL}
  AIRFLOW__SMTP__SMTP_PASSWORD: ${ALERT_EMAIL_PASSWORD}
  AIRFLOW__SMTP__SMTP_PORT: 587
  AIRFLOW__SMTP__SMTP_MAIL_FROM: ${ALERT_EMAIL}
  AIRFLOW__SMTP__SMTP_STARTTLS: "true"
  AIRFLOW__SMTP__SMTP_SSL: "false"

x-common-volumes: &common-volumes
  - ..:/opt/airflow
  - ../credentials:/opt/airflow/credentials
  - ../mlflow:/mlflow
  - ../results:/opt/airflow/results
  - ../dbt_reddit_summary_cloud:/opt/airflow/dbt_reddit_summary_cloud

x-airflow-common: &airflow-common
  volumes: *common-volumes
  environment:
    <<: *common-env
  user: "${AIRFLOW_UID:-50000}:0"
```

The Airflow services are deployed as separate containers, each with specific responsibilities:

1. **Airflow Webserver**
   ```yaml
  airflow-webserver:
    image: ${DOCKER_REGISTRY}/reddit-airflow-webserver:latest
    <<: *airflow-common
    depends_on:
      airflow-init:
        condition: service_completed_successfully
    ports:
      - "8080:8080"
    command: airflow webserver
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
      interval: 30s
      timeout: 30s
      retries: 10
      start_period: 60s
    restart: always
   ```
Our Airflow webserver provides the web UI for DAG management and monitoring, handles user authentication and authorization, exposes REST API endpoints, and includes health checks to ensure UI availability.

2. **Airflow Scheduler**
   ```yaml
  airflow-scheduler:
    image: ${DOCKER_REGISTRY}/reddit-airflow-scheduler:latest
    <<: *airflow-common
    depends_on:
      airflow-webserver:
        condition: service_healthy
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    command: airflow scheduler
    healthcheck:
      test: ["CMD-SHELL", 'airflow jobs check --job-type SchedulerJob --hostname "$${HOSTNAME}"']
      interval: 30s
      timeout: 30s
      retries: 10
      start_period: 60s
    restart: always
   ```
The Airflow scheduler monitors and triggers task execution, manages DAG parsing and scheduling, handles task dependencies and queuing, and ensures proper task distribution to workers. This component is crucial for orchestrating our data pipeline.

3. **Airflow Worker**
   ```yaml
  airflow-worker:
    image: ${DOCKER_REGISTRY}/reddit-airflow-worker:latest
    <<: *airflow-common
    depends_on:
      airflow-webserver:
        condition: service_healthy
    command: airflow celery worker
    healthcheck:
      test: ["CMD-SHELL", 'celery --app airflow.providers.celery.executors.celery_executor.app inspect ping -d "celery@$${HOSTNAME}"']
      interval: 30s
      timeout: 30s
      retries: 5
    restart: always
   ```
Our Airflow worker executes the actual tasks, handles ML model inference, manages resource allocation, supports parallel task execution, and is configured for ML workloads with PyTorch and Transformers.

4. **Airflow Init**
   ```yaml
  airflow-init:
    build:
      context: .
      dockerfile: Dockerfile.webserver
    env_file:
      - ../.env
    <<: *airflow-common
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      <<: *common-env
      GIT_PYTHON_REFRESH: quiet
    command: >
      bash -c "
      airflow db init &&
      airflow db upgrade &&
      airflow users create -r Admin -u admin -p admin -e admin@example.com -f Anonymous -l Admin
      "
   ```
The Airflow init service initializes the Airflow database, creates an admin user, performs database migrations, and runs only during the initial setup. This component is essential for setting up the Airflow environment.


##### 2. Pipeline Structure
Our DAG (`reddit_pipeline`) is organized into 27 stages but can be categorized into 6 main sections, each with specific responsibilities and metrics collection:

1. **Data Collection and Preprocessing**
   ```python
   ingest_task = PythonOperator(
       task_id='ingest_and_preprocess',
       python_callable=ingest_preprocess_process,
       provide_context=True
   )
   ```
Data is ingested and preprocessed.
2. **DBT Transformations**
   ```python
   dbt_staging_task = BashOperator(
       task_id='run_dbt_staging',
       bash_command='cd /opt/airflow/dags/dbt_reddit_summary_cloud && dbt run --select current_summary_staging',
       dag=dag
   )
   ```
DBT transformations are run to prepare the data for analysis.
3. **Model Processing**
```python
    
    summarize_metrics_task = PythonOperator(
        task_id='parse_summarize_metrics',
        python_callable=parse_summarize_metrics,
        provide_context=True
    )

    sentiment_task = PythonOperator(
        task_id='run_sentiment_analysis',
        python_callable=sentiment_analysis_process,
        provide_context=True
    )


    gemini_task = PythonOperator(
        task_id='run_gemini',
        python_callable=gemini_analysis_process,
        dag=dag
    )

```
Our analysis pipeline includes text summarization using BART, sentiment analysis with RoBERTa, and advanced analysis with Gemini AI.


4. **Quality Checks**
   ```python
   dbt_test_raw_sources = BashOperator(
       task_id='test_raw_sources',
       bash_command=DBT_TEST_CMD.format(selector='source:raw_data source:processed_data'),
       dag=dag
   )
   ```
DBT tests are run to ensure the data is valid.
5. **Metrics Collection**
    - We monitor our pipeline with dedicated metrics tasks, using a StatsD exporter to send real-time data to Prometheus, and MLflow tracking for model performance.

6. **Shutdown VM using Cloud Function**
    ```python    
        shutdown_vm = PythonOperator(
            task_id='shutdown_vm',
            python_callable=trigger_vm_shutdown,
            provide_context=True,
            trigger_rule='all_done',  # Run even if upstream tasks fail
            retries=0,  # Don't retry on failure since VM will be shutting down
            dag=dag,
            email_on_failure=False,
        )
    ```  
We use a Cloud Function to shutdown the VM after the pipeline is complete.

##### 3. Task Dependencies
The pipeline follows a clear dependency chain:
```python
ingest_task >> ingest_metrics_task >> \
dbt_test_raw_sources >> dbt_test_raw_metrics_task >> \
dbt_staging_task >> dbt_test_staging_models >> \
summarize_task >> sentiment_task >> \
dbt_join_summary_analysis_task >> \
gemini_task >> push_gemini_results_task >> shutdown_vm
```
Our pipeline starts with data ingestion and preprocessing, followed by DBT testing and staging, then text summarization and sentiment analysis, and finally Gemini AI analysis, result pushing and VM shutdown.

##### 4. Integration Points
The orchestrator connects with BigQuery for pipeline data storage, local model deployment for model serving, StatsD and Prometheus for monitoring, and GitHub for version control of results. These connections are essential for the functionality of our pipeline.



##### 5. Cloud Function

###### 5.1. Start VM
The Cloud Function to start the VM is deployed using Cloud Run.
It ensures seamless startup of virtual machines on-demand, optimizing resource utilization.

```python
# Create Flask app
app = Flask(__name__)

def get_gcloud_path():
    """Get the full path to gcloud executable based on OS"""
    if platform.system() == "Windows":
        # Common installation paths for gcloud on Windows
        possible_paths = [
            r"C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
            r"C:\Program Files\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
            os.path.expanduser("~") + r"\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        raise Exception("gcloud not found. Please ensure Google Cloud SDK is installed and in PATH")
    return "gcloud"  # For non-Windows systems, assume it's in PATH

@app.route("/", methods=["POST"])
def start_vm_http():
    """
    HTTP endpoint for Cloud Run to start a Compute Engine instance.
    Accepts POST requests with optional JSON body:
    {'trigger_source': 'scheduler'|'manual'|'dag'}
    """
    return start_vm(request)

def start_vm(request):
    """
    Function to start a Compute Engine instance and execute startup script.
    """
    project = os.environ.get('PROJECT_ID')
    zone = os.environ.get('ZONE')
    instance = os.environ.get('INSTANCE')
    gh_repo = os.environ.get('GH_REPO')

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"Environment: PROJECT_ID={project}, ZONE={zone}, INSTANCE={instance}, GH_REPO={gh_repo}")

    # Parse request data
    trigger_source = "scheduler"
    if request and hasattr(request, 'is_json') and request.is_json:
        data = request.get_json()
        if data and 'trigger_source' in data:
            trigger_source = data.get('trigger_source')
    
    logger.info(f"Start triggered by: {trigger_source}")

    try:
        # Get gcloud path
        gcloud_path = get_gcloud_path()
        logger.info(f"Using gcloud path: {gcloud_path}")

        # Check if instance is running
        status_cmd = [gcloud_path, "compute", "instances", "describe", instance,
                     f"--zone={zone}", f"--project={project}",
                     "--format=get(status)"]
        status = subprocess.run(status_cmd, text=True, capture_output=True)
        
        is_running = status.stdout.strip().upper() == "RUNNING"
        
        if not is_running:
            # Start the instance
            start_cmd = [gcloud_path, "compute", "instances", "start", instance,
                        f"--zone={zone}", f"--project={project}", "--quiet"]
            logger.info(f"Starting instance with command: {' '.join(start_cmd)}")
            process = subprocess.run(start_cmd, text=True, capture_output=True)
            
            if process.returncode != 0:
                error_msg = f"Failed to start instance: {process.stderr}"
                logger.error(error_msg)
                return jsonify({'status': 'error', 'message': error_msg}), 500
            
            # Wait for VM to be fully started
            logger.info("Waiting 90 seconds for VM to be fully ready...")
            time.sleep(90)

        # Execute startup script
        logger.info("Executing startup script...")
        startup_cmd = [gcloud_path, "compute", "ssh", "--quiet", f"airflow@{instance}",
                      f"--zone={zone}", f"--project={project}", 
                      "--command", f"bash /opt/airflow/{gh_repo}/Cloud/infrastructure/terraform/vm_scripts/start_dag.sh"]
        
        process = subprocess.run(startup_cmd, text=True, capture_output=True)
        
        if process.returncode == 0:
            logger.info("Startup script executed successfully")
            logger.info(f"Startup stdout: {process.stdout}")
            msg = f'Successfully started instance and executed startup script'
            return jsonify({'status': 'success', 'message': msg})
        else:
            error_msg = f"Startup script failed: {process.stderr}"
            logger.error(error_msg)
            return jsonify({'status': 'error', 'message': error_msg}), 500

    except Exception as e:
        error_msg = f'Error during startup process: {str(e)}'
        logger.error(error_msg)
        logger.error(f"Full stack trace: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': error_msg}), 500
```
###### 5.2. Stop VM
The Cloud Function to stop the VM is deployed using Cloud Run.
This helps automate shutdown processes, reducing costs by powering down unused resources.

```python
import os
import json
import logging
import subprocess
from flask import Flask, request, jsonify
import platform
import traceback

# Create Flask app
app = Flask(__name__)

def get_gcloud_path():
    """Get the full path to gcloud executable based on OS"""
    if platform.system() == "Windows":
        # Common installation paths for gcloud on Windows
        possible_paths = [
            r"C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
            r"C:\Program Files\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
            os.path.expanduser("~") + r"\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        raise Exception("gcloud not found. Please ensure Google Cloud SDK is installed and in PATH")
    return "gcloud"  # For non-Windows systems, assume it's in PATH

@app.route("/", methods=["POST"])
def stop_vm_http():
    """
    HTTP endpoint for Cloud Run to stop a Compute Engine instance.
    Accepts POST requests with optional JSON body:
    {'trigger_source': 'scheduler'|'manual'|'dag'}
    """
    return stop_vm(request)

def stop_vm(request):
    """
    Function to stop a Compute Engine instance.
    Only runs shutdown script if VM is running.
    
    Args:
        request (flask.Request): The request object
        - If called from manual trigger: expects JSON with {'trigger_source': 'manual'}
        - If called from Scheduler: no specific payload needed
    Returns:
        JSON response with status message
    """
    project = os.environ.get('PROJECT_ID')
    zone = os.environ.get('ZONE')
    instance = os.environ.get('INSTANCE')
    gh_repo = os.environ.get('GH_REPO')

    # Setup detailed logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Log environment variables (excluding any sensitive data)
    logger.info(f"Environment: PROJECT_ID={project}, ZONE={zone}, INSTANCE={instance}, GH_REPO={gh_repo}")

    # Parse request data if present
    trigger_source = "scheduler"
    if request and hasattr(request, 'is_json') and request.is_json:
        data = request.get_json()
        if data and 'trigger_source' in data:
            trigger_source = data.get('trigger_source')
    
    logger.info(f"Shutdown triggered by: {trigger_source}")

    try:
        # Get gcloud path
        gcloud_path = get_gcloud_path()
        logger.info(f"Using gcloud path: {gcloud_path}")

        # Check if instance is running
        status_cmd = [gcloud_path, "compute", "instances", "describe", instance, 
                     f"--zone={zone}", f"--project={project}", 
                     "--format=get(status)"]
        status = subprocess.run(status_cmd, text=True, capture_output=True)
        
        is_running = status.stdout.strip().upper() == "RUNNING"
        
        if not is_running:
            msg = f'Instance {instance} is already stopped (Triggered by: {trigger_source})'
            logger.info(msg)
            return jsonify({'status': 'success', 'message': msg})

        # Execute shutdown script via SSH since VM is running
        logger.info("Executing shutdown script via SSH...")
        shutdown_cmd = f"bash /opt/airflow/{gh_repo}/Cloud/infrastructure/terraform/vm_scripts/shutdown.sh"
        ssh_cmd = [gcloud_path, "compute", "ssh", "--quiet", f"airflow@{instance}", 
                  f"--zone={zone}", f"--project={project}", "--command", shutdown_cmd]
        
        logger.info(f"Executing command: {' '.join(ssh_cmd)}")
        process = subprocess.run(ssh_cmd, text=True, capture_output=True)
        
        if process.returncode != 0:
            logger.error(f"SSH command failed with return code {process.returncode}")
            logger.error(f"SSH stderr: {process.stderr}")
            logger.error(f"SSH stdout: {process.stdout}")
            raise Exception(f"Failed to execute shutdown script: {process.stderr}")
        
        logger.info("Shutdown script executed successfully. Stopping instance...")
        logger.info(f"SSH stdout: {process.stdout}")

        # Stop the instance
        stop_cmd = [gcloud_path, "compute", "instances", "stop", instance, 
                   f"--zone={zone}", f"--project={project}", "--quiet"]
        logger.info(f"Executing stop command: {' '.join(stop_cmd)}")
        process = subprocess.run(stop_cmd, text=True, capture_output=True)
        
        if process.returncode != 0:
            logger.error(f"Stop command failed with return code {process.returncode}")
            logger.error(f"Stop stderr: {process.stderr}")
            logger.error(f"Stop stdout: {process.stdout}")
            raise Exception(f"Failed to stop instance: {process.stderr}")
        
        logger.info(f"Stop command stdout: {process.stdout}")
        msg = f'Successfully executed shutdown script and stopped instance {instance} (Triggered by: {trigger_source})'
        logger.info(msg)
        return jsonify({'status': 'success', 'message': msg})

    except Exception as e:
        error_msg = f'Error during shutdown process: {str(e)}'
        logger.error(error_msg)
        logger.error(f"Full stack trace: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': error_msg}), 500

if __name__ == "__main__":
    # For local testing with Flask
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 9099)), debug=False) 
```

###### 5.3. Cloud Scheduler
Cloud Scheduler is used to trigger the Cloud Function to start and stop the VM daily at 4:00 PM EST and 6:00 PM EST respectively.

```terraform
# Service account for Cloud Scheduler
resource "google_service_account" "scheduler_sa" {
  account_id   = "vm-scheduler-sa"
  display_name = "Service Account for VM Scheduler"

  lifecycle {
    ignore_changes = [
      account_id,
      display_name
    ]
  }
}

# Grant necessary permissions
resource "google_project_iam_member" "scheduler_sa_compute_admin" {
  project = var.project
  role    = "roles/compute.instanceAdmin.v1"
  member  = "serviceAccount:${google_service_account.scheduler_sa.email}"
}

# Cloud Scheduler job for starting VM (4:00 PM EST)
resource "google_cloud_scheduler_job" "start_vm_schedule" {
  name             = "start-vm-daily"
  description      = "Starts the Airflow VM daily at 4:00 PM EST"
  schedule         = "0 16 * * *"  # 16:00 = 4:00 PM
  time_zone        = "America/New_York"
  attempt_deadline = "900s"  # 15 minutes

  http_target {
    http_method = "POST"
    uri         = google_cloud_run_service.start_vm.status[0].url
    
    body = base64encode(jsonencode({
      trigger_source = "scheduler"
    }))

    headers = {
      "Content-Type" = "application/json"
    }
  }
}

# Cloud Scheduler job for stopping VM (6:00 PM EST)
resource "google_cloud_scheduler_job" "stop_vm_schedule" {
  name             = "stop-vm-daily"
  description      = "Stops the Airflow VM daily at 6:00 PM EST (backup)"
  schedule         = "0 18 * * *"  # 18:00 = 6:00 PM
  time_zone        = "America/New_York"
  attempt_deadline = "900s"  # 15 minutes

  http_target {
    http_method = "POST"
    uri         = google_cloud_run_service.stop_vm.status[0].url
    
    body = base64encode(jsonencode({
      trigger_source = "scheduler"
    }))

    headers = {
      "Content-Type" = "application/json"
    }
  }
} 
```


#### Observability Stack
We use Grafana, Prometheus, and MLflow for comprehensive pipeline observability, with each component containerized and integrated through a centralized metrics system.

##### 1. Metrics Collection Architecture
The metrics collection is structured in layers:

1. **Source Metrics** (via StatsD)
   ```yaml
   AIRFLOW__METRICS__STATSD_ON: "true"
   AIRFLOW__METRICS__STATSD_HOST: "statsd-exporter"
   AIRFLOW__METRICS__STATSD_PORT: "9125"
   AIRFLOW__METRICS__STATSD_PREFIX: "airflow"
   ```

2. **Prometheus Targets**
   ```yaml
   scrape_configs:
     - job_name: 'airflow'
       targets: ['statsd-exporter:9102']
     - job_name: 'redis'
       targets: ['redis-exporter:9121']
     - job_name: 'postgres'
       targets: ['postgres-exporter:9187']
     - job_name: 'mlflow'
       targets: ['mlflow:5000']
   ```

##### 2. Task-Specific Metrics
Each pipeline component has dedicated metrics tracking:

1. **Gemini Analysis Metrics**
   ```python
   # From log_gemini_metrics.py
   metrics = {
       'outputs_generated': 0,      # Number of successful analyses
       'subreddits_processed': 0,   # Subreddits completed
       'duration_seconds': 0,       # Processing time
       'attempt': current_attempt   # Retry tracking
   }
   ```

2. **Sentiment Analysis Performance**
   ```python
   # From log_sentiment_analysis_metrics.py
   metrics = {
       'sentiments_processed': 0,   # Comments analyzed
       'duration_seconds': 0,       # Processing duration
       'attempt': current_attempt   # Execution attempts
   }
   ```

3. **Pipeline Processing Volumes**
   ```json
   {
     "targets": [
       {"expr": "reddit_ingest_preprocess_posts_total"},
       {"expr": "reddit_summarize_summaries_added"},
       {"expr": "reddit_sentiment_processed_total"},
       {"expr": "reddit_joined_table_rows"}
     ]
   }
   ```

##### 3. Performance Dashboards
Our Grafana dashboard is designed to visualize processing efficiency (task durations, volumes, success rates, resource utilization), data quality (DBT test results, validation metrics, processing status, error rates), and provide real-time monitoring.

##### 4. MLflow Integration
MLflow's model versioning, experiment tracking, model serving, and artifact management features aid in model selection by allowing us to identify the best-performing models.

##### 5. Alert Configuration
The system monitors critical thresholds for performance, including task duration, error rate spikes, resource exhaustion, and pipeline stalls. It also monitors quality thresholds, including data validation failures, model performance degradation, processing anomalies, and system health issues

##### 6. Cloud Monitoring
We use Cloud Monitoring to monitor the health of our VM and Cloud Run service, and to set up alerts for critical issues.

```terraform
# Cloud Monitoring Dashboard
resource "google_monitoring_dashboard" "airflow_dashboard" {
  dashboard_json = jsonencode({
    displayName = "Reddit Pipeline Dashboard"
    gridLayout = {
      columns = "2"
      widgets = [
        # VM Metrics
        {
          title = "VM CPU Utilization"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"compute.googleapis.com/instance/cpu/utilization\" resource.type=\"gce_instance\""
                }
              }
            }]
          }
        },
        {
          title = "VM Memory Usage"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"compute.googleapis.com/instance/memory/usage\" resource.type=\"gce_instance\""
                }
              }
            }]
          }
        },
        # Airflow Task Metrics
        {
          title = "DAG Success Rate"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/airflow/dag_success_rate\""
                }
              }
            }]
          }
        },
        {
          title = "Task Duration"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/airflow/task_duration\""
                }
              }
            }]
          }
        },
        # Pipeline Metrics
        {
          title = "Records Processed"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/reddit/records_processed\""
                }
              }
            }]
          }
        },
        {
          title = "Processing Errors"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/reddit/processing_errors\""
                }
              }
            }]
          }
        }
      ]
    }
  })
}

# Alert Policies
resource "google_monitoring_alert_policy" "vm_cpu_utilization" {
  display_name = "High CPU Utilization Alert"
  combiner     = "OR"
  conditions {
    display_name = "CPU Usage > 80%"
    condition_threshold {
      filter          = "metric.type=\"compute.googleapis.com/instance/cpu/utilization\" resource.type=\"gce_instance\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.8
      trigger {
        count = 1
      }
    }
  }
  notification_channels = [google_monitoring_notification_channel.email.id]
}

resource "google_monitoring_alert_policy" "airflow_task_failures" {
  display_name = "Airflow Task Failures Alert"
  combiner     = "OR"
  conditions {
    display_name = "Task Failure Rate > 20%"
    condition_threshold {
      filter          = "metric.type=\"${google_monitoring_metric_descriptor.airflow_task_failure_rate.type}\" AND resource.type=\"gce_instance\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.2
      trigger {
        count = 1
      }
    }
  }
  notification_channels = [google_monitoring_notification_channel.email.id]
  enabled = true
  depends_on = [google_monitoring_metric_descriptor.airflow_task_failure_rate]
}

resource "google_monitoring_alert_policy" "pipeline_errors" {
  display_name = "Pipeline Processing Errors Alert"
  combiner     = "OR"
  conditions {
    display_name = "Processing Error Rate"
    condition_threshold {
      filter          = "metric.type=\"${google_monitoring_metric_descriptor.processing_errors.type}\" AND resource.type=\"gce_instance\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 10
      trigger {
        count = 1
      }
    }
  }
  notification_channels = [google_monitoring_notification_channel.email.id]
  enabled = true
  depends_on = [google_monitoring_metric_descriptor.processing_errors]
}

resource "google_monitoring_alert_policy" "vm_automation" {
  display_name = "VM Automation Alert"
  combiner     = "OR"
  conditions {
    display_name = "VM Start/Stop Failures"
    condition_threshold {
      filter          = "metric.type=\"${google_monitoring_metric_descriptor.vm_automation_failures.type}\" AND resource.type=\"gce_instance\""
      duration        = "0s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      trigger {
        count = 1
      }
    }
  }
  notification_channels = [google_monitoring_notification_channel.email.id]
  enabled = true
  depends_on = [google_monitoring_metric_descriptor.vm_automation_failures]
}

# Notification Channels
resource "google_monitoring_notification_channel" "email" {
  display_name = "Email Notification Channel"
  type         = "email"
  labels = {
    email_address = var.alert_email_address
  }
}

# Optional: Slack notification channel
resource "google_monitoring_notification_channel" "slack" {
  count = var.slack_webhook_url != "" ? 1 : 0
  
  display_name = "Slack Notification Channel"
  type         = "slack"
  labels = {
    channel_name = "#airflow-alerts"
  }
  sensitive_labels {
    auth_token = var.slack_webhook_url
  }
}

# First, define the custom metric descriptors
resource "google_monitoring_metric_descriptor" "airflow_task_failure_rate" {
  description = "Rate of Airflow task failures"
  display_name = "Airflow Task Failure Rate"
  type = "custom.googleapis.com/airflow/task_failure_rate"
  metric_kind = "GAUGE"
  value_type = "DOUBLE"
  unit = "1"
  labels {
    key = "task_id"
    value_type = "STRING"
    description = "The ID of the Airflow task"
  }
}

resource "google_monitoring_metric_descriptor" "processing_errors" {
  description = "Number of processing errors in the pipeline"
  display_name = "Pipeline Processing Errors"
  type = "custom.googleapis.com/reddit/processing_errors"
  metric_kind = "GAUGE"
  value_type = "INT64"
  unit = "1"
}

resource "google_monitoring_metric_descriptor" "vm_automation_failures" {
  description = "Number of VM automation failures"
  display_name = "VM Automation Failures"
  type = "custom.googleapis.com/airflow/vm_automation_failures"
  metric_kind = "GAUGE"
  value_type = "INT64"
  unit = "1"
} 
```

## Results and Impact

The pipeline successfully processes a thousand of Reddit posts daily, generates concise and meaningful summaries, identifies key trends and discussions, maintains high accuracy in content filtering, provides real-time insights through a web interface. 


## Future Improvements

1. **Enhanced Analysis**
    - Integration with additional AI models
    - Advanced visualization features
    - Real-time analysis capabilities
2. **System Scalability**
    - Distributed processing implementation
    - Enhanced caching mechanisms
    - API endpoint for analysis access
3. **User Experience**
    - Interactive dashboard development
    - Customizable analysis parameters
    - Trend prediction features

## Conclusion

This project showcases the power of automation in extracting valuable insights from social media using AI and data engineering, while handling real-world challenges. The open-source code is on GitHub, and welcome community contributions.

## Resources and References
- [Project GitHub Repository](https://github.com/SulmanK/reddit_ai_pulse_cloud_public)
- [Web Application](https://reddit-text-insight-and-sentiment-website-local.vercel.app/)



