---
draft: false


date: 2025-01-08
slug: Reddit AI Pulse (On-Prem)

categories:
  - AI
  - Data Engineering
  - LLMs
  - end-to-end pipeline
  - ML
authors:
  - sulmank
---


# From Reddit to Insights: Building an AI-Powered Data Pipeline with Gemini (On-Prem)

## Introduction

The field of artificial intelligence and machine learning evolves at an unprecedented pace, with new breakthroughs, trends, and discussions emerging daily. Platforms like Reddit host vibrant AI-focused communities that are rich with valuable insights. However, manually monitoring multiple subreddits to extract meaningful information is both time-consuming and inefficient.

In this blog post, I document the process of building an AI-driven, on-premises data pipeline to automate this task. Using Google’s Gemini AI, the pipeline collects, processes, and synthesizes discussions from AI-related subreddits into structured daily reports. The system is designed to filter out irrelevant or harmful content, ensuring the extracted insights are both meaningful and actionable.

### **What You’ll Learn**
- **Designing a robust data pipeline orchestrated with Airflow**: How to automate and orchestrate complex data workflows.  
- **Optimizing data flow with PostgreSQL and dbt**: Best practices for structuring and managing collected data efficiently.  
- **Applying AI for sentiment analysis and summarization**: Techniques to extract concise insights from unstructured Reddit discussions.    
- **Utilizing Google’s Gemini AI for content analysis**: Insights into leveraging advanced AI models to categorize and interpret data.  

Check out the [project GitHub repository](https://github.com/SulmanK/reddit_ai_pulse_local_public) for the full code and detailed documentation and [Web Application](https://reddit-text-insight-and-sentiment-website-local.vercel.app/).


## System Architecture: Modular and Scalable Design

Our pipeline is designed with modularity and scalability in mind, comprising six main layers. Below is a high-level overview of how the components interact:

![Pipeline Architecture Diagram](./img/reddit_ai_pulse_on_prem_pipeline.png)

The diagram above illustrates the flow of data through our system, from collection to presentation. Each layer has specific responsibilities and communicates with adjacent layers through well-defined interfaces.

---

### 1. Data Collection Layer
- **Reddit API Integration**:
    - Fetch posts and comments from AI-focused subreddits.
- **Text Preprocessing and Cleaning**:
    - Separate processing pipelines for posts and comments.
    - Remove special characters and formatting.
- **Content Validation and Filtering**: 
    - Ensure only relevant and high-quality data is processed.
- **Rate Limiting and Error Handling**: 
    - Manage API limits.
- **DBT Transformations**:
    - Stage posts and comments for further processing.
    - Clean and standardize data structures.
    - Manage processing state for each pipeline run.
    - Automate cleanup upon pipeline completion.

---

### 2. Storage Layer
- **PostgreSQL Database**: 
    - Structured data storage with the following tables:
        - Raw data tables for posts and comments.
        - DBT staging tables for processing queues.
        - Processed results tables for analysis outputs.
- **Organized File System**:
    - Store analysis outputs in a structured and accessible manner.
- **Data Versioning and Backup**:
    - Implement robust strategies for data integrity and recovery.

---

### 3. Processing Layer
- **Text Summarization**: Generate concise summaries of discussions.
- **Sentiment Analysis**: Extract emotional context from posts and comments.
- **Gemini AI Integration**: Perform comprehensive content analysis using LLMs.

---

### 4. Orchestration Layer
- **Airflow DAGs**: Manage pipeline workflows efficiently.
- **Scheduled Task Execution**: Automate task triggers based on predefined schedules.
- **Monitoring and Error Handling**: Ensure reliability through automated checks.

---

### 5. Observability Layer
- **Grafana Dashboards**: Provide real-time monitoring of pipeline health and performance.
- **Prometheus Metrics Collection**: Gather and analyze system metrics.
- **MLflow**: Track machine learning experiments and maintain a model registry.

---

### 6. Presentation Layer
- **React-Based Web Application**: Present insights through an interactive user interface.
- **Automated Deployment**: Streamline updates using GitHub Actions.
- **Real-Time Data Synchronization**: Provides updates in batches (daily updates) to ensure users receive timely insights.
- **Interactive Visualizations**: Enable dynamic exploration of AI-related discussions.

---

This modular design ensures adaptability, maintainability, and scalability, enabling seamless interaction between components and the efficient transformation of Reddit data into actionable insights.



### Deep Dive: Key Components

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

The raw data is stored in PostgreSQL with the following schema:

```sql
-- Raw posts table schema
CREATE TABLE raw_data.raw_posts (
    id SERIAL PRIMARY KEY,
    subreddit VARCHAR(50) NOT NULL,
    post_id VARCHAR(20) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    author VARCHAR(100) NOT NULL,
    url TEXT,
    score INTEGER,
    created_utc TIMESTAMP WITH TIME ZONE,
    comments JSONB,  -- Stores comment data as JSON
    processed BOOLEAN DEFAULT FALSE,
    inserted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

Our system is designed for efficient and reliable data processing. We use incremental data collection, fetching only new data based on timestamps. To manage the volume of Reddit data, we focus on the top 10 comments per post, allowing us to process approximately 1,000 comments daily in about 50 minutes. This can be easily scaled by adding more compute resources. We store comments in JSONB format, which provides flexibility for handling semi-structured data.

We've implemented robust error handling with retry mechanisms and transaction management to ensure data consistency. Batch processing is used to improve scalability and efficiency when dealing with large datasets. While stream processing could further enhance scalability, we opted for batch processing for this use case.

JSONB storage was chosen for its ability to handle semi-structured data like Reddit comments, which often vary in format and content. By storing comments as JSONB, the system accommodates diverse data structures without rigid schemas, while still allowing efficient querying and indexing for analytics.


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

Similar to posts, we also preprocess Reddit comments to ensure data quality. The following code snippet demonstrates how we parse, filter, and clean comments stored in JSONB format:

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
Our comment preprocessing involves defining a schema for the JSONB data, parsing and exploding the comments, selecting and aliasing relevant fields (including converting the timestamp), filtering out invalid comments, cleaning and normalizing the text, and removing duplicates.

###### Processed Data Schema
The cleaned data is stored in separate tables for posts and comments:

```sql
-- Processed posts schema
CREATE TABLE processed_data.posts (
    id SERIAL PRIMARY KEY,
    subreddit VARCHAR(50) NOT NULL,
    post_id VARCHAR(20) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    author VARCHAR(100) NOT NULL,
    url TEXT,
    score INTEGER,
    created_utc TIMESTAMP WITH TIME ZONE,
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Processed comments schema
CREATE TABLE processed_data.comments (
    id SERIAL PRIMARY KEY,
    post_id VARCHAR(20) REFERENCES processed_data.posts(post_id),
    comment_id VARCHAR(20) UNIQUE NOT NULL,
    body TEXT NOT NULL,
    author VARCHAR(100) NOT NULL,
    created_utc TIMESTAMP WITH TIME ZONE,
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
);
```

###### Daily Summary Processing
After preprocessing, we generate daily summaries of the Reddit data using PySpark. The following code snippet demonstrates how we load, filter, and join posts and comments to create these summaries:

```python
def generate_daily_summaries(spark):
    """Generates daily summaries of the processed data using PySpark."""
    current_timestamp = datetime.now()
    last_processed = get_last_processed_timestamp(spark)
    total_summaries_added = 0
    
    for subreddit in SUBREDDITS:
        # Load posts and comments tables
        posts_df = spark.read.format("jdbc") \
            .option("url", jdbc_url) \
            .option("dbtable", f"processed_data.posts_{subreddit.lower()}") \
            .load()

        comments_df = spark.read.format("jdbc") \
            .option("url", jdbc_url) \
            .option("dbtable", f"processed_data.comments_{subreddit.lower()}") \
            .load()

        # Filter for new data since last processing
        if last_processed:
            daily_posts_df = posts_df.filter(
                (F.col("created_utc") > last_processed) &
                (F.col("created_utc") <= current_timestamp)
            daily_comments_df = comments_df.filter(
                (F.col("created_utc") > last_processed) &
                (F.col("created_utc") <= current_timestamp)
            )
        else:
            daily_posts_df = posts_df.filter(F.col("created_utc") <= current_timestamp)
            daily_comments_df = comments_df.filter(F.col("created_utc") <= current_timestamp)

        # Join posts and comments
        daily_summary_df = daily_posts_df.alias("posts").join(
            daily_comments_df.alias("comments"), 
            "post_id", 
            "right"
        ).select(
            F.col("subreddit"),
            F.col("posts.post_id").alias("post_id"),
            F.col("posts.score").alias("post_score"),
            F.col("posts.url").alias("post_url"),
            F.col("comments.comment_id").alias("comment_id"),
            F.to_date(F.col("comments.created_utc")).alias("summary_date"),
            F.current_timestamp().alias("processed_date"),
            F.col("posts.title").alias("post_content"),
            F.col("comments.body").alias("comment_body")
        )

        # Deduplication and quality filtering
        daily_summary_df = daily_summary_df.filter(
            (F.col("comment_body").isNotNull()) & 
            (F.col("comment_id").isNotNull()) &
            (F.col("post_id").isNotNull())
        )
```
To create daily summaries, we load new posts and comments for each subreddit. We then join these datasets on post_id, select the necessary fields, and filter out any rows with null values. This function combines post and comment data, preparing it for further analysis.


The daily summary data is stored with the following schema:

```sql
CREATE TABLE processed_data.daily_summary_data (
    id SERIAL,
    subreddit TEXT,
    post_id TEXT,
    post_score INTEGER,
    post_url TEXT,
    comment_id TEXT,
    summary_date DATE,
    processed_date TIMESTAMPTZ,
    needs_processing BOOLEAN DEFAULT TRUE,
    post_content TEXT,
    comment_body TEXT,
    PRIMARY KEY (post_id, comment_id)
);
```

#### dbt
##### DBT Data Transformations
Our data transformations are managed using dbt, which allows us to structure our project with specific models for each stage of processing. This approach provides a clear, modular, and maintainable data pipeline.

```yaml
# dbt_project.yml configuration
name: 'dbt_reddit_summary_local'
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
-- Materialized as a view for real-time filtering
SELECT
    id, subreddit, post_id, post_score, post_url,
    comment_id, summary_date, processed_date,
    post_content, comment_body
FROM {{ source('summary_analytics', 'daily_summary_data') }}
WHERE comment_body IS NOT NULL
    AND needs_processing = TRUE
```
The current_summary_staging model is a view that selects key fields from the daily summary data, filtering out rows with null comment bodies and those that do not need processing. This view is materialized as a view for real-time filtering.


2. **Analysis Model** (`joined_summary_analysis`):
```sql
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
            WHEN ts.comment_summary IS NULL THEN 'Missing'
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
    """Main function to analyze Reddit data using Gemini."""
    # Set up dated output directory
    current_date = datetime.now()
    output_dir = os.path.join(
        '/opt/airflow/results',
        current_date.strftime('%Y'),
        current_date.strftime('%m'),
        current_date.strftime('%d')
    )
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Gemini model
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    # Process each subreddit
    for subreddit in SUBREDDITS:
        process_subreddit(model, cur, subreddit, output_dir)
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
  AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
  AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://airflow:airflow@postgres/airflow
  AIRFLOW__CELERY__BROKER_URL: redis://:@redis:6379/0
  AIRFLOW__METRICS__STATSD_ON: "true"
  AIRFLOW__METRICS__STATSD_HOST: "statsd-exporter"
  AIRFLOW__METRICS__STATSD_PORT: "9125"
```

The Airflow services are deployed as separate containers, each with specific responsibilities:

1. **Airflow Webserver**
   ```yaml
   airflow-webserver:
     build:
       context: .
       dockerfile: Dockerfile.webserver
     ports:
       - "8080:8080"
     volumes: *common-volumes
     environment:
       <<: *common-env
     command: airflow webserver
     healthcheck:
       test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
       interval: 30s
       timeout: 30s
       retries: 10
   ```
Our Airflow webserver provides the web UI for DAG management and monitoring, handles user authentication and authorization, exposes REST API endpoints, and includes health checks to ensure UI availability.

2. **Airflow Scheduler**
   ```yaml
   airflow-scheduler:
     build:
       context: .
       dockerfile: Dockerfile.scheduler
     volumes: *common-volumes
     environment:
       <<: *common-env
     command: airflow scheduler
     healthcheck:
       test: ["CMD-SHELL", 'airflow jobs check --job-type SchedulerJob --hostname "$${HOSTNAME}"']
       interval: 30s
       retries: 10
   ```
The Airflow scheduler monitors and triggers task execution, manages DAG parsing and scheduling, handles task dependencies and queuing, and ensures proper task distribution to workers. This component is crucial for orchestrating our data pipeline.

3. **Airflow Worker**
   ```yaml
   airflow-worker:
     build:
       context: .
       dockerfile: Dockerfile.worker
     volumes: *common-volumes
     environment:
       <<: *common-env
       TRANSFORMERS_OFFLINE: "0"
       TOKENIZERS_PARALLELISM: "false"
       PYTORCH_ENABLE_MPS_FALLBACK: "1"
     command: airflow celery worker
     healthcheck:
       test: ["CMD-SHELL", 'celery inspect ping -d "celery@$${HOSTNAME}"']
       interval: 30s
       retries: 5
   ```
Our Airflow worker executes the actual tasks, handles ML model inference, manages resource allocation, supports parallel task execution, and is configured for ML workloads with PyTorch and Transformers.

4. **Airflow Init**
   ```yaml
   airflow-init:
     build:
       context: .
       dockerfile: Dockerfile.webserver
     command: >
       bash -c "
       airflow db init &&
       airflow db upgrade &&
       airflow users create -r Admin -u admin -p admin -e admin@example.com -f Anonymous -l Admin
       "
   ```
The Airflow init service initializes the Airflow database, creates an admin user, performs database migrations, and runs only during the initial setup. This component is essential for setting up the Airflow environment.


##### 2. Pipeline Structure
Our DAG (`reddit_pipeline`) is organized into 26 stages but can be categorized into 5 main sections, each with specific responsibilities and metrics collection:

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
       bash_command='cd /opt/airflow/dags/dbt_reddit_summary_local && dbt run --select current_summary_staging',
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

##### 3. Task Dependencies
The pipeline follows a clear dependency chain:
```python
ingest_task >> ingest_metrics_task >> \
dbt_test_raw_sources >> dbt_test_raw_metrics_task >> \
dbt_staging_task >> dbt_test_staging_models >> \
summarize_task >> sentiment_task >> \
dbt_join_summary_analysis_task >> \
gemini_task >> push_gemini_results_task
```
Our pipeline starts with data ingestion and preprocessing, followed by DBT testing and staging, then text summarization and sentiment analysis, and finally Gemini AI analysis and result pushing.

##### 4. Integration Points
The orchestrator connects with PostgreSQL for pipeline data storage, local model deployment for model serving, StatsD and Prometheus for monitoring, and GitHub for version control of results. These connections are essential for the functionality of our pipeline

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
- [Project GitHub Repository](https://github.com/SulmanK/reddit_ai_pulse_local_public)
- [Web Application](https://reddit-text-insight-and-sentiment-website-local.vercel.app/)



