#!/usr/bin/env python3

import argparse
import requests
import yaml
import os
import logging
import time
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from openai import OpenAI
from typing import List, Dict, Optional, Set
import tiktoken
from tvdb_v4_official import TVDB

import logging

def setup_logging(log_level):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Directly map input log_level to Python logging levels
    if log_level == 0:
        level = logging.CRITICAL
    elif log_level == 1:
        level = logging.ERROR
    elif log_level == 2:
        level = logging.WARNING
    elif log_level == 3:
        level = logging.INFO
    elif log_level == 4:
        level = logging.DEBUG
    else:
        level = logging.INFO  # Default to INFO if an invalid level is provided
    
    # Set the basic configuration
    logging.basicConfig(level=level, format=log_format)
    
    # Configure urllib3 and openai loggers
    if log_level < 3:  # CRITICAL, ERROR, or WARNING
        urllib3_level = openai_level = logging.WARNING
    elif log_level == 3:  # INFO
        urllib3_level = openai_level = logging.INFO
    else:  # DEBUG
        urllib3_level = openai_level = logging.DEBUG

    # Explicitly set logging levels for specific loggers
    logging.getLogger('urllib3').setLevel(urllib3_level)
    logging.getLogger('openai').setLevel(openai_level)
    
    # Explicitly set logging level for all openai subloggers
    for name in logging.root.manager.loggerDict:
        if name.startswith('openai'):
            logging.getLogger(name).setLevel(openai_level)

    if log_level == 4:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.debug(f"Root logger level: {logging.getLogger().getEffectiveLevel()}")
    logging.debug(f"urllib3 logger level: {logging.getLogger('urllib3').getEffectiveLevel()}")
    logging.debug(f"openai logger level: {logging.getLogger('openai').getEffectiveLevel()}")
    logging.info(f"Logging level set to: {logging.getLevelName(level)}")

def fetch_page_content(url: str, timeout: int, session: requests.Session) -> Optional[str]:
    try:
        logging.info(f"Fetching URL: {url}")
        start_time = time.time()
        response = session.get(
            url,
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=timeout
        )
        response.raise_for_status()
        elapsed_time = time.time() - start_time
        logging.info(
            f"Successfully fetched content from {url} in "
            f"{elapsed_time:.2f} seconds"
        )
        return response.text
    except requests.RequestException as e:
        logging.error(f"Error fetching URL {url}: {e}")
        return None

def extract_thread_urls(
    flair_url: str,
    max_threads: int,
    timeout: int,
    session: requests.Session
) -> List[str]:
    logging.info(f"Extracting thread URLs from flair URL: {flair_url}")
    start_time = time.time()

    html_content = fetch_page_content(flair_url, timeout, session)
    if not html_content:
        logging.error("Failed to fetch flair URL content")
        return []

    soup = BeautifulSoup(html_content, 'html.parser')
    posts = soup.find_all('a', class_='search-title')
    logging.info(f"Found {len(posts)} posts from flair URL")

    thread_urls = []

    for post in posts[:max_threads]:
        thread_url = post['href']
        if not thread_url.startswith('http'):
            thread_url = f"https://old.reddit.com{thread_url}"
        logging.info(f"Thread URL: {thread_url}")
        thread_urls.append(thread_url)

    elapsed_time = time.time() - start_time
    logging.info(
        f"Extracted {len(thread_urls)} thread URLs in "
        f"{elapsed_time:.2f} seconds"
    )
    return thread_urls

def num_tokens_from_messages(messages: List[Dict], model: str) -> int:
    """
    Returns the number of tokens used by a list of messages.
    """
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == 'name':
                num_tokens += -1  # Role is always required and always 1 token
    num_tokens += 2  # Every reply is primed with <im_start>assistant
    return num_tokens

def extract_tv_shows_from_reddit_thread(
    thread_url: str,
    openai_config: Dict,
    openai_client: OpenAI,
    timeout: int,
    session: requests.Session
) -> List[str]:
    logging.info(f"Extracting TV shows from thread URL: {thread_url}")
    start_time = time.time()

    logging.info("Fetching page content")
    html_content = fetch_page_content(thread_url, timeout, session)
    if not html_content:
        logging.error(f"Failed to fetch content from {thread_url}")
        return []

    logging.info("Parsing HTML content")
    soup = BeautifulSoup(html_content, 'html.parser')
    comments = soup.find_all('div', class_='md')
    logging.info(f"Found {len(comments)} comments in thread")

    # Limit comments for testing
    max_comments = 5  # Adjust this number as needed
    comments = comments[:max_comments]
    logging.info(f"Processing {len(comments)} comments")

    all_comments = [comment.get_text() for comment in comments]

    logging.info("Extracting TV shows using GPT")
    tv_shows = extract_tv_shows_with_gpt(
        all_comments, openai_config, openai_client
    )

    elapsed_time = time.time() - start_time
    logging.info(
        f"Extracted {len(tv_shows)} TV shows in {elapsed_time:.2f} seconds"
    )
    return tv_shows

def extract_tv_shows_with_gpt(
    comments: List[str],
    openai_config: Dict,
    openai_client: OpenAI
) -> List[str]:
    logging.info("Entered extract_tv_shows_with_gpt")
    start_time = time.time()

    model = openai_config['model']
    max_tokens = openai_config.get('max_tokens', 150)
    temperature = openai_config.get('temperature', 0.7)
    top_p = openai_config.get('top_p', 1.0)
    max_context_tokens = openai_config.get('max_context_tokens', 8000)
    prompt_buffer_tokens = openai_config.get('prompt_buffer_tokens', 1000)

    # Define the maximum tokens we can use for the prompt
    max_prompt_tokens = max_context_tokens - max_tokens - prompt_buffer_tokens

    encoding = tiktoken.encoding_for_model(model)

    tv_shows = set()
    batch_comments = []
    batch_tokens = 0

    logging.info(f"Total comments to process: {len(comments)}")

    for comment in comments:
        logging.debug(f"Processing comment: {comment[:50]}...")
        comment_tokens = len(encoding.encode(comment))
        if batch_tokens + comment_tokens > max_prompt_tokens:
            # Process the current batch
            batch_tv_shows = process_comment_batch_with_gpt(
                batch_comments, model, max_tokens, temperature, top_p,
                openai_client
            )
            tv_shows.update(batch_tv_shows)
            # Reset batch
            batch_comments = [comment]
            batch_tokens = comment_tokens
        else:
            batch_comments.append(comment)
            batch_tokens += comment_tokens

    # Process any remaining comments
    if batch_comments:
        batch_tv_shows = process_comment_batch_with_gpt(
            batch_comments, model, max_tokens, temperature, top_p,
            openai_client
        )
        tv_shows.update(batch_tv_shows)

    elapsed_time = time.time() - start_time
    logging.info(
        f"Extracted {len(tv_shows)} unique TV shows using GPT in "
        f"{elapsed_time:.2f} seconds"
    )
    return list(tv_shows)

def process_comment_batch_with_gpt(
    batch_comments: List[str],
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    openai_client: OpenAI
) -> List[str]:
    combined_comments = "\n".join(batch_comments)
    prompt = (
        "Given the following list of comments from a Reddit thread about TV shows, "
        "extract and return a list of unique TV show names mentioned. "
        "Ignore any other text that isn't a TV show name.\n\n"
        "Comments:\n"
        f"{combined_comments}\n\n"
        "Return the list of TV show names, one per line."
    )
    try:
        logging.info("Preparing prompt for OpenAI API")
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts TV show names from text."
            },
            {
                "role": "user",
                "content": prompt
            },
        ]
        # Estimate total tokens
        total_tokens = num_tokens_from_messages(messages, model) + max_tokens
        if total_tokens > 8192:
            logging.warning("Batch prompt is too long, skipping this batch.")
            return []

        logging.info("Calling OpenAI API")
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        logging.info("Received response from OpenAI API")
        batch_tv_shows = response.choices[0].message.content.strip().split('\n')
        batch_tv_shows = [show.strip() for show in batch_tv_shows if show.strip()]
        return batch_tv_shows
    except Exception as e:
        logging.error(f"Error with OpenAI API: {e}")
        return []

def fetch_tvdb_details(
    show_name: str,
    tvdb_client: TVDB
) -> Optional[Dict]:
    logging.info(f"Fetching TVDB details for show: {show_name}")
    start_time = time.time()

    try:
        # Search for the TV show
        logging.info(f"Searching for TV show '{show_name}' in TVDB")
        search_results = tvdb_client.search(show_name, type="series")

        if not search_results:
            logging.warning(f"No TVDB results found for {show_name}")
            return None

        # Get the first result
        series_data = search_results[0]

        # Extract relevant information
        title = series_data.get('name', 'Unknown Title')
        overview = series_data.get('overview', 'No overview available.')
        year = series_data.get('year', 'Unknown Year')

        tvdb_details = {
            'title': title,
            'overview': overview,
            'year': year
        }

        elapsed_time = time.time() - start_time
        logging.info(
            f"Fetched TVDB details for {show_name} in {elapsed_time:.2f} seconds: "
            f"{tvdb_details}"
        )
        return tvdb_details
    except Exception as e:
        logging.error(f"Error fetching TVDB details for {show_name}: {e}")
        return None

def load_config(config_file: str) -> Dict:
    logging.info(f"Loading config from file: {config_file}")
    start_time = time.time()

    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        elapsed_time = time.time() - start_time
        logging.info(
            f"Loaded config in {elapsed_time:.2f} seconds: {config}"
        )
        return config
    logging.error(f"Config file not found: {config_file}")
    return {}

def analyze_with_openai(
    show_list: List[Dict],
    preferences: Dict,
    openai_config: Dict,
    openai_client: OpenAI
) -> str:
    logging.info("Generating recommendations using OpenAI")
    start_time = time.time()

    model = openai_config['model']
    max_tokens = openai_config.get('max_tokens', 150)
    temperature = openai_config.get('temperature', 0.7)
    top_p = openai_config.get('top_p', 1.0)
    max_context_tokens = openai_config.get('max_context_tokens', 8000)
    prompt_buffer_tokens = openai_config.get('prompt_buffer_tokens', 1000)

    # Define the maximum tokens we can use for the prompt
    max_prompt_tokens = max_context_tokens - max_tokens - prompt_buffer_tokens

    encoding = tiktoken.encoding_for_model(model)

    # Build the initial prompt
    prompt_header = (
        "You are given a list of TV shows and some preferences about "
        "what the user likes and dislikes. Please recommend which "
        "shows the user should watch based on their preferences.\n\n"
    )
    recommendations_header = "\n\nRecommendations:"

    # Build the TV shows list, adding one by one until the token limit is reached
    tv_show_list_str = ''
    for show in show_list:
        show_info = (
            f"Title: {show['title']} ({show['year']})\n"
            f"Overview: {show['overview']}\n\n"
        )
        temp_prompt = (
            prompt_header +
            f"TV Shows:\n{tv_show_list_str + show_info}"
            f"User Preferences:\n{preferences}{recommendations_header}"
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": temp_prompt}
        ]
        total_tokens = num_tokens_from_messages(messages, model) + max_tokens
        logging.debug(f"Total tokens with current show: {total_tokens}")
        if total_tokens > max_prompt_tokens:
            # Stop adding more shows
            logging.error("Reached max token limit, stopping addition of shows")
            break
        else:
            tv_show_list_str += show_info

    # Build the final prompt
    prompt = (
        prompt_header +
        f"TV Shows:\n{tv_show_list_str}"
        f"User Preferences:\n{preferences}{recommendations_header}"
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    try:
        logging.info("Calling OpenAI API for recommendations")
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        logging.info("Received response from OpenAI API")
        recommendations = response.choices[0].message.content.strip()
        elapsed_time = time.time() - start_time
        logging.info(
            f"Generated OpenAI recommendations in {elapsed_time:.2f} seconds"
        )
        return recommendations
    except Exception as e:
        logging.error(f"Error with OpenAI API: {e}")
        return "Could not generate recommendations due to an error."

def get_session_with_retries(
    retries: int,
    backoff_factor: float,
    status_forcelist: List[int]
) -> requests.Session:
    session = requests.Session()
    retry = requests.packages.urllib3.util.retry.Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

def read_shows_from_file(file_path: str) -> List[str]:
    logging.info(f"Reading shows from file: {file_path}")
    with open(file_path, 'r') as f:
        shows = [line.strip() for line in f if line.strip()]
    logging.info(f"Read {len(shows)} shows from file")
    return shows


def read_processed_shows(file_path: str) -> Dict[str, List[Dict]]:
    logging.debug(f"Attempting to read processed shows from {file_path}")
    if not os.path.exists(file_path):
        logging.debug(f"File {file_path} does not exist. Returning empty structure.")
        return {
            "recommendations": [],
            "avoid": []
        }
    with open(file_path, 'r') as f:
        content = f.read()
        logging.debug(f"Content read from {file_path}: {content}")
        if not content.strip():
            logging.debug(f"File {file_path} is empty. Returning empty structure.")
            return {
                "recommendations": [],
                "avoid": []
            }
        try:
            data = yaml.safe_load(content)
            logging.debug(f"Parsed YAML data: {data}")
            if not data or not isinstance(data, dict):
                logging.warning(f"Invalid data structure in {file_path}. Returning empty structure.")
                return {
                    "recommendations": [],
                    "avoid": []
                }
            # Ensure both categories exist
            if "recommendations" not in data:
                data["recommendations"] = []
            if "avoid" not in data:
                data["avoid"] = []
            return data
        except yaml.YAMLError as e:
            logging.error(f"Error reading YAML file: {e}")
            return {
                "recommendations": [],
                "avoid": []
            }
def write_processed_shows(file_path: str, shows: Dict[str, List[Dict]]):
    logging.debug(f"Attempting to write processed shows to {file_path}")
    logging.debug(f"Data to be written: {shows}")
    with open(file_path, 'w') as f:
        yaml.dump(shows, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=1000, indent=2)
    logging.debug(f"Finished writing to {file_path}")
    # Read back the file contents to verify
    with open(file_path, 'r') as f:
        content = f.read()
        logging.debug(f"Content written to {file_path}: {content}")

def update_processed_shows(processed_shows: Dict[str, List[Dict]], new_recommendations: str) -> Dict[str, List[Dict]]:
    logging.debug("Entering update_processed_shows")
    logging.debug(f"Current processed shows: {processed_shows}")
    logging.debug(f"New recommendations: {new_recommendations}")

    new_shows = {
        "recommendations": [],
        "avoid": []
    }
    current_category = None
    current_show = {}

    lines = new_recommendations.split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        logging.debug(f"Processing line: {line}")

        if "Shows to **Avoid**" in line:
            current_category = "avoid"
        elif line.startswith(("1.", "2.", "3.", "4.", "5.")) and "**" in line:
            if current_show:
                new_shows[current_category].append(current_show)
                current_show = {}
            show_name = line.split("**")[1].strip()
            current_show = {"title": show_name}
            if current_category is None:
                current_category = "recommendations"
        elif line.startswith("- **Overview**:") and current_show:
            current_show["overview"] = line.split(":", 1)[1].strip()
        elif line.startswith("- **Why**:") and current_show:
            if current_category == "recommendations":
                current_show["reason_for_recommendation"] = line.split(":", 1)[1].strip()
            else:
                current_show["reason_to_avoid"] = line.split(":", 1)[1].strip()

    if current_show:
        new_shows[current_category].append(current_show)

    logging.debug(f"Parsed new shows: {new_shows}")

    # Create sets of existing show titles
    existing_recommendations = set(show['title'] for show in processed_shows['recommendations'])
    existing_avoid = set(show['title'] for show in processed_shows['avoid'])

    # Add new recommendations
    for show in new_shows['recommendations']:
        if show['title'] not in existing_recommendations:
            processed_shows['recommendations'].append(show)
            logging.debug(f"Added new recommendation: {show['title']}")
        if show['title'] in existing_avoid:
            processed_shows['avoid'] = [s for s in processed_shows['avoid'] if s['title'] != show['title']]
            logging.debug(f"Removed {show['title']} from avoid list")

    # Add new shows to avoid
    for show in new_shows['avoid']:
        if show['title'] not in existing_avoid:
            processed_shows['avoid'].append(show)
            logging.debug(f"Added new show to avoid: {show['title']}")
        if show['title'] in existing_recommendations:
            processed_shows['recommendations'] = [s for s in processed_shows['recommendations'] if s['title'] != show['title']]
            logging.debug(f"Removed {show['title']} from recommendations list")

    logging.debug(f"Updated processed shows: {processed_shows}")
    return processed_shows

def process_shows(shows: List[str], config: Dict, tvdb_client: TVDB, processed_shows: Dict[str, List[Dict]]) -> str:
    logging.info(f"Processing {len(shows)} shows")
    start_time = time.time()

    # Initialize OpenAI client
    openai_client = OpenAI(api_key=config['openai']['key'])

    all_shows = []
    processed_show_names = set()
    for category in processed_shows.values():
        for show in category:
            if isinstance(show, dict) and 'title' in show:
                processed_show_names.add(show['title'].split(' (')[0])
            elif isinstance(show, str):
                processed_show_names.add(show.split(' (')[0])
            else:
                logging.warning(f"Unexpected show format in processed shows: {show}")

    for show in shows:
        try:
            if show in processed_show_names:
                logging.info(f"Skipping already processed show: {show}")
                continue
            tvdb_details = fetch_tvdb_details(show, tvdb_client)
            if tvdb_details:
                all_shows.append(tvdb_details)
            else:
                logging.warning(f"Could not fetch TVDB details for {show}")
        except Exception as e:
            logging.error(f"Error processing show '{show}': {str(e)}")

    max_top_shows = config['reddit']['max_top_shows']
    top_shows = all_shows[:max_top_shows]
    logging.info(f"Top {max_top_shows} shows for recommendation: {top_shows}")

    if not top_shows:
        logging.error("No valid shows found for recommendations")
        return "Could not generate recommendations due to lack of valid show data."

    try:
        recommendations = analyze_with_openai(
            top_shows, config['preferences'], config['openai'], openai_client
        )
    except Exception as e:
        logging.error(f"Error generating recommendations: {str(e)}")
        return "Could not generate recommendations due to an error."

    elapsed_time = time.time() - start_time
    logging.info(
        f"Processed all shows and generated recommendations in "
        f"{elapsed_time:.2f} seconds"
    )
    return recommendations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TV Show Recommender")
    parser.add_argument('-c', '--config', default='config.yml', help="YAML config file with settings")
    parser.add_argument('-s', '--source', help="File containing a list of TV shows to process")

    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get('debug_level', 0))
    logging.debug("Logging setup complete. This is a debug message.")

    try:
        overall_start_time = time.time()

        # Initialize TVDB client
        tvdb_api_key = config['tvdb']['api_key']
        tvdb_pin = config['tvdb']['pin']
        tvdb_client = TVDB(tvdb_api_key, pin=tvdb_pin)

        # Read processed shows
        processed_file = config.get('processed_file', 'processed.yml')
        processed_shows = read_processed_shows(processed_file)
        logging.debug("Content of processed.md before processing:")
        with open(processed_file, 'r') as f:
          logging.debug(f.read())

        if args.source:
            shows = read_shows_from_file(args.source)
            logging.debug(f"Contents of {args.source}:")
            with open(args.source, 'r') as f:
                logging.debug(f.read())
        else:
            timeout = config['network'].get('timeout', 10)
            max_threads = config['reddit'].get('max_threads', 2)
            session = get_session_with_retries(
                retries=3,
                backoff_factor=0.3,
                status_forcelist=[500, 502, 504]
            )
            thread_urls = extract_thread_urls(
                config['reddit']['flair_url'],
                max_threads,
                timeout,
                session
            )
            if not thread_urls:
                logging.error("No thread URLs found. Exiting.")
                exit(1)
            shows = extract_tv_shows_from_reddit_thread(
                thread_urls[0], config['openai'], OpenAI(api_key=config['openai']['key']), timeout, session
            )

        recommendations = process_shows(shows, config, tvdb_client, processed_shows)

        # Update processed shows with new recommendations
        updated_processed_shows = update_processed_shows(processed_shows, recommendations)

        # Write updated processed shows
        write_processed_shows(processed_file, updated_processed_shows)
        logging.debug("Content of processed.yml after processing:")
        with open(processed_file, 'r') as f:
          logging.debug(f.read())

        overall_elapsed_time = time.time() - overall_start_time
        logging.info(
            f"Total script execution time: {overall_elapsed_time:.2f} seconds"
        )

        print(f"Recommendations have been updated in {processed_file}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        exit(1)
