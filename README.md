# TV Show Recommender

Disclaimer: I'm not a good python dev, I just tinker around. I wanted something to easily recommend shows for me using ChatGPT

TV Show Recommender is a Python script that helps you discover new TV shows based on your preferences. It can process a list of shows from a file or scrape recommendations from Reddit, and then use TVDB and OpenAI's GPT model to analyze and recommend shows tailored to your tastes.

## Features

- Process TV shows from a file or scrape from Reddit
- Fetch show details from TVDB
- Use OpenAI's GPT model to generate personalized recommendations
- Store and update recommendations in a YAML file
- Configurable via a YAML configuration file

## Prerequisites

- Python 3.6+
- OpenAI API key
- TVDB API key and PIN

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/tv-show-recommender.git
   cd tv-show-recommender
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create and configure your `config.yml`:
   ```
   cp config.sample.yml config.yml
   ```
   Edit `config.yml` and add your OpenAI API key, TVDB API key and PIN, and adjust other settings as needed.

## Configuration

The `config.yml` file contains several important settings:

- `debug_level`: Set the logging level (0-4, where 4 is the most verbose)
- `processed_file`: Name of the file to store processed shows (default: "recommendations.yml")
- `openai`: Settings for the OpenAI API (key, model, tokens, etc.)
- `tvdb`: TVDB API key and PIN
- `reddit`: Settings for Reddit scraping (URL, thread limits, etc.)
- `network`: Network request settings
- `preferences`: Your TV show preferences (favorite shows, genres, themes to avoid, etc.)

Make sure to set your API keys and adjust the preferences to get the most relevant recommendations.

## Usage

### Recommending shows from a file

1. Create a text file (e.g., `shows.txt`) with a list of TV shows, one per line.
2. Run the script:
   ```
   python what2watch.py -s shows.txt
   ```

### Scraping recommendations from Reddit

Run the script without the `-s` option:
```
python what2watch.py
```

### Additional options

- Use a different config file:
  ```
  python what2watch.py -c my_config.yml
  ```

## Output

The script will generate and update recommendations in the file specified by `processed_file` in your config (default: `recommendations.yml`). This file will contain two sections:

- `recommendations`: Shows recommended based on your preferences
- `avoid`: Shows that don't match your preferences

Each show entry includes:
- Title and year
- Overview
- Genres
- Themes
- Relevance to your preferences

## Troubleshooting

- If you encounter any issues, check the log output for error messages.
- Ensure your API keys are correctly set in the `config.yml` file.
- For more detailed logs, increase the `debug_level` in your config file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- OpenAI for their GPT model
- TVDB for their TV show database
- The Python community for the excellent libraries used in this project
