# XCheck

I thought I was doing well on X with 20K followers. Turns out, I was mostly talking to ghosts and bots that have invaded X but are hard to detect. That's why I built XCheck, a personal X detective that analyzes your network.

![XCheck](https://samim.io/static/upload/Screenshot-20240909090323-1340x758.png)

**Backstory:** X/Twitter user since 2007, gradually built a network. Recently noticed odd plummeting engagement. Manual analysis revealed a suspicion: inactive followers and bots were significantly impacting my reach. Digging into other accounts, I discovered this wasn't isolated - it's a widespread X phenomenon.

**Enter XCheck:** A little open-source tool that crawls and analyzes your X account (or any other's) and uncovers hidden patterns in your social network.

**XCheck's key features:**

- Intuitive, interactive web UI for X crawling and exploration
- Analyzes followers/following of any public X account
- Auto-assigns quality scores to accounts
- Filtering, search and visualization tools

## Requirements

- Python 3.10+
- MongoDB

## Installation

pip install -r requirements.txt

Set enviroment variables:

- export X_USERNAME="your username"
- export X_PASSWORD="your password"
- export MONGO_URI="mongodb://localhost:27017"
- export MONGO_DB="xcheck"
- export RATE_LIMIT_DELAY=2.0

## Usage

python main.py
