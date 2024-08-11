import os
import json
import logging
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright, Playwright, TimeoutError as PlaywrightTimeoutError
import pandas as pd
import numpy as np
from dotenv import load_dotenv, set_key
import time
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import matplotlib.pyplot as plt
import base64
import io
from pymongo import MongoClient
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
from contextlib import asynccontextmanager

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

X_USERNAME = os.getenv('X_USERNAME')
X_PASSWORD = os.getenv('X_PASSWORD')
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
MONGO_DB = os.getenv('MONGO_DB', 'x_followers')

# Create MongoDB client
client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
followers_collection = db['followers']

# A global variable to store the scraping status
scraping_status = {"status": "idle", "progress": 0, "total": 0}

# Rate limiting
RATE_LIMIT_DELAY = float(os.getenv('RATE_LIMIT_DELAY', '2.0'))  # Default to 2 seconds

# Crawl Settings
headless = True

# Define a constant user agent
CONSISTENT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

def load_cookies(context):
    if os.path.exists('cookies.json'):
        with open('cookies.json', 'r') as f:
            cookies = json.load(f)
        context.add_cookies(cookies)

def save_cookies(context):
    cookies = context.cookies()
    with open('cookies.json', 'w') as f:
        json.dump(cookies, f)

async def create_persistent_context(playwright: Playwright):
    user_data_dir = Path("./user_data_dir")
    user_data_dir.mkdir(exist_ok=True)

    browser = await playwright.chromium.launch_persistent_context(
        user_data_dir=str(user_data_dir),
        headless=headless,
        args=[
            f'--user-agent={CONSISTENT_USER_AGENT}',
            '--disable-blink-features=AutomationControlled',
        ],
        viewport={"width": 1280, "height": 720},
        device_scale_factor=1,
        ignore_https_errors=True,
        java_script_enabled=True,
        locale='en-US',
        timezone_id='America/New_York',
        timeout=60000  # Increase timeout to 60 seconds
    )

    return browser

async def login_to_x(page, username, password):
    await page.goto("https://x.com/home")
    
    # Check if already logged in
    if await is_logged_in(page):
        print("Already logged in, skipping login process")
        return

    print("Not logged in, proceeding with login")
    await page.goto("https://x.com/i/flow/login")
    
    try:
        print("Logging in to X...")
        
        # Wait for the username input field to be present
        username_selector = 'input[autocomplete="username"]'
        await page.wait_for_selector(username_selector, timeout=10000)
        
        # Fill the username
        await page.fill(username_selector, username)
        
        # Click the 'Next' button
        await page.locator("//span[text()='Next']").click()
        await page.wait_for_timeout(1000)  # Wait for 1 second
        
        # Wait for the password input field to be present
        password_selector = 'input[autocomplete="current-password"]'
        await page.wait_for_selector(password_selector, timeout=10000)
        
        # Fill the password
        await page.fill(password_selector, password)
        
        # Click the 'Log in' button
        await page.locator("//span[text()='Log in']").click()
        
        # Wait for URL to change or for navigation to occur
        await page.wait_for_url("**/*", timeout=30000)
        await page.wait_for_timeout(2000)
        
        # Check if login was successful
        if await is_logged_in(page):
            print("Successfully logged in to X")
        else:
            print("Login may have failed. Please check the page manually.")
        
    except PlaywrightTimeoutError as e:
        print(f"Login failed: Timeout while waiting for element. Error: {str(e)}")
        raise
    except Exception as e:
        print(f"An error occurred during login: {str(e)}")
        raise

async def is_logged_in(page):
    return page.url.startswith("https://x.com/home") or await page.locator('a[data-testid="AppTabBar_Home_Link"]').count() > 0

async def background_scrape(username: str, max_followers: int):
    global scraping_status, persistent_context, X_USERNAME, X_PASSWORD
    scraping_status = {"status": "running", "progress": 0, "total": max_followers}

    try:
        if not persistent_context:
            logging.info("Persistent context is not available. Creating a new one.")
            p = await async_playwright().start()
            persistent_context = await create_persistent_context(p)

        page = await persistent_context.new_page()
        try:
            await login_to_x(page, X_USERNAME, X_PASSWORD)
            
            followers = await scrape_follower_list(page, username, max_followers)
            if followers:
                enriched_followers = await enrich_follower_data(page, followers, max_detailed=max_followers)
                df_followers = pd.DataFrame(enriched_followers)
                save_to_mongodb(df_followers)
            
            scraping_status = {"status": "completed", "progress": max_followers, "total": max_followers}
        except PlaywrightTimeoutError as e:
            logging.error(f"Timeout error during scraping: {str(e)}")
            scraping_status = {"status": "error", "message": f"Timeout error during scraping: {str(e)}"}
        except Exception as e:
            logging.error(f"Error during scraping: {str(e)}")
            scraping_status = {"status": "error", "message": f"Error during scraping: {str(e)}"}
        finally:
            await page.close()
    except Exception as e:
        logging.error(f"Critical error in background_scrape: {str(e)}")
        scraping_status = {"status": "error", "message": f"Critical error occurred: {str(e)}"}
        
async def scrape_follower_list(page, username, max_followers=1000):
    global scraping_status
    print(f"Scraping followers list for {username}...")
    await page.goto(f'https://x.com/{username}/followers')
    followers = []
    retry_count = 0
    max_retries = 3

    while len(followers) < max_followers and retry_count < max_retries:
        try:
            await page.wait_for_selector('div[data-testid="cellInnerDiv"]', timeout=10000)
            follower_elements = await page.query_selector_all('div[data-testid="cellInnerDiv"]')
            
            if not follower_elements:
                retry_count += 1
                print(f"No followers found. Retry {retry_count}/{max_retries}")
                await page.reload()
                await asyncio.sleep(5)
                continue

            for element in follower_elements:
                if len(followers) >= max_followers:
                    break
                
                try:
                    name = await element.query_selector('div[dir="ltr"] > span')
                    handle = await element.query_selector('div[dir="ltr"] > span:has-text("@")')
                    name = await name.inner_text() if name else "N/A"
                    handle = await handle.inner_text() if handle else "N/A"
                    followers.append({'name': name.strip(), 'handle': handle.strip()})
                    scraping_status["progress"] = len(followers)
                except Exception as e:
                    print(f"Error extracting follower basic data: {str(e)}")

            if len(followers) == 0:
                retry_count += 1
                print(f"No new followers found. Retry {retry_count}/{max_retries}")
            else:
                retry_count = 0

            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"Error during follower list scraping: {str(e)}")
            retry_count += 1
            await page.reload()
            await asyncio.sleep(5)

    print(f"Scraped basic data for {len(followers)} followers")
    return followers

def save_to_mongodb(followers_df):
    print("Saving followers data to MongoDB...")
    followers_df['quality_score'] = calculate_quality_score(followers_df)
    
    for _, follower in followers_df.iterrows():
        followers_collection.update_one(
            {'handle': follower['handle']},
            {'$set': follower.to_dict()},
            upsert=True
        )

async def extract_detailed_follower_data(page, handle):
    try:
        await page.goto(f'https://x.com/{handle}')
        await page.wait_for_selector('div[data-testid="UserName"]', timeout=10000)

        bio = await page.query_selector('div[data-testid="UserDescription"]')
        bio = await bio.inner_text() if bio else ""

        follower_count_element = await page.query_selector('a[href$="/verified_followers"] > span > span')
        follower_count = "0"
        if follower_count_element:
            follower_count = await follower_count_element.inner_text()

        following_count_element = await page.query_selector('a[href$="/following"] > span > span')
        following_count = "0"
        if following_count_element:
            following_count = await following_count_element.inner_text()

        join_date = await page.query_selector('span[data-testid="UserJoinDate"]')
        join_date = (await join_date.inner_text()).replace("Joined ", "") if join_date else "Unknown"

        verified = await page.query_selector('svg[data-testid="icon-verified"]') is not None

        # New code to extract profile image URL
        profile_image_element = await page.query_selector('img[alt="Opens profile photo"]')
        profile_image_url = await profile_image_element.get_attribute('src') if profile_image_element else None

        print(f"Debug - {handle}: Followers: {follower_count}, Following: {following_count}")
        await asyncio.sleep(RATE_LIMIT_DELAY)  # Add rate limiting delay

        return {
            'bio': bio,
            'follower_count': follower_count,
            'following_count': following_count,
            'join_date': join_date,
            'verified': verified,
            'profile_image_url': profile_image_url  # Add this line
        }
    except Exception as e:
        print(f"Failed to extract detailed data for {handle}: {str(e)}")
        return None

async def enrich_follower_data(page, followers, max_detailed=100):
    print("Enriching follower data with detailed information...")
    enriched_followers = []
    for follower in followers[:max_detailed]:
        detailed_data = await extract_detailed_follower_data(page, follower['handle'])
        if detailed_data:
            enriched_followers.append({**follower, **detailed_data})
        await asyncio.sleep(1)  # Add a delay to avoid rate limiting
    return enriched_followers

def analyze_followers_concurrently(followers, max_workers=5):
    print("Analyzing followers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_follower = {executor.submit(analyze_single_follower, follower): follower for follower in followers}
        results = []
        for future in tqdm(as_completed(future_to_follower), total=len(followers), desc="Analyzing followers"):
            results.append(future.result())
    return pd.DataFrame(results)

def analyze_single_follower(follower):
    follower['follower_count'] = convert_count(follower['follower_count'])
    follower['following_count'] = convert_count(follower['following_count'])
    follower['follower_following_ratio'] = follower['follower_count'] / max(follower['following_count'], 1)
    follower['account_age_days'] = calculate_account_age(follower['join_date'])
    follower['is_suspicious'] = check_suspicious_patterns(follower)
    follower['sentiment_score'] = analyze_sentiment(follower['bio'])
    return follower

def convert_count(count):
    if isinstance(count, str):
        if 'K' in count:
            return float(count.replace('K', '')) * 1000
        elif 'M' in count:
            return float(count.replace('M', '')) * 1000000
    return float(count)

def calculate_account_age(join_date):
    try:
        join_date = datetime.strptime(join_date, "%B %Y")
        return (datetime.now() - join_date).days
    except:
        return 0

def check_suspicious_patterns(row):
    suspicious_patterns = [
        r'\bbot\b',
        r'follow.*back',
        r'buy.*followers',
        r'get.*followers',
        r'increase.*followers',
    ]
    text = f"{row['username'].lower()} {row['name'].lower()} {row['bio'].lower()}"
    return any(re.search(pattern, text) for pattern in suspicious_patterns)

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']

def preprocess_follower_data(df):
    def convert_count(count):
        if pd.isna(count) or count == '':
            return 0.0
        if isinstance(count, str):
            count = count.replace(',', '')  # Remove commas
            if 'K' in count:
                return float(count.replace('K', '')) * 1000
            elif 'M' in count:
                return float(count.replace('M', '')) * 1000000
            elif count.isdigit():
                return float(count)
        return 0.0  # Default to 0 if conversion fails

    df['follower_count'] = df['follower_count'].apply(convert_count)
    df['following_count'] = df['following_count'].apply(convert_count)
    
    def parse_join_date(date_str):
        try:
            return pd.to_datetime(date_str, format='%B %Y')
        except:
            try:
                return pd.to_datetime(date_str)
            except:
                return pd.NaT

    df['join_date'] = df['join_date'].apply(parse_join_date)
    df['days_since_joining'] = (pd.to_datetime('now') - df['join_date']).dt.days

    # Handle NaN values
    df['days_since_joining'] = df['days_since_joining'].fillna(0)

    return df

def calculate_quality_score(df):
    print("Calculating quality scores...")
    df = preprocess_follower_data(df)
    
    score = pd.Series(0, index=df.index)
    
    # Follower count (log-scaled)
    score += np.log1p(df['follower_count']) / 10
    
    # Follower-following ratio (capped)
    df['follower_following_ratio'] = df['follower_count'] / df['following_count'].replace(0, 1)
    score += df['follower_following_ratio'].clip(upper=10) / 5
    
    # Account age
    score += np.log1p(df['days_since_joining']) / 10
    
    # Bio length
    score += df['bio'].str.len() / 100
    
    # Sentiment score (if available)
    if 'sentiment_score' in df.columns:
        score += df['sentiment_score']
    
    # Verified status
    score += df['verified'].astype(int) * 2
    
    # Normalize the score
    score = (score - score.min()) / (score.max() - score.min())
    
    return score

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global persistent_context
    p = await async_playwright().start()
    persistent_context = await create_persistent_context(p)
    yield
    # Shutdown
    if persistent_context:
        await persistent_context.close()

app = FastAPI(lifespan=lifespan)

# Global variable to store the persistent context
persistent_context = None

# Serve static files (like index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

class Credentials(BaseModel):
    username: str
    password: str

@app.get("/api/credentials")
async def get_credentials():
    return {"username": os.getenv('X_USERNAME', ''), "password": os.getenv('X_PASSWORD', '')}

@app.post("/api/credentials")
async def update_credentials(credentials: Credentials):
    global X_USERNAME, X_PASSWORD
    try:
        set_key('.env', 'X_USERNAME', credentials.username)
        set_key('.env', 'X_PASSWORD', credentials.password)
        load_dotenv(override=True)
        X_USERNAME = os.getenv('X_USERNAME')
        X_PASSWORD = os.getenv('X_PASSWORD')
        return {"message": "Credentials updated successfully"}
    except Exception as e:
        logging.error(f"Error updating credentials: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

@app.get("/api/followers")
async def get_followers():
    followers = list(followers_collection.find({}, {'_id': 0}))
    print("Sample follower data:", followers[0] if followers else "No followers found")
    return followers

class ScrapeRequest(BaseModel):
    username: str
    max_followers: int

@app.post("/api/scrape")
async def start_scrape(request: ScrapeRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(background_scrape, request.username, request.max_followers)
    return {"message": "Scraping started"}

@app.get("/api/scrape-status")
async def get_scrape_status():
    return JSONResponse(content=scraping_status)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)