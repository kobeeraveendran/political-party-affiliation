import praw
from config import reddit_credentials

user_agent, client_id, client_secret, username, password = reddit_credentials()

if not client_id or not client_secret or not username or not password:
    raise ValueError("Please provide the Reddit API credentials of your project in ../config.py")

reddit = praw.Reddit(client_id = client_id, 
                     client_secret = client_secret, 
                     username = username, 
                     password = password, 
                     user_agent = )