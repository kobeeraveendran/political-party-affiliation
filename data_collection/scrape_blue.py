import praw
from config import reddit_credentials

user_agent, client_id, client_secret, username, password = reddit_credentials()

if None in [user_agent, client_id, client_secret, username, password]:
    raise ValueError("Please provide the Reddit API credentials of your project in ../config.py")
