import praw
from config import get_reddit_credentials

credentials = get_reddit_credentials()

if None in credentials.values() or "" in credentials.values():
    raise ValueError("Please provide valid Reddit API credentials of your project in a file called 'config.py', following the format in 'example_config.py'")

reddit = praw.Reddit(client_id = credentials["client_id"], 
                     client_secret = credentials["client_secret"], 
                     username = credentials["username"], 
                     password = credentials["password"], 
                     user_agent = credentials["user_agent"])

print(reddit.user.me())