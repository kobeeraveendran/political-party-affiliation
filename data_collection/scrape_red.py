import praw
from config import get_reddit_credentials
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--subreddit", type = str, help = "Name of the subreddit to be scraped.")
parser.add_argument("--batch_size", type = int, default = 1500, help = "Number of comments to scrape from the specified subreddit.")

args = parser.parse_args()

credentials = get_reddit_credentials()

if None in credentials.values() or "" in credentials.values():
    raise ValueError("Please provide valid Reddit API credentials of your project in a file called 'config.py', following the format in 'example_config.py'")

reddit = praw.Reddit(client_id = credentials["client_id"], 
                     client_secret = credentials["client_secret"], 
                     username = credentials["username"], 
                     password = credentials["password"], 
                     user_agent = credentials["user_agent"])

if not args.subreddit:
    raise ValueError("Invalid subreddit name. Please specify one, or use 'all'.")


batch_size = args.batch_size

print("Batch size: ", batch_size)
print("Subreddit specified: ", args.subreddit)

subreddit = reddit.subreddit(args.subreddit)

# print("Subreddit name: ", subreddit.display_name)
# print("Subreddit title: ", subreddit.title)
# print("Subreddit description: ", subreddit.description)

comment_count = 0

for comment in subreddit.comments(limit = 5):
    print("\n\nBODY: ", comment.body)
    print("\n\nREPLIES: ", comment.replies)
    print("\nSCORE: ", comment.score)
    print("\n\nPERMALINK: ", comment.permalink)
    