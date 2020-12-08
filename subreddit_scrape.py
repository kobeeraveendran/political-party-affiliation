import praw
from config import get_reddit_credentials
from progress.bar import ChargingBar

import argparse
import os
import time

parser = argparse.ArgumentParser()

parser.add_argument("--subreddit", type = str, help = "Name of the subreddit to be scraped.")
parser.add_argument("--batch_size", type = int, default = 1500, help = "Number of comments to scrape from the specified subreddit. Default = 1500")
parser.add_argument("--score_threshold", type = int, default = 10, help = "Minimum score of a comment required for it to be logged. Default = 10")

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

os.makedirs("datasets/", exist_ok = True)

# print("Subreddit name: ", subreddit.display_name)
# print("Subreddit title: ", subreddit.title)
# print("Subreddit description: ", subreddit.description)

def is_ascii(s):
    return all(ord(char) < 128 for char in s)


def extract_comments_tree():
    comment_count = 0

    bar = ChargingBar("Comments collected: ", max = batch_size, suffix = '%(index)d/%(max)d (%(percent)d%%)')

    for submission in subreddit.hot():

        submission.comments.replace_more(limit = None)

        for comment in submission.comments.list():

            if comment.score > args.score_threshold:
                if len(comment.body.split()) > 5 and comment.body.isascii():

                    # save to log
                    with open("datasets/{}.txt".format(args.subreddit), 'a') as logfile:
                        logfile.write(comment.body.replace('\n', '') + "\n")

                    comment_count += 1
                    bar.next()

            if comment_count >= batch_size:
                break

        if comment_count >= batch_size:
            break

    bar.finish()

    print("Comments collected: ", comment_count)

def extract_top_level_comments():

    comment_count = 0

    bar = ChargingBar("Comments collected: ", max = batch_size, suffix = '%(index)d/%(max)d (%(percent)d%%)')

    for submission in subreddit.hot(limit = None):

        submission.comments.replace_more(limit = None)

        for comment in submission.comments:

            if comment.score > args.score_threshold:
                if len(comment.body.split()) > 5 and comment.body.isascii() and comment.body.isprintable():

                    # save to log
                    with open("datasets/{}_toplvl.txt".format(args.subreddit), 'a') as logfile:
                        logfile.write(comment.body.replace('\n', '') + "\n")

                    comment_count += 1
                    bar.next()

            if comment_count >= batch_size:
                break

        if comment_count >= batch_size:
            break

    bar.finish()

    # print("Comments collected: ", comment_count)

print("Beginning comment extraction...\n")

start = time.time()
extract_comments_tree()
#extract_top_level_comments()
end = time.time()

print("Execution time: {:.2f}s".format(end - start))

# for comment in subreddit.comments(limit = 5):
#     print("\n\nBODY: ", comment.body)
#     print("\n\nREPLIES: ", comment.replies)
#     print("\nSCORE: ", comment.score)
#     print("\n\nPERMALINK: ", comment.permalink)
    