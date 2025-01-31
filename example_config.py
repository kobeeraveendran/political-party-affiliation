def get_reddit_credentials():
    
    '''
    Input:
        - None

    Output:
        - client_id (string): the 14-character client ID of your Reddit API project (personal use script)
        - client_secret (string): the 27-character client secret of your Reddit API project
        - username (string): the username of the Reddit account used to register the project
        - password (string): the password of the Reddit account used to register the project
    '''

    user_agent = "user agent name goes here"
    client_id = "abc123def456gh"
    client_secret = "abcdefghijklmnopqrstuvwxyz5"
    username = "your reddit username"
    password = "your reddit password"

    credentials = {
        "user_agent": user_agent, 
        "client_id": client_id, 
        "client_secret": client_secret, 
        "username": username, 
        "password": password
    }

    return credentials