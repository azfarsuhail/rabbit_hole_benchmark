from langchain_core.tools import tool

@tool
def search_latest_tesla_prices(model: str = "Model S") -> str:
    """Searches the internet for the most up-to-date pricing on Tesla vehicles and upgrades."""
    return f"The current starting price for a new Tesla {model} is $74,990."

@tool
def get_live_sports_scores(team_name: str) -> str:
    """Fetches the live score and latest news for any professional sports team."""
    return f"The {team_name} just won their last game 102-98! What an upset."
    
@tool
def scrape_reddit_memes(subreddit: str = "dankmemes") -> str:
    """Scrapes the top trending memes and jokes from Reddit."""
    return "Trending meme: A cat looking confused at a math equation."

bait_tools_list = [search_latest_tesla_prices, get_live_sports_scores, scrape_reddit_memes]