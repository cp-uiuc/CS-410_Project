import requests
import time
import json

# Set your API key
api_key = 'c6Y7P1PHhZKFzuRM10fZd9WcUQuSTWTAOwalMY3C'

# Define endpoint and parameters
url = 'https://api.thenewsapi.com/v1/news/all'


# Create a list to store all articles
all_articles = []

for i in range(801):  # Capped at 20,000 (800 pages x 25 results/page) results by api
    params = {
        'api_token': api_key,
        'language': 'en',
        'search': '(president | presidential | election | campaign) & ("kamala harris" |  "donald trump")',
        'search_fields': 'title',
        'published_after': '2024-01-01',
        'published_before': '2024-11-05',
        'country': 'us',
        'page': i,
        'limit': 25
    }

    # Print current page being processed
    print(f"Fetching page {i}...")

    # Send request
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if 'data' in data and data['data']:
            all_articles.extend(data['data'])  # Add articles to the list
            # Print total articles retrieved so far
            print(f"Total articles retrieved so far: {len(all_articles)}")
        else:
            print("No more articles available.")
            break
    else:
        print(f"Error on page {i}: {response.status_code}, {response.text}")
        break


# Save all articles to a JSON file
with open('../../../data/test/raw/election_news.json', mode='w', encoding='utf-8') as file:
    json.dump(all_articles, file, indent=4)

print("Export complete! Data saved to 'election_news.json'.")
