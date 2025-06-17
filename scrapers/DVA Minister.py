import requests
from bs4 import BeautifulSoup
import csv
from itertools import cycle
import time

# ——— Proxy Configuration ———
proxies = [
    'http://beqcfgqd:zx2ta8sl24bs@85.204.255.7:6422',
    'http://beqcfgqd:zx2ta8sl24bs@64.43.90.225:6740',
    'http://beqcfgqd:zx2ta8sl24bs@82.153.248.29:5405',
    'http://beqcfgqd:zx2ta8sl24bs@145.223.51.147:6680',
    'http://beqcfgqd:zx2ta8sl24bs@150.107.202.103:6720',
    'http://beqcfgqd:zx2ta8sl24bs@172.245.158.37:5990',
]
proxy_pool = cycle(proxies)

# ——— Headers ———
headers = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "referer": "https://minister.dva.gov.au/minister-media-releases?page=1",
    "sec-ch-ua": '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
    "sec-ch-ua-mobile": "?1",
    "sec-ch-ua-platform": '"Android"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Mobile Safari/537.36"
}

# ——— URLs ———
BASE_URL = "https://minister.dva.gov.au"
PAGE_URL = BASE_URL + "/minister-media-releases?page={}"

# ——— Function to scrape article details ———
def scrape_article(full_url):
    proxy = next(proxy_pool)
    try:
        print("running")
        response = requests.get(full_url, headers=headers, proxies={"http": proxy, "https": proxy}, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title_tag = soup.find("h1", class_="au-header-heading")
        title = title_tag.get_text(strip=True) if title_tag else None

        content_blocks = soup.find_all("div", class_="field field--name-body field--type-text-with-summary field--label-hidden field__item")
        content = ""
        for block in content_blocks[1:]:  # skip the first, use from 2nd onward
            content += block.get_text(separator="\n", strip=True) + "\n"

        return title, content.strip()
    
    except Exception as e:
        print(f"Error scraping article: {full_url} - {e}")
        return None, None

# ——— Function to scrape listing page ———
def fetch_page_data(page_num):
    proxy = next(proxy_pool)
    url = PAGE_URL.format(page_num)
    try:
        response = requests.get(url, headers=headers, proxies={"http": proxy, "https": proxy}, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        articles = []
        blocks = soup.find_all('div', class_='media_release_listing--content-title')
        for block in blocks:
            a_tag = block.find('a', href=True)
            if a_tag:
                relative_url = a_tag['href']
                full_url = BASE_URL + relative_url
                print(f"Scraping article: {full_url}")
                title, content = scrape_article(full_url)
                if title and content:
                    articles.append({
                        'url': full_url,
                        'title': title,
                        'content': content
                    })
                    # print(f"url: {full_url}")
                    # print(f"title: {title}")
                    # print(f"content: {content}")


                time.sleep(1)  # be polite

        return articles
    except Exception as e:
        print(f"Error fetching page {page_num}: {e}")
        return []

# ——— CSV Saving ———
csv_filename = 'DVA Minister.csv'
csv_headers = ['url', 'title', 'content']

with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=csv_headers)
    writer.writeheader()

    for page in range(5):  # Adjust range as needed
        print(f"\n--- Fetching page {page} ---")
        page_articles = fetch_page_data(page)
        if page_articles:
            writer.writerows(page_articles)

print("\n✅ Done. All data saved to CSV.")
