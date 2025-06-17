import requests
from itertools import cycle
import time
from bs4 import BeautifulSoup
import csv
import random

# Proxies
proxies = [
    'https://beqcfgqd:zx2ta8sl24bs@66.225.236.89:6118',
    'https://beqcfgqd:zx2ta8sl24bs@45.39.2.114:6546',
    'https://beqcfgqd:zx2ta8sl24bs@104.252.194.86:6995',
    'https://beqcfgqd:zx2ta8sl24bs@45.84.44.108:5586',
    'https://beqcfgqd:zx2ta8sl24bs@104.233.20.188:6204',
    'https://beqcfgqd:zx2ta8sl24bs@166.0.6.18:5979',
    'https://beqcfgqd:zx2ta8sl24bs@89.116.71.120:6086',
    'https://beqcfgqd:zx2ta8sl24bs@94.177.21.47:5416',
    'https://beqcfgqd:zx2ta8sl24bs@172.245.158.37:5990',
    'https://beqcfgqd:zx2ta8sl24bs@198.23.214.31:6298',
    'https://beqcfgqd:zx2ta8sl24bs@43.245.117.205:5789',
    'https://beqcfgqd:zx2ta8sl24bs@45.43.82.199:6193',
    'https://beqcfgqd:zx2ta8sl24bs@104.238.8.28:5886',
    'https://beqcfgqd:zx2ta8sl24bs@145.223.41.229:6500',
    'https://beqcfgqd:zx2ta8sl24bs@193.161.2.176:6599',
    'https://beqcfgqd:zx2ta8sl24bs@45.250.64.94:5731',
    'https://beqcfgqd:zx2ta8sl24bs@194.5.3.85:5597',
    'https://beqcfgqd:zx2ta8sl24bs@82.24.225.210:8051',
    'https://beqcfgqd:zx2ta8sl24bs@104.239.40.34:6653',
    'https://beqcfgqd:zx2ta8sl24bs@191.101.25.152:6549',
    'https://beqcfgqd:zx2ta8sl24bs@216.173.109.172:6403',
    'https://beqcfgqd:zx2ta8sl24bs@173.211.69.3:6596',
    'https://beqcfgqd:zx2ta8sl24bs@191.101.11.25:6423',
    'https://beqcfgqd:zx2ta8sl24bs@209.242.203.142:6857',
    'https://beqcfgqd:zx2ta8sl24bs@161.123.101.69:6695',
    'https://beqcfgqd:zx2ta8sl24bs@38.153.136.82:5105',
    'https://beqcfgqd:zx2ta8sl24bs@104.222.187.190:6314',
]
proxy_pool = cycle(proxies)

# Headers
headers = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "priority": "u=0, i",
    "sec-ch-ua": '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
    "sec-ch-ua-mobile": "?1",
    "sec-ch-ua-platform": '"Android"',
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Mobile Safari/537.36"
}

# Base URLs
base_url = "https://www.dva.gov.au"
listing_url = f"{base_url}/about/news/latest-news"

# Helper to fetch a page with proxy rotation
def fetch_request(url):
    proxy = next(proxy_pool)
    print(f"Fetching {url} using proxy {proxy}")
    try:
        response = requests.get(url, headers=headers, proxies={"http": proxy, "https": proxy}, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return None

# Extract data from article page
def scrape_article_page(full_url):
    html = fetch_request(full_url)
    if not html:
        return None, None

    soup = BeautifulSoup(html, 'html.parser')

    title_tag = soup.select_one("div.field.field--name-node-title h1")
    title = title_tag.get_text(strip=True) if title_tag else None

    body_divs = soup.select("div.clearfix.text-formatted.field.field--name-body.field--type-text-with-summary.field--label-hidden.field__item")
    content = body_divs[3].get_text(separator="\n", strip=True) if len(body_divs) >= 4 else None



    return title, content

# CSV setup
csv_file = 'DVA Website Latest News.csv'
csv_fields = ['url', 'title', 'content']
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=csv_fields)
    writer.writeheader()

    for page in range(112):
        print(f"\n========== Page {page} ==========")
        page_url = f"{listing_url}?page={page}"
        page_html = fetch_request(page_url)

        if not page_html:
            continue

        soup = BeautifulSoup(page_html, 'html.parser')
        cards = soup.find_all('a', class_='card', href=True)

        for card in cards:
            relative_href = card['href']
            full_url = base_url + relative_href
            print(f"→ Scraping article: {full_url}")

            title, content = scrape_article_page(full_url)
            if title and content:
                writer.writerow({
                    'url': full_url,
                    'title': title,
                    'content': content
                })
                print("✓ Saved")
            else:
                print("✗ Missing data")

            time.sleep(random.uniform(1, 2.5))  # be polite
