import requests
from itertools import cycle
import time
from bs4 import BeautifulSoup
import csv
proxies_list = [
    '172.245.158.37:5990',
    '85.204.255.7:6422',
    '64.43.90.225:6740',
    '82.153.248.29:5405',
    '145.223.51.147:6680',
    '150.107.202.103:6720',
    '172.245.158.37:5990',

]

proxy_cycle = cycle(proxies_list)
requests_data = [
    {
        "url": "https://www.dva.gov.au/about/overview/repatriation-commission/gwen-cherne-veteran-family-advocate-commissioner/veteran-family-advocate-commissioner-gwen-cherne",
        "headers": {
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "max-age=0",
            "sec-ch-ua": "\"Google Chrome\";v=\"135\", \"Not-A.Brand\";v=\"8\", \"Chromium\";v=\"135\"",
            "sec-ch-ua-mobile": "?1",
            "sec-ch-ua-platform": "\"Android\"",
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "cross-site",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Mobile Safari/537.36"
        },
        "cookies": {
            "monsido": "4081743240673336",
            "_ga_XXXXXXXX": "GS1.1.1743507599.3.0.1743507599.0.0.0",
            "_gid": "GA1.3.1095878290.1743940223",
        }
    }]
session = requests.Session()
with open('DVA Repatriation Commission.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = [ 'Type','Name','URL','Content']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()  
    for req in requests_data:
        try:
            proxy = next(proxy_cycle)
            proxies = {"http": proxy, "https": proxy}

            if req.get("method") == "POST":
                response = session.post(
                    req["url"],
                    headers=req["headers"],
                    data=req.get("data"),
                    proxies=proxies,
                    timeout=10
                )
            else:
                response = session.get(
                    req["url"],
                    headers=req["headers"],
                    cookies=req.get("cookies", {}),
                    proxies=proxies,
                    timeout=10
                )

            print(f"Request to {req['url']} - Status Code: {response.status_code}")

            soup = BeautifulSoup(response.content, 'html.parser')
            main_wrapper = soup.find('div', id='main-wrapper')

            if main_wrapper:
                for heading in main_wrapper.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    heading_text = heading.get_text(strip=True)
                    writer.writerow({
                       
                        'Type': 'Heading',
                        'Name': heading_text,
                        'URL': '',
                        'Content': heading_text
                    })
                for link in main_wrapper.find_all('a', href=True):
                    link_text = link.get_text(strip=True)
                    link_url = link['href']
                    writer.writerow({
                      
                        'Type': 'Link',
                        'Name': link_text,
                        'URL': link_url,
                        'Content': link_text
                    })
                for img in main_wrapper.find_all('img', src=True):
                    img_url = img['src']
                    img_alt = img.get('alt', '')
                    writer.writerow({
                        
                        'Type': 'Image',
                        'Name': img_alt,
                        'URL': img_url,
                        'Content': img_alt
                    })
                text = main_wrapper.get_text(separator="\n", strip=True)
                if text:
                    writer.writerow({
                        
                        'Type': 'Text',
                        'Name': 'N/A',
                        'URL': '',
                        'Content': text
                    })

            else:
                print(f"No div with id='main-wrapper' found in {req['url']}.")

        except Exception as e:
            print(f"Error processing request to {req['url']}: {e}")
session.close()