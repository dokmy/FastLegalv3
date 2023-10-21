from bs4 import BeautifulSoup
import requests

url = "https://www.hklii.hk/en/cases/hkdc/"

def scrape_table(url):
    case_table = {}

    results = []
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "html.parser")

    text = soup.find_all("h1")

    print(text)

scrape_table(url)