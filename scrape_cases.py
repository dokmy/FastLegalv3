import requests
from bs4 import BeautifulSoup

# Define the URL of the webpage
url = 'https://www.hklii.hk/en/cases/hkdc/'

# Make a GET request to fetch the raw HTML content
response = requests.get(url)
response.raise_for_status()  # Check if the request was successful
print(response.text[:100000])  # print the first 1000 characters of the page
print(response.status_code)

# Parse the HTML with BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Extract all <a> tags with the specified data attribute
a_tags = soup.find_all('a', attrs={'data-v-1b9257e4': ''})

# Create a list of href links and text content
links_texts = [(a['href'], a.get_text()) for a in a_tags]

# Print the results
for link, text in links_texts:
    print(f"Link: {link}")
    print(f"Text: {text}")
    print("-" * 50)

# If you want to save the results to a file, you can uncomment the following lines:
# with open("output.txt", "w") as file:
#     for link, text in links_texts:
#         file.write(f"Link: {link}\n")
#         file.write(f"Text: {text}\n")
#         file.write("-" * 50 + "\n")
