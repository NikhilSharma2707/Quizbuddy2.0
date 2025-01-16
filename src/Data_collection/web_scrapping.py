import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import csv


def get_project_gutenberg_books(topic):
    url = f"https://www.gutenberg.org/ebooks/search/?query={topic}&submit_search=Go%21"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    books = []
    for book in soup.select('.booklink'):
        title = book.select_one('.title').text.strip()

        # Check for the presence of author element
        author_element = book.select_one('.subtitle')
        author = author_element.text.strip() if author_element else "Unknown Author"

        books.append({"title": title, "author": author})
    return books


def get_arxiv_papers(query, max_results=10):
    base_url = 'http://export.arxiv.org/api/query?'
    search_query = f'search_query=all:{query}&start=0&max_results={max_results}'
    response = requests.get(base_url + search_query)
    root = ET.fromstring(response.content)

    papers = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
        papers.append({"title": title, "summary": summary})
    return papers


def get_openstax_books():
    url = "https://openstax.org/apps/cms/api/v2/pages/?type=books.Book&fields=title,description&limit=100"
    response = requests.get(url)
    books = response.json()['items']
    return [{"title": book['title'], "description": book['description']} for book in books]


def save_to_csv(data, filename, fieldnames):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


# Example usage
gutenberg_books = get_project_gutenberg_books("mathematics")
save_to_csv(gutenberg_books, "gutenberg_math_books.csv", ["title", "author"])

arxiv_papers = get_arxiv_papers("quantum computing")
save_to_csv(arxiv_papers, "arxiv_quantum_computing_papers.csv", ["title", "summary"])

openstax_books = get_openstax_books()
save_to_csv(openstax_books, "openstax_books.csv", ["title", "description"])

print("Data has been saved to CSV files.")
