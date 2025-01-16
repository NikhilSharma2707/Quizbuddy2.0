import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import time
from urllib.parse import urljoin
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeeksForGeeksScraper:
    def __init__(self):
        self.base_url = "https://www.geeksforgeeks.org"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.data = []

    def get_dsa_topics(self):
        """Get list of DSA topic pages"""
        dsa_urls = [
            f"{self.base_url}/data-structures/",
            f"{self.base_url}/algorithms/"
        ]

        topic_links = []
        for dsa_url in dsa_urls:
            try:
                logger.info(f"Fetching topics from {dsa_url}")
                response = requests.get(dsa_url, headers=self.headers)
                response.raise_for_status()  # Raise exception for bad status codes

                soup = BeautifulSoup(response.content, 'html.parser')

                # Look for links in multiple potential containers
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if any(term in href.lower() for term in ['/dsa-', '/data-structures', '/algorithms', '/tutorial']):
                        full_url = urljoin(self.base_url, href)
                        if full_url not in topic_links:
                            topic_links.append(full_url)
                            logger.info(f"Found topic URL: {full_url}")

            except Exception as e:
                logger.error(f"Error fetching topics from {dsa_url}: {str(e)}")

        logger.info(f"Total topic links found: {len(topic_links)}")
        return topic_links

    def scrape_article(self, url):
        """Scrape content from a single article"""
        try:
            logger.info(f"Scraping article: {url}")
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Try multiple possible title selectors
            title = None
            title_selectors = ['h1.article-title', 'h1.entry-title', 'h1.head']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.text.strip()
                    break

            if not title:
                logger.warning(f"No title found for {url}")
                return None

            # Try multiple possible content selectors
            content_selectors = ['article.content', 'div.article-text', 'div.entry-content']
            content_div = None
            for selector in content_selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    break

            if not content_div:
                logger.warning(f"No content div found for {url}")
                return None

            # Extract paragraphs while removing code blocks
            paragraphs = []
            for p in content_div.find_all(['p', 'li']):
                if not p.find('code') and not p.find('pre'):
                    text = p.text.strip()
                    if text and len(text) > 50:  # Filter out short snippets
                        paragraphs.append(text)

            if not paragraphs:
                logger.warning(f"No valid paragraphs found for {url}")
                return None

            content = ' '.join(paragraphs)

            logger.info(f"Successfully scraped article: {title}")
            return {
                'title': title,
                'url': url,
                'content': content
            }

        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None

    def scrape_all_topics(self, max_articles=50):
        """Scrape content from all DSA topics"""
        topic_urls = self.get_dsa_topics()
        logger.info(f"Beginning to scrape {len(topic_urls)} topics")

        for topic_url in topic_urls:
            try:
                logger.info(f"Processing topic: {topic_url}")
                response = requests.get(topic_url, headers=self.headers)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                article_links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if '/article/' in href or '/tutorial/' in href:
                        full_url = urljoin(self.base_url, href)
                        if full_url not in article_links:
                            article_links.append(full_url)

                logger.info(f"Found {len(article_links)} articles in topic {topic_url}")

                # Scrape individual articles
                for url in article_links[:max_articles]:
                    article_data = self.scrape_article(url)
                    if article_data:
                        self.data.append(article_data)
                        logger.info(f"Total articles scraped so far: {len(self.data)}")
                    time.sleep(2)  # Increased delay to be more conservative

            except Exception as e:
                logger.error(f"Error processing topic {topic_url}: {str(e)}")
                continue

    def save_data(self, json_file='gfg_dsa_data.json', excel_file='gfg_dsa_data.xlsx'):
        """Save scraped data to JSON and Excel files"""
        logger.info(f"Saving {len(self.data)} articles to files")

        # Save to JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON file: {json_file}")

        # Save to Excel
        if self.data:
            df = pd.DataFrame(self.data)
            df.to_excel(excel_file, index=False, engine='openpyxl')
            logger.info(f"Saved Excel file: {excel_file}")
        else:
            logger.warning("No data to save to Excel file")


def main():
    scraper = GeeksForGeeksScraper()
    scraper.scrape_all_topics(max_articles=50)
    scraper.save_data()


if __name__ == "__main__":
    main()