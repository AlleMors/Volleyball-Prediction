from scrapy.http import HtmlResponse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

class SeleniumMiddleware:
    def __init__(self):
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Esegui Chrome in modalità headless (opzionale)
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    def process_request(self, request, spider):
        self.driver.get(request.url)
        time.sleep(2)  # Attendere il caricamento della pagina
        body = self.driver.page_source
        return HtmlResponse(self.driver.current_url, body=body, encoding='utf-8', request=request)

    def close(self):
        self.driver.quit()
