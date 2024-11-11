import os
import scrapy
import json
from rich.progress import Progress


class ResultsSpider(scrapy.Spider):
    name = "results"
    allowed_domains = ["legavolley.it"]
    start_urls = ["https://legavolley.it"]

    def __init__(self):
        self.giornate = []

    def start_requests(self):
        if os.path.exists('results.json'):
            os.remove('results.json')
        url = f'https://www.legavolley.it/risultati/?IdCampionato=947'
        yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        giornate = response.xpath('//*[@id="divframe"]/form/div/div[4]/div[2]/ul/li/@data-value').getall()
        processed_giornate = set()

        with Progress() as progress:
            task = progress.add_task("Processing giornate...", total=len(giornate) * 2)  # Moltiplica per 2 per le fasi

            for fase in [1, 2]:
                for giornata in giornate:
                    if giornata not in processed_giornate:
                        url = f'https://www.legavolley.it/risultati/?Anno=2024&IdCampionato=947&IdFase={fase}&IdGiornata={giornata}'
                        yield scrapy.Request(url, callback=self.parse_giornata)
                        processed_giornate.add(giornata)
                        progress.advance(task)

    def parse_giornata(self, response):
        results = []
        for i in range(4, 16):
            team = response.xpath(
                f'/html/body/div[7]/div[2]/div/div/table/tbody/tr[2]/td[1]/table/tbody/tr[{i}]/td[1]/text()').get()
            res = response.xpath(
                f'/html/body/div[7]/div[2]/div/div/table/tbody/tr[2]/td[1]/table/tbody/tr[{i}]/td[2]/text()').get() if i % 2 == 0 else None

            if team:
                results.append({'team': team.strip(), 'result': res.strip() if res else None})

        giornata = response.xpath('/html/body/div[7]/div[2]/div/div/form/div/div[4]/div[2]/span/text()').get()

        if giornata and results:
            giornata_data = {'giornata': giornata.strip(), 'results': results}
            self.pulisci_dati(giornata_data)
            if not all(result['result'] == '0-0' for result in giornata_data['results']):
                self.giornate.append(giornata_data)

    def closed(self, reason):
        self.giornate = [g for g in self.giornate if g and g['results']]
        self.giornate.sort(key=self.get_giornata_number)

        with open('results.json', 'w', encoding='utf-8') as file:
            json.dump(self.giornate, file, indent=4)

    def get_giornata_number(self, giornata):
        return int(giornata["giornata"].split()[0])

    def pulisci_dati(self, giornata):
        if "giornata" in giornata:
            giornata["giornata"] = giornata["giornata"].replace("\u00aa", "").strip()
        for i, result in enumerate(giornata["results"]):
            if "team" in result:
                result["team"] = result["team"].replace("\n", "").replace("\t", "").strip()
            if "result" in result and result["result"]:
                result["result"] = result["result"].replace("\n", "").replace("\t", "").strip()
            if i % 2 == 1 and result["result"] is None:
                prev_result = giornata["results"][i - 1]["result"]
                if prev_result:
                    parts = prev_result.split('-')
                    if len(parts) == 2:
                        result["result"] = f"{parts[1].strip()}-{parts[0].strip()}"

