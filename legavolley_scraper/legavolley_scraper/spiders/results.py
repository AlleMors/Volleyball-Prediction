import os
import scrapy
import json

class ResultsSpider(scrapy.Spider):
    name = "results"
    allowed_domains = ["legavolley.it"]
    start_urls = ["https://legavolley.it"]

    def __init__(self):
        self.giornate = []

    def start_requests(self):
        # Prima di fare le richieste, rimuoviamo il file 'results.json' se esiste
        if os.path.exists('results.json'):
            os.remove('results.json')

        # URL della fase di andata
        url_andata = f'https://www.legavolley.it/risultati/?Anno=2024&IdCampionato=947&IdFase=1'
        self.logger.info(f"Inviando richiesta per la fase 1: {url_andata}")  # Logging della fase 1
        yield scrapy.Request(url=url_andata, callback=self.parse, meta={'fase': 1})

        # URL della fase di ritorno
        url_ritorno = f'https://www.legavolley.it/risultati/?Anno=2024&IdCampionato=947&IdFase=2'
        self.logger.info(f"Inviando richiesta per la fase 2: {url_ritorno}")  # Logging della fase 2
        yield scrapy.Request(url=url_ritorno, callback=self.parse, meta={'fase': 2})

    def parse(self, response):
        fase = response.meta['fase']
        self.logger.info(f"Inizio elaborazione per la fase {fase}...")

        # Estrai tutte le giornate dalla pagina
        giornate = response.xpath('//*[@id="divframe"]/form/div/div[4]/div[2]/ul/li/@data-value').getall()
        self.logger.info(f"Giornate trovate per la fase {fase}: {giornate}")

        processed_giornate = set()

        for giornata in giornate:
            self.logger.info(f"Esaminando la giornata {giornata} della fase {fase}")

            if giornata not in processed_giornate:
                url = f'https://www.legavolley.it/risultati/?Anno=2024&IdCampionato=947&IdFase={fase}&IdGiornata={giornata}'
                yield scrapy.Request(url, callback=self.parse_giornata, meta={'giornata': giornata, 'fase': fase})
                processed_giornate.add(giornata)

    def parse_giornata(self, response):
        giornata = response.meta['giornata']
        fase = response.meta['fase']
        self.logger.info(f"Elaborazione dettagli per la giornata {giornata}, fase {fase}...")

        results = []
        for i in range(4, 16, 2):
            team = response.xpath(
                f'/html/body/div[7]/div[2]/div/div/table/tbody/tr[2]/td[1]/table/tbody/tr[{i}]/td[1]/text()').get()
            res = response.xpath(
                f'/html/body/div[7]/div[2]/div/div/table/tbody/tr[2]/td[1]/table/tbody/tr[{i}]/td[2]/text()').get()
            opponent = response.xpath(
                f'/html/body/div[7]/div[2]/div/div/table/tbody/tr[2]/td[1]/table/tbody/tr[{i + 1}]/td[1]/text()').get()

            if team:
                results.append(
                    {'team': team.strip(), 'result': res.strip(), 'opponent': opponent.strip() if res else None})

        giornata_nome = response.xpath('/html/body/div[7]/div[2]/div/div/form/div/div[4]/div[2]/span/text()').get()

        if giornata_nome and results:
            giornata_data = {'giornata': giornata_nome.strip(), 'results': results}
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
