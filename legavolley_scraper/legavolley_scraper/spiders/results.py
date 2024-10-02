import scrapy
import json


class ResultsSpider(scrapy.Spider):
    name = "results"
    allowed_domains = ["legavolley.it"]
    start_urls = ["https://legavolley.it"]

    def __init__(self):
        self.giornate = []

    def start_requests(self):
        # Prima richiesta per ottenere i nomi delle squadre
        url = f'https://www.legavolley.it/risultati/?IdCampionato=947'
        yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # Estrae i codici delle giornate dalla lista
        giornate = response.xpath('//*[@id="divframe"]/form/div/div[4]/div[2]/ul/li/@data-value').getall()

        # Cicla tra tutte le giornate e costruisce l'URL per ogni giornata
        for giornata in giornate:
            url = f'https://www.legavolley.it/risultati/?Anno=2024&IdCampionato=947&IdFase=1&IdGiornata={giornata}'
            yield scrapy.Request(url, callback=self.parse_giornata)

    def parse_giornata(self, response):
        results = []
        for i in range(4, 16):
            team = response.xpath(
                f'/html/body/div[7]/div[2]/div/div/table/tbody/tr[2]/td[1]/table/tbody/tr[{i}]/td[1]/text()').get()

            if i % 2 == 0:
                res = response.xpath(
                    f'/html/body/div[7]/div[2]/div/div/table/tbody/tr[2]/td[1]/table/tbody/tr[{i}]/td[2]/text()').get()
            else:
                res = None

            result = {
                'team': team,
                'result': res
            }
            results.append(result)

        giornata = response.xpath('/html/body/div[7]/div[2]/div/div/form/div/div[4]/div[2]/span/text()').get()

        # Check if "giornata" text and results are valuable before appending
        if giornata and results:
            giornata_data = {
                'giornata': giornata,
                'results': results
            }

            # Clean the data
            self.pulisci_dati(giornata_data)

            # Check if all results are "0-0"
            if not all(result['result'] == '0-0' for result in giornata_data['results']):
                self.giornate.append(giornata_data)

    def closed(self, reason):
        # Filtra eventuali elementi vuoti e ordina le giornate per numero crescente
        self.giornate = [g for g in self.giornate if g and g['results']]
        self.giornate.sort(key=self.get_giornata_number)

        # Rimuovi eventuali array vuoti alla fine della lista
        while self.giornate and isinstance(self.giornate[-1], list) and not self.giornate[-1]:
            self.giornate.pop()

        # Scriviamo tutti i risultati nel file JSON al termine dello spider
        with open('results.json', 'w') as file:
            json.dump(self.giornate, file, indent=4)

    def get_giornata_number(self, giornata):
        # Estrai il numero della giornata
        return int(giornata["giornata"].split()[0])

    def pulisci_dati(self, giornata):
        # Pulizia del campo giornata
        if "giornata" in giornata:
            giornata["giornata"] = giornata["giornata"].replace("\u00aa", "").strip()

        for i, result in enumerate(giornata["results"]):
            # Rimuovi '\n' e '\t' da 'team' e 'result'
            if "team" in result:
                result["team"] = result["team"].replace("\n", "").replace("\t", "").strip()
            if "result" in result and result["result"]:
                result["result"] = result["result"].replace("\n", "").replace("\t", "").strip()

            # Se `result` è None e la riga è dispari, inverti il risultato precedente
            if i % 2 == 1 and result["result"] is None:
                risultato_precedente = giornata["results"][i - 1]["result"]
                if risultato_precedente:
                    # Inversione del risultato
                    parts = risultato_precedente.split('-')
                    if len(parts) == 2:
                        result["result"] = f"{parts[1].strip()}-{parts[0].strip()}"
