import scrapy
from scrapy.exceptions import CloseSpider


class PlayersStatsSpider(scrapy.Spider):
    name = 'players_stats'
    allowed_domains = ['legavolley.it']

    def __init__(self, serie='1', anno_inizio='2024', fase='1', giornata='0', *args, **kwargs):
        super(PlayersStatsSpider, self).__init__(*args, **kwargs)
        self.serie = serie
        self.anno_inizio = anno_inizio
        self.fase = fase
        self.giornata = giornata

    def start_requests(self):
        # Prima richiesta per ottenere i nomi delle squadre
        url = f'https://www.legavolley.it/statistiche/?TipoStat=2.2&Serie={self.serie}&AnnoInizio={self.anno_inizio}&Fase={self.fase}&Giornata={self.giornata}'
        yield scrapy.Request(url=url, callback=self.parse_teams)

    def parse_teams(self, response):
        self.logger.info("Estrazione dei nomi delle squadre")

        # Estrai i nomi delle squadre
        players = response.xpath('//*[@id="divframe"]/form/div[1]/div[6]/div[2]/ul/li/@data-value').getall()
        players = [player.strip() for player in players if
                   player.strip()]  # Rimuovi spazi bianchi e filtra i nomi vuoti

        if not players:
            self.logger.error("Nessuna squadra trovata. Controlla l'XPath.")
            raise CloseSpider("Nessuna squadra trovata")

        # Genera le richieste per ciascuna squadra
        for player in players:
            url = f'https://www.legavolley.it/statistiche/?TipoStat=2.2&Serie={self.serie}&AnnoInizio={self.anno_inizio}&Fase={self.fase}&Giornata={self.giornata}&Atleta={player}'
            yield scrapy.Request(url=url, callback=self.parse, meta={'Atleta': player})

    def parse(self, response):
        self.logger.info(f"Status code: {response.status}")

        atleta_nome = response.xpath('//*[@id="divframe"]/form/div[1]/div[6]/div[2]/span/text()').get()
        atleta_code = response.meta['Atleta']
        table = response.xpath('/html/body/div[7]/div[2]/div/div/form/table[2]')

        if len(table) < 4:
            self.logger.error("La tabella non contiene abbastanza righe.")
            return

        giornate = []

        for row in table[2:-2]:  # Salta le prime 2 righe di intestazione e l'ultima riga
            giornata = row.xpath('td[1]/text()').get(default='').strip()  # Rimuove spazi bianchi

            if giornata and not giornata.isdigit() and "Legenda" not in giornata and "Totali di Squadra" not in giornata:
                stats = [row.xpath(f'td[{i}]/text()').get(default='') for i in range(2, 25)]
                if any(stat for stat in stats):  # Assicurati che almeno una statistica non sia null
                    giornata = {
                        'giornata': giornata,
                        'set_giocati': row.xpath('td[2]/text()').get(default=''),
                        'punti_totali': row.xpath('td[3]/text()').get(default=''),
                        'punti_bp': row.xpath('td[4]/text()').get(default=''),
                        'battuta_totale': row.xpath('td[5]/text()').get(default=''),
                        'ace': row.xpath('td[6]/text()').get(default=''),
                        'errori_battuta': row.xpath('td[7]/text()').get(default=''),
                        'ace_per_set': row.xpath('td[8]/text()').get(default=''),
                        'battuta_efficienza': row.xpath('td[9]/text()').get(default=''),
                        'ricezione_totale': row.xpath('td[10]/text()').get(default=''),
                        'errori_ricezione': row.xpath('td[11]/text()').get(default=''),
                        'ricezione_negativa': row.xpath('td[12]/text()').get(default=''),
                        'ricezione_perfetta': row.xpath('td[13]/text()').get(default=''),
                        'ricezione_perfetta_perc': row.xpath('td[14]/text()').get(default=''),
                        'ricezione_efficienza': row.xpath('td[15]/text()').get(default=''),
                        'attacco_totale': row.xpath('td[16]/text()').get(default=''),
                        'errori_attacco': row.xpath('td[17]/text()').get(default=''),
                        'attacco_murati': row.xpath('td[18]/text()').get(default=''),
                        'attacco_perfetti': row.xpath('td[19]/text()').get(default=''),
                        'attacco_perfetti_perc': row.xpath('td[20]/text()').get(default=''),
                        'attacco_efficienza': row.xpath('td[21]/text()').get(default=''),
                        'muro_perfetti': row.xpath('td[22]/text()').get(default=''),
                        'muro_per_set': row.xpath('td[23]/text()').get(default='')
                    }
                    giornate.append(giornata)

        totals_row = table[-2]
        totals = {
            'set_giocati': totals_row.xpath('td[1]/text()').get(default=''),
            'punti_totali': totals_row.xpath('td[2]/text()').get(default=''),
            'punti_bp': totals_row.xpath('td[3]/text()').get(default=''),
            'battuta_totale': totals_row.xpath('td[4]/text()').get(default=''),
            'ace': totals_row.xpath('td[5]/text()').get(default=''),
            'errori_battuta': totals_row.xpath('td[6]/text()').get(default=''),
            'ace_per_set': totals_row.xpath('td[7]/text()').get(default=''),
            'battuta_efficienza': totals_row.xpath('td[8]/text()').get(default=''),
            'ricezione_totale': totals_row.xpath('td[9]/text()').get(default=''),
            'errori_ricezione': totals_row.xpath('td[10]/text()').get(default=''),
            'ricezione_negativa': totals_row.xpath('td[11]/text()').get(default=''),
            'ricezione_perfetta': totals_row.xpath('td[12]/text()').get(default=''),
            'ricezione_perfetta_perc': totals_row.xpath('td[13]/text()').get(default=''),
            'ricezione_efficienza': totals_row.xpath('td[14]/text()').get(default=''),
            'attacco_totale': totals_row.xpath('td[15]/text()').get(default=''),
            'errori_attacco': totals_row.xpath('td[16]/text()').get(default=''),
            'attacco_murati': totals_row.xpath('td[17]/text()').get(default=''),
            'attacco_perfetti': totals_row.xpath('td[18]/text()').get(default=''),
            'attacco_perfetti_perc': totals_row.xpath('td[19]/text()').get(default=''),
            'attacco_efficienza': totals_row.xpath('td[20]/text()').get(default=''),
            'muro_perfetti': totals_row.xpath('td[21]/text()').get(default=''),
            'muro_per_set': totals_row.xpath('td[22]/text()').get(default='')
        }

        averages_row = table[-1]
        averages = {
            'set_giocati': averages_row.xpath('td[1]/text()').get(default=''),
            'punti_totali': averages_row.xpath('td[2]/text()').get(default=''),
            'punti_bp': averages_row.xpath('td[3]/text()').get(default=''),
            'battuta_totale': averages_row.xpath('td[4]/text()').get(default=''),
            'ace': averages_row.xpath('td[5]/text()').get(default=''),
            'errori_battuta': averages_row.xpath('td[6]/text()').get(default=''),
            'ace_per_set': averages_row.xpath('td[7]/text()').get(default=''),
            'battuta_efficienza': averages_row.xpath('td[8]/text()').get(default=''),
            'ricezione_totale': averages_row.xpath('td[9]/text()').get(default=''),
            'errori_ricezione': averages_row.xpath('td[10]/text()').get(default=''),
            'ricezione_negativa': averages_row.xpath('td[11]/text()').get(default=''),
            'ricezione_perfetta': averages_row.xpath('td[12]/text()').get(default=''),
            'ricezione_perfetta_perc': averages_row.xpath('td[13]/text()').get(default=''),
            'ricezione_efficienza': averages_row.xpath('td[14]/text()').get(default=''),
            'attacco_totale': averages_row.xpath('td[15]/text()').get(default=''),
            'errori_attacco': averages_row.xpath('td[16]/text()').get(default=''),
            'attacco_murati': averages_row.xpath('td[17]/text()').get(default=''),
            'attacco_perfetti': averages_row.xpath('td[18]/text()').get(default=''),
            'attacco_perfetti_perc': averages_row.xpath('td[19]/text()').get(default=''),
            'attacco_efficienza': averages_row.xpath('td[20]/text()').get(default=''),
            'muro_perfetti': averages_row.xpath('td[21]/text()').get(default=''),
            'muro_per_set': averages_row.xpath('td[22]/text()').get(default='')
        }

        yield {
            'atleta': atleta_nome,
            'codice': atleta_code,
            'giornate': giornate,
            'totals': totals,
            'averages': averages
        }
