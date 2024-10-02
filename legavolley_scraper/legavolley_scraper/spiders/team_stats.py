import scrapy
from scrapy.exceptions import CloseSpider


class TeamStatsSpider(scrapy.Spider):
    name = 'legavolley_stats'
    allowed_domains = ['legavolley.it']

    def __init__(self, serie='1', anno_inizio='2024', fase='1', giornata='0', *args, **kwargs):
        super(TeamStatsSpider, self).__init__(*args, **kwargs)
        self.serie = serie
        self.anno_inizio = anno_inizio
        self.fase = fase
        self.giornata = giornata

    def start_requests(self):
        # Prima richiesta per ottenere i nomi delle squadre
        url = f'https://www.legavolley.it/statistiche/?TipoStat=1.1&Serie={self.serie}&AnnoInizio={self.anno_inizio}&Fase={self.fase}&Giornata={self.giornata}'
        yield scrapy.Request(url=url, callback=self.parse_players)

    def parse_players(self, response):
        self.logger.info("Estrazione dei nomi delle squadre")

        # Estrai i nomi delle squadre
        teams = response.xpath('//*[@id="divframe"]/form/div[1]/div[6]/div[2]/ul/li/@data-value').getall()
        teams = [team.strip() for team in teams if team.strip()]  # Rimuovi spazi bianchi e filtra i nomi vuoti

        if not teams:
            self.logger.error("Nessuna squadra trovata. Controlla l'XPath.")
            raise CloseSpider("Nessuna squadra trovata")

        # Genera le richieste per ciascuna squadra
        for team in teams:
            url = f'https://www.legavolley.it/statistiche/?TipoStat=1.1&Serie={self.serie}&AnnoInizio={self.anno_inizio}&Fase={self.fase}&Giornata={self.giornata}&Squadra={team}'
            yield scrapy.Request(url=url, callback=self.parse, meta={'squadra': team})

    def parse(self, response):
        self.logger.info(f"Status code: {response.status}")

        squadra_nome=response.xpath('//*[@id="divframe"]/form/div[1]/div[6]/div[2]/span/text()').get()
        squadra_code = response.meta['squadra']
        table = response.xpath('//table[@id="Statistica"]//tr')

        if len(table) < 4:
            self.logger.error("La tabella non contiene abbastanza righe.")
            return

        players = []

        for row in table[2:-2]:  # Salta le prime 2 righe di intestazione e l'ultima riga
            atleta = row.xpath('td[1]/text()').get(default='').strip()  # Rimuove spazi bianchi

            if atleta and not atleta.isdigit() and "Legenda" not in atleta and "Totali di Squadra" not in atleta:
                stats = [row.xpath(f'td[{i}]/text()').get(default='') for i in range(2, 25)]
                if any(stat for stat in stats):  # Assicurati che almeno una statistica non sia null
                    player = {
                        'atleta': atleta,
                        'partite_giocate': row.xpath('td[2]/text()').get(default=''),
                        'set_giocati': row.xpath('td[3]/text()').get(default=''),
                        'punti_totali': row.xpath('td[4]/text()').get(default=''),
                        'punti_bp': row.xpath('td[5]/text()').get(default=''),
                        'battuta_totale': row.xpath('td[6]/text()').get(default=''),
                        'ace': row.xpath('td[7]/text()').get(default=''),
                        'errori_battuta': row.xpath('td[8]/text()').get(default=''),
                        'ace_per_set': row.xpath('td[9]/text()').get(default=''),
                        'battuta_efficienza': row.xpath('td[10]/text()').get(default=''),
                        'ricezione_totale': row.xpath('td[11]/text()').get(default=''),
                        'errori_ricezione': row.xpath('td[12]/text()').get(default=''),
                        'ricezione_negativa': row.xpath('td[13]/text()').get(default=''),
                        'ricezione_perfetta': row.xpath('td[14]/text()').get(default=''),
                        'ricezione_perfetta_perc': row.xpath('td[15]/text()').get(default=''),
                        'ricezione_efficienza': row.xpath('td[16]/text()').get(default=''),
                        'attacco_totale': row.xpath('td[17]/text()').get(default=''),
                        'errori_attacco': row.xpath('td[18]/text()').get(default=''),
                        'attacco_murati': row.xpath('td[19]/text()').get(default=''),
                        'attacco_perfetti': row.xpath('td[20]/text()').get(default=''),
                        'attacco_perfetti_perc': row.xpath('td[21]/text()').get(default=''),
                        'attacco_efficienza': row.xpath('td[22]/text()').get(default=''),
                        'muro_perfetti': row.xpath('td[23]/text()').get(default=''),
                        'muro_per_set': row.xpath('td[24]/text()').get(default='')
                    }
                    players.append(player)

        # Gestisci l'ultima riga dei totali di squadra, se presente

        totals_row = response.xpath('/html/body/div[7]/div[2]/div/div/form/table[2]/tbody/tr[12]')
        totals = {
            'partite_giocate': totals_row.xpath('td[1]/text()').get(default=''),
            'set_giocati': totals_row.xpath('td[2]/text()').get(default=''),
            'punti_totali': totals_row.xpath('td[3]/text()').get(default=''),
            'punti_bp': totals_row.xpath('td[4]/text()').get(default=''),
            'battuta_totale': totals_row.xpath('td[5]/text()').get(default=''),
            'ace': totals_row.xpath('td[6]/text()').get(default=''),
            'errori_battuta': totals_row.xpath('td[7]/text()').get(default=''),
            'ace_per_set': totals_row.xpath('td[8]/text()').get(default=''),
            'battuta_efficienza': totals_row.xpath('td[9]/text()').get(default=''),
            'ricezione_totale': totals_row.xpath('td[10]/text()').get(default=''),
            'errori_ricezione': totals_row.xpath('td[11]/text()').get(default=''),
            'ricezione_negativa': totals_row.xpath('td[12]/text()').get(default=''),
            'ricezione_perfetta': totals_row.xpath('td[13]/text()').get(default=''),
            'ricezione_perfetta_perc': totals_row.xpath('td[14]/text()').get(default=''),
            'ricezione_efficienza': totals_row.xpath('td[15]/text()').get(default=''),
            'attacco_totale': totals_row.xpath('td[16]/text()').get(default=''),
            'errori_attacco': totals_row.xpath('td[17]/text()').get(default=''),
            'attacco_murati': totals_row.xpath('td[18]/text()').get(default=''),
            'attacco_perfetti': totals_row.xpath('td[19]/text()').get(default=''),
            'attacco_perfetti_perc': totals_row.xpath('td[20]/text()').get(default=''),
            'attacco_efficienza': totals_row.xpath('td[21]/text()').get(default=''),
            'muro_perfetti': totals_row.xpath('td[22]/text()').get(default=''),
            'muro_per_set': totals_row.xpath('td[23]/text()').get(default='')
        }

        yield {
            'squadra': squadra_nome,
            'codice': squadra_code,
            'players': players,
            'totali': totals
        }
