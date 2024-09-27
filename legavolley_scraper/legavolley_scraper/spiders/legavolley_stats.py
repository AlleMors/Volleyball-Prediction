import scrapy


class LegavolleyStats(scrapy.Spider):
    name = 'legavolley_stats'
    allowed_domains = ['legavolley.it']
    teams = []

    def __init__(self, serie='1', anno_inizio='2023', fase='1', giornata='0', *args, **kwargs):
        super(LegavolleyStats, self).__init__(*args, **kwargs)
        self.serie = serie
        self.anno_inizio = anno_inizio
        self.fase = fase
        self.giornata = giornata

    def start_requests(self):
        # Genera una richiesta per ogni squadra nell'elenco
        for team in self.teams:
            url = f'https://www.legavolley.it/statistiche/?TipoStat=1.1&Serie={self.serie}&AnnoInizio={self.anno_inizio}&Fase={self.fase}&Giornata={self.giornata}&Squadra={team}'
            yield scrapy.Request(url=url, callback=self.parse, meta={'squadra': team})

    def parse(self, response):
        # Seleziona tutte le squadre dall'elenco
        squadre = response.xpath('//*[@id="divframe"]/form/div[1]/div[6]/div[2]/ul/li')
        self.teams = []
        print(self.teams)

        for squadra in squadre:
            data_value = squadra.xpath('@data-value').get()  # Codice della squadra
            nome_squadra = squadra.xpath('text()').get()  # Nome visualizzato della squadra

            # Log per debugging
            self.logger.info(f'Trovata squadra: {nome_squadra} con codice: {data_value}')

            # Salta l'elemento "-- Selezionare un Club --"
            if data_value != "0" and data_value is not None:
                self.teams.append({'data_value': data_value, 'nome': nome_squadra.strip()})

        # Genera una richiesta per ogni squadra nell'elenco
        for team in self.teams:
            squadra_code = team['data_value']
            squadra_nome = team['nome']
            url = f'https://www.legavolley.it/statistiche/?TipoStat=1.1&Serie={self.serie}&AnnoInizio={self.anno_inizio}&Fase={self.fase}&Giornata={self.giornata}&Squadra={squadra_code}'
            yield scrapy.Request(url=url, callback=self.parse_stats, meta={'squadra_nome': squadra_nome})

    def parse_stats(self, response):
        squadra_nome = response.meta['squadra_nome']  # Recupera il nome della squadra dalle meta informazioni
        table = response.xpath('//table[@id="Statistica"]//tr')  # Individua la tabella con l'ID "Statistica"

        players = []
        print(players)

        # Estrai i dati per i giocatori
        for row in table[2:-3]:  # Salta le prime 2 righe di intestazione e l'ultima riga
            atleta = row.xpath('td[1]/text()').get().strip()  # Rimuove spazi bianchi

            # Verifica se l'atleta è un dato valido
            if atleta and not atleta.isdigit() and "Legenda" not in atleta and "Totali di Squadra" not in atleta:
                # Controlla se almeno alcune statistiche sono disponibili
                stats = [row.xpath(f'td[{i}]/text()').get() for i in range(2, 25)]
                if any(stat for stat in stats):  # Assicurati che almeno una statistica non sia null
                    player = {
                        'squadra': squadra_nome,  # Aggiungi il nome della squadra ai dati del giocatore
                        'atleta': atleta,
                        'partite_giocate': row.xpath('td[2]/text()').get(),
                        'set_giocati': row.xpath('td[3]/text()').get(),
                        'punti_totali': row.xpath('td[4]/text()').get(),
                        'punti_bp': row.xpath('td[5]/text()').get(),
                        'battuta_totale': row.xpath('td[6]/text()').get(),
                        'ace': row.xpath('td[7]/text()').get(),
                        'errori_battuta': row.xpath('td[8]/text()').get(),
                        'ace_per_set': row.xpath('td[9]/text()').get(),
                        'battuta_efficienza': row.xpath('td[10]/text()').get(),
                        'ricezione_totale': row.xpath('td[11]/text()').get(),
                        'errori_ricezione': row.xpath('td[12]/text()').get(),
                        'ricezione_negativa': row.xpath('td[13]/text()').get(),
                        'ricezione_perfetta': row.xpath('td[14]/text()').get(),
                        'ricezione_perfetta_perc': row.xpath('td[15]/text()').get(),
                        'ricezione_efficienza': row.xpath('td[16]/text()').get(),
                        'attacco_totale': row.xpath('td[17]/text()').get(),
                        'errori_attacco': row.xpath('td[18]/text()').get(),
                        'attacco_murati': row.xpath('td[19]/text()').get(),
                        'attacco_perfetti': row.xpath('td[20]/text()').get(),
                        'attacco_perfetti_perc': row.xpath('td[21]/text()').get(),
                        'attacco_efficienza': row.xpath('td[22]/text()').get(),
                        'muro_perfetti': row.xpath('td[23]/text()').get(),
                        'muro_per_set': row.xpath('td[24]/text()').get()
                    }
                    players.append(player)  # Corretto: aggiungi il giocatore alla lista

        # Ritorna il dizionario con i dati dei giocatori
        yield {
            'squadra': squadra_nome,
            'giocatori': players  # Corretto: utilizza la lista 'players'
        }

        # Gestisci l'ultima riga dei totali di squadra, se presente
        if len(table) > 3:  # Assicurati che ci siano almeno 4 righe
            totals_row = table[-4]  # Prendi l'ultima riga
            totals = {
                'squadra': squadra_nome,  # Aggiungi il nome della squadra ai dati dei totali
                'partite_giocate': totals_row.xpath('td[1]/text()').get(),
                'set_giocati': totals_row.xpath('td[2]/text()').get(),
                'punti_totali': totals_row.xpath('td[3]/text()').get(),
                'punti_bp': totals_row.xpath('td[4]/text()').get(),
                'battuta_totale': totals_row.xpath('td[5]/text()').get(),
                'ace': totals_row.xpath('td[6]/text()').get(),
                'errori_battuta': totals_row.xpath('td[7]/text()').get(),
                'ace_per_set': totals_row.xpath('td[8]/text()').get(),
                'battuta_efficienza': totals_row.xpath('td[9]/text()').get(),
                'ricezione_totale': totals_row.xpath('td[10]/text()').get(),
                'errori_ricezione': totals_row.xpath('td[11]/text()').get(),
                'ricezione_negativa': totals_row.xpath('td[12]/text()').get(),
                'ricezione_perfetta': totals_row.xpath('td[13]/text()').get(),
                'ricezione_perfetta_perc': totals_row.xpath('td[14]/text()').get(),
                'ricezione_efficienza': totals_row.xpath('td[15]/text()').get(),
                'attacco_totale': totals_row.xpath('td[16]/text()').get(),
                'errori_attacco': totals_row.xpath('td[17]/text()').get(),
                'attacco_murati': totals_row.xpath('td[18]/text()').get(),
                'attacco_perfetti': totals_row.xpath('td[19]/text()').get(),
                'attacco_perfetti_perc': totals_row.xpath('td[20]/text()').get(),
                'attacco_efficienza': totals_row.xpath('td[21]/text()').get(),
                'muro_perfetti': totals_row.xpath('td[22]/text()').get(),
                'muro_per_set': totals_row.xpath('td[23]/text()').get()
            }

            # Ritorna i totali della squadra
            yield {
                'squadra': squadra_nome,
                'totali': totals
            }
