import json

import numpy as np
import pandas as pd
from IPython.core.macro import coding_declaration
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import Parallel, delayed
import warnings
import logging
import os

# Configuration for warnings and logging
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO)


class GiornataGiocatore:
    def __init__(self, giornata_data):
        self.giornata = pd.to_numeric(giornata_data['giornata'].replace(" ", "0"), errors='coerce')
        self.set_giocati = pd.to_numeric(giornata_data['set_giocati'].replace(" ", "0"), errors='coerce')
        self.punti_totali = pd.to_numeric(giornata_data['punti_totali'].replace(" ", "0"), errors='coerce')
        self.punti_bp = pd.to_numeric(giornata_data['punti_bp'].replace(" ", "0"), errors='coerce')
        self.battuta_totale = pd.to_numeric(giornata_data['battuta_totale'].replace(" ", "0"), errors='coerce')
        self.ace = pd.to_numeric(giornata_data['ace'].replace(" ", "0"), errors='coerce')
        self.errori_battuta = pd.to_numeric(giornata_data['errori_battuta'].replace(" ", "0"), errors='coerce')
        self.ace_per_set = pd.to_numeric(giornata_data['ace_per_set'].replace(" ", "0"), errors='coerce')
        self.battuta_efficienza = pd.to_numeric(giornata_data['battuta_efficienza'].replace(" ", "0"), errors='coerce')
        self.ricezione_totale = pd.to_numeric(giornata_data['ricezione_totale'].replace(" ", "0"), errors='coerce')
        self.errori_ricezione = pd.to_numeric(giornata_data['errori_ricezione'].replace(" ", "0"), errors='coerce')
        self.ricezione_negativa = pd.to_numeric(giornata_data['ricezione_negativa'].replace(" ", "0"), errors='coerce')
        self.ricezione_perfetta = pd.to_numeric(giornata_data['ricezione_perfetta'].replace(" ", "0"), errors='coerce')
        self.ricezione_perfetta_perc = pd.to_numeric(giornata_data['ricezione_perfetta_perc'].replace(" ", "0"),
                                                     errors='coerce')
        self.ricezione_efficienza = pd.to_numeric(giornata_data['ricezione_efficienza'].replace(" ", "0"),
                                                  errors='coerce')
        self.attacco_totale = pd.to_numeric(giornata_data['attacco_totale'].replace(" ", "0"), errors='coerce')
        self.errori_attacco = pd.to_numeric(giornata_data['errori_attacco'].replace(" ", "0"), errors='coerce')
        self.attacco_murati = pd.to_numeric(giornata_data['attacco_murati'].replace(" ", "0"), errors='coerce')
        self.attacco_perfetti = pd.to_numeric(giornata_data['attacco_perfetti'].replace(" ", "0"), errors='coerce')
        self.attacco_perfetti_perc = pd.to_numeric(giornata_data['attacco_perfetti_perc'].replace(" ", "0"),
                                                   errors='coerce')
        self.attacco_efficienza = pd.to_numeric(giornata_data['attacco_efficienza'].replace(" ", "0"), errors='coerce')
        self.muro_perfetti = pd.to_numeric(giornata_data['muro_perfetti'].replace(" ", "0"), errors='coerce')
        self.muro_per_set = pd.to_numeric(giornata_data['muro_per_set'].replace(" ", "0"), errors='coerce')


class Player:
    giornate = []
    totals = []
    averages = []

    def __init__(self, player_data):
        self.nome = player_data['atleta']
        self.partite_giocate = pd.to_numeric(player_data['partite_giocate'].replace(" ", "0"), errors='coerce')
        self.set_giocati = pd.to_numeric(player_data['set_giocati'].replace(" ", "0"), errors='coerce')
        self.punti_totali = pd.to_numeric(player_data['punti_totali'].replace(" ", "0"), errors='coerce')
        self.punti_bp = pd.to_numeric(player_data['punti_bp'].replace(" ", "0"), errors='coerce')
        self.battuta_totale = pd.to_numeric(player_data['battuta_totale'].replace(" ", "0"), errors='coerce')
        self.ace = pd.to_numeric(player_data['ace'].replace(" ", "0"), errors='coerce')
        self.errori_battuta = pd.to_numeric(player_data['errori_battuta'].replace(" ", "0"), errors='coerce')
        self.ace_per_set = pd.to_numeric(player_data['ace_per_set'].replace(" ", "0"), errors='coerce')
        self.battuta_efficienza = pd.to_numeric(player_data['battuta_efficienza'].replace(" ", "0"), errors='coerce')
        self.ricezione_totale = pd.to_numeric(player_data['ricezione_totale'].replace(" ", "0"), errors='coerce')
        self.errori_ricezione = pd.to_numeric(player_data['errori_ricezione'].replace(" ", "0"), errors='coerce')
        self.ricezione_negativa = pd.to_numeric(player_data['ricezione_negativa'].replace(" ", "0"), errors='coerce')
        self.ricezione_perfetta = pd.to_numeric(player_data['ricezione_perfetta'].replace(" ", "0"), errors='coerce')
        self.ricezione_perfetta_perc = pd.to_numeric(player_data['ricezione_perfetta_perc'].replace(" ", "0"),
                                                     errors='coerce')
        self.ricezione_efficienza = pd.to_numeric(player_data['ricezione_efficienza'].replace(" ", "0"),
                                                  errors='coerce')
        self.attacco_totale = pd.to_numeric(player_data['attacco_totale'].replace(" ", "0"), errors='coerce')
        self.errori_attacco = pd.to_numeric(player_data['errori_attacco'].replace(" ", "0"), errors='coerce')
        self.attacco_murati = pd.to_numeric(player_data['attacco_murati'].replace(" ", "0"), errors='coerce')
        self.attacco_perfetti = pd.to_numeric(player_data['attacco_perfetti'].replace(" ", "0"), errors='coerce')
        self.attacco_perfetti_perc = pd.to_numeric(player_data['attacco_perfetti_perc'].replace(" ", "0"),
                                                   errors='coerce')
        self.attacco_efficienza = pd.to_numeric(player_data['attacco_efficienza'].replace(" ", "0"), errors='coerce')
        self.muro_perfetti = pd.to_numeric(player_data['muro_perfetti'].replace(" ", "0"), errors='coerce')
        self.muro_per_set = pd.to_numeric(player_data['muro_per_set'].replace(" ", "0"), errors='coerce')


class Team:
    players = []
    totals = []

    def __init__(self, team_data):
        self.results = []
        self.name = team_data['squadra']
        self.codename = team_data['codice']
        self.players = [Player(player) for player in team_data['players']]
        self.starters = []
        self.totals = convert_to_numeric(team_data['totali'])

    def select_starters(self, starter_names):
        self.starters = [player for player in self.players if player.nome in starter_names]

        if len(self.starters) != 7:
            raise ValueError("Devi selezionare esattamente 7 titolari.")

    def load_results(self, results_data):
        for giornata in results_data:
            for result in giornata['results']:
                if result['team'] == self.name:
                    self.results.append(result['result'])

    def compute_points(self, results_path):
        with open(results_path, 'r') as file:
            results_data = json.load(file)

        points_map = {'3-0': 3, '3-1': 3, '3-2': 2, '2-3': 1, '1-3': 0, '0-3': 0}

        points = []
        for result in results_data:
            for match in result['results']:
                if match['team'] == self.name:
                    match_points = points_map[match['result']]
                    points.append(match_points)

        return points

    def find_player_by_name(self, name):
        for player in self.players:
            if player.nome == name:
                return player
        return None

    def train_match_winner_model(self, team_2, test_size=0.2):
        combined_data = []

        # Aggrega le statistiche passate dei titolari di entrambe le squadre in una singola riga per partita
        for i in range(max(len(player.giornate) for player in self.players)):
            match_stats = {}
            for player in self.starters:
                match_stats.update(aggregate_past_stats(player))

            for player in team_2.starters:
                match_stats.update(aggregate_past_stats(player))

            combined_data.append(match_stats)

        # Crea il DataFrame per i dati combinati
        combined_data_df = pd.DataFrame(combined_data).apply(pd.to_numeric, errors='coerce')

        # Gestisci i NaN
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        combined_data_df = pd.DataFrame(imputer.fit_transform(combined_data_df), columns=combined_data_df.columns)

        # Prepara l'output Y: 1 per vittoria di self, 0 per vittoria di team_2
        outcome_map = {'3-0': 1, '3-1': 1, '3-2': 1, '2-3': 0, '1-3': 0, '0-3': 0}
        y = pd.Series([outcome_map[result] for result in self.results], name='Match_Outcome')

        # Dividi il dataset in training e test
        X_train, X_test, y_train, y_test = train_test_split(combined_data_df, y, test_size=test_size, random_state=42)

        # Normalizzazione dei dati
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        # Usa RandomForestClassifier per prevedere chi vincerà
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train, y_train)

        # Predizione e valutazione
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")

        # Prepara i dati aggregati direttamente per la predizione
        new_match_stats = {}

        # Usa direttamente le statistiche già aggregate
        for player in self.starters:
            new_match_stats.update(aggregate_past_stats(player))

        for player in team_2.starters:
            new_match_stats.update(aggregate_past_stats(player))

        # Convert to DataFrame and ensure the feature names match
        new_match_stats_df = pd.DataFrame([new_match_stats], columns=combined_data_df.columns)

        # Predire il vincitore per la nuova partita
        winner = self.model.predict(new_match_stats_df)  # L'input deve essere 2D
        result = team_1.name if winner == 1 else team_2.name
        print(f"Predicted winner: {result}")

        return self.model


def aggregate_past_stats(player):
    """Aggrega le statistiche passate di un giocatore."""
    aggregated_stats = {}

    # Prendi la media delle statistiche su tutte le giornate giocate
    if player.giornate:
        aggregated_stats['set_giocati'] = np.mean([giornata['set_giocati'] for giornata in player.giornate])
        aggregated_stats['punti_totali'] = np.mean([giornata['punti_totali'] for giornata in player.giornate])
        aggregated_stats['punti_bp'] = np.mean([giornata['punti_bp'] for giornata in player.giornate])
        aggregated_stats['battuta_totale'] = np.mean([giornata['battuta_totale'] for giornata in player.giornate])
        aggregated_stats['ace'] = np.mean([giornata['ace'] for giornata in player.giornate])
        aggregated_stats['errori_battuta'] = np.mean([giornata['errori_battuta'] for giornata in player.giornate])
        aggregated_stats['ace_per_set'] = np.mean([giornata['ace_per_set'] for giornata in player.giornate])
        aggregated_stats['battuta_efficienza'] = np.mean(
            [giornata['battuta_efficienza'] for giornata in player.giornate])
        aggregated_stats['ricezione_totale'] = np.mean([giornata['ricezione_totale'] for giornata in player.giornate])
        aggregated_stats['errori_ricezione'] = np.mean([giornata['errori_ricezione'] for giornata in player.giornate])
        aggregated_stats['ricezione_negativa'] = np.mean(
            [giornata['ricezione_negativa'] for giornata in player.giornate])
        aggregated_stats['ricezione_perfetta'] = np.mean(
            [giornata['ricezione_perfetta'] for giornata in player.giornate])
        aggregated_stats['ricezione_perfetta_perc'] = np.mean(
            [giornata['ricezione_perfetta_perc'] for giornata in player.giornate])
        aggregated_stats['ricezione_efficienza'] = np.mean(
            [giornata['ricezione_efficienza'] for giornata in player.giornate])
        aggregated_stats['attacco_totale'] = np.mean([giornata['attacco_totale'] for giornata in player.giornate])
        aggregated_stats['errori_attacco'] = np.mean([giornata['errori_attacco'] for giornata in player.giornate])
        aggregated_stats['attacco_murati'] = np.mean([giornata['attacco_murati'] for giornata in player.giornate])
        aggregated_stats['attacco_perfetti'] = np.mean([giornata['attacco_perfetti'] for giornata in player.giornate])
        aggregated_stats['attacco_perfetti_perc'] = np.mean(
            [giornata['attacco_perfetti_perc'] for giornata in player.giornate])
        aggregated_stats['attacco_efficienza'] = np.mean(
            [giornata['attacco_efficienza'] for giornata in player.giornate])
        aggregated_stats['muro_perfetti'] = np.mean([giornata['muro_perfetti'] for giornata in player.giornate])
        aggregated_stats['muro_per_set'] = np.mean([giornata['muro_per_set'] for giornata in player.giornate])

    else:
        aggregated_stats['set_giocati'] = 0
        aggregated_stats['punti_totali'] = 0
        aggregated_stats['punti_bp'] = 0
        aggregated_stats['battuta_totale'] = 0
        aggregated_stats['ace'] = 0
        aggregated_stats['errori_battuta'] = 0
        aggregated_stats['ace_per_set'] = 0
        aggregated_stats['battuta_efficienza'] = 0
        aggregated_stats['ricezione_totale'] = 0
        aggregated_stats['errori_ricezione'] = 0
        aggregated_stats['ricezione_negativa'] = 0
        aggregated_stats['ricezione_perfetta'] = 0
        aggregated_stats['ricezione_perfetta_perc'] = 0
        aggregated_stats['ricezione_efficienza'] = 0
        aggregated_stats['attacco_totale'] = 0
        aggregated_stats['errori_attacco'] = 0
        aggregated_stats['attacco_murati'] = 0
        aggregated_stats['attacco_perfetti'] = 0
        aggregated_stats['attacco_perfetti_perc'] = 0
        aggregated_stats['attacco_efficienza'] = 0
        aggregated_stats['muro_perfetti'] = 0
        aggregated_stats['muro_per_set'] = 0

    return aggregated_stats


def convert_to_numeric(data):
    for key, value in data.items():
        if isinstance(value, str) and value.strip() == "":
            data[key] = 0
        else:
            try:
                data[key] = float(value.replace(",", "."))
            except (ValueError, AttributeError):
                pass
    return data


def load_json(teams_file_path, players_file_path, results_file_path):
    teams = []
    file_teams = json.load(open(teams_file_path))
    for data in file_teams:
        teams.append(Team(data))

    file_players = json.load(open(players_file_path))
    for data in file_players:
        data['totals'] = convert_to_numeric(data['totals'])
        data['averages'] = convert_to_numeric(data['averages'])
        for giornata in data['giornate']:
            convert_to_numeric(giornata)
        for team in teams:
            found_player = team.find_player_by_name(data['atleta'])
            if found_player:
                found_player.giornate = data['giornate']
                found_player.totals = data['totals']
                found_player.averages = data['averages']

    results_data = json.load(open(results_file_path))
    for team in teams:
        team.load_results(results_data)

    return teams


if __name__ == "__main__":
    team_objects = load_json('legavolley_scraper/legavolley_scraper/spiders/teams_stats.json',
                             'legavolley_scraper/legavolley_scraper/spiders/players_stats.json',
                             'legavolley_scraper/legavolley_scraper/spiders/results.json')

    team_2 = next(team for team in team_objects if team.name == "Sir Susa Vim Perugia")
    team_1 = next(team for team in team_objects if team.name == "Cisterna Volley")

    modena_starters = ["Sanguinetti Giovanni", "Anzani Simone", "Davyskiba Vlad", "De Cecco Luciano", "Buchegger Paul",
                       "Rinaldi Tommaso", "Federici Filippo"]
    trento_starters = ["Garcia Fernandez Gabi", "Kozamernik Jan", "Laurenzano Gabriele", "Lavia Daniele",
                       "Michieletto Alessandro",
                       "Resende Gualberto Flavio", "Sbertoli Riccardo"]
    perugia_starters = ["Giannelli Simone", "Loser Agustin", "Ben Tara Wassim", "Russo Roberto", "Colaci Massimo",
                        "Ishikawa Yuki", "Semeniuk Kamil"]
    cisterna_starters = ["Baranowicz Michele", "Bayram Efe", "Faure Theo", "Nedeljkovic Aleksandar", "Pace Domenico",
                         "Ramon Jordi", "Mazzone Daniele"]

    team_2.select_starters(perugia_starters)
    team_1.select_starters(cisterna_starters)

    # Addestra il modello per predire il vincitore tra le due squadre
    model = team_1.train_match_winner_model(team_2)
