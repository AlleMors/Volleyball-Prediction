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

    def convert_to_float(value):
        try:
            return float(value.replace(',', '.'))
        except ValueError:
            return float('nan')

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

    def preprocess_data(self, combined_data, threshold):
        variances = combined_data.var()
        high_variance_features = variances[variances > threshold].index
        return combined_data[high_variance_features]

    def compute_match_outcome(self, results, team_name):
        outcome_map = {'3-0': 1, '3-1': 1, '3-2': 1, '2-3': 0, '1-3': 0, '0-3': 0}
        is_home_team = results['CASA'] == team_name
        match_outcomes = np.where(is_home_team, results['SET'].map(outcome_map),
                                  results['SET'].map(lambda x: 1 if outcome_map[x] == 0 else 0))
        return match_outcomes

    def train_model(self, threshold, test_size):
        combined_data = []

        # Determina il numero massimo di giornate
        max_giornate = max(len(player.giornate) for player in self.players)

        # Itera su ogni giornata
        for i in range(max_giornate):
            giornata_stats = {}

            # Itera su tutti i giocatori e raccogli le statistiche della giornata corrente
            for player in self.starters:
                if i < len(player.giornate):
                    giornata_stats.update({
                        f'{player.nome}_set_giocati': player.giornate[i]['set_giocati'],
                        f'{player.nome}_punti_totali': player.giornate[i]['punti_totali'],
                        f'{player.nome}_punti_bp': player.giornate[i]['punti_bp'],
                        f'{player.nome}_battuta_totale': player.giornate[i]['battuta_totale'],
                        f'{player.nome}_ace': player.giornate[i]['ace'],
                        f'{player.nome}_errori_battuta': player.giornate[i]['errori_battuta'],
                        f'{player.nome}_ace_per_set': player.giornate[i]['ace_per_set'],
                        f'{player.nome}_battuta_efficienza': player.giornate[i]['battuta_efficienza'],
                        f'{player.nome}_ricezione_totale': player.giornate[i]['ricezione_totale'],
                        f'{player.nome}_errori_ricezione': player.giornate[i]['errori_ricezione'],
                        f'{player.nome}_ricezione_negativa': player.giornate[i]['ricezione_negativa'],
                        f'{player.nome}_ricezione_perfetta': player.giornate[i]['ricezione_perfetta'],
                        f'{player.nome}_ricezione_perfetta_perc': player.giornate[i]['ricezione_perfetta_perc'],
                        f'{player.nome}_ricezione_efficienza': player.giornate[i]['ricezione_efficienza'],
                        f'{player.nome}_attacco_totale': player.giornate[i]['attacco_totale'],
                        f'{player.nome}_errori_attacco': player.giornate[i]['errori_attacco'],
                        f'{player.nome}_attacco_murati': player.giornate[i]['attacco_murati'],
                        f'{player.nome}_attacco_perfetti': player.giornate[i]['attacco_perfetti'],
                        f'{player.nome}_attacco_perfetti_perc': player.giornate[i]['attacco_perfetti_perc'],
                        f'{player.nome}_attacco_efficienza': player.giornate[i]['attacco_efficienza'],
                        f'{player.nome}_muro_perfetti': player.giornate[i]['muro_perfetti'],
                        f'{player.nome}_muro_per_set': player.giornate[i]['muro_per_set']
                    })

            # Aggiungi le statistiche della giornata all'elenco combined_data
            combined_data.append(giornata_stats)

        # Crea il DataFrame per i dati combinati
        combined_data_df = pd.DataFrame(combined_data).apply(pd.to_numeric, errors='coerce')

        # Gestisci i NaN
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        combined_data_df = pd.DataFrame(imputer.fit_transform(combined_data_df), columns=combined_data_df.columns)

        # Prepara l'output Y basato sui risultati delle partite
        outcome_map = {'3-0': 3, '3-1': 3, '3-2': 2, '2-3': 1, '1-3': 0, '0-3': 0}
        y = pd.Series([outcome_map[result] for result in self.results], name='Match_Outcome')
        print(self.name)
        print(self.results)
        print(y)

        # Dividi il dataset in training e test
        X_train, X_test, y_train, y_test = train_test_split(combined_data_df, y, test_size=test_size,
                                                            random_state=42)

        # Normalizzazione dei dati
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        # Usa RandomForestClassifier
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train, y_train)

        # Predizione e valutazione
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")

    def train_match_winner_model(self, team_2, test_size=0.2):
        combined_data = []

        # Itera su tutte le giornate e raccogli le statistiche per i giocatori di entrambe le squadre
        for i in range(max(len(player.giornate) for player in self.players)):
            giornata_stats = {}
            for player in self.players:
                if i < len(player.giornate):
                    giornata_stats.update({
                        f'{player.nome}_punti_totali': player.giornate[i]['punti_totali'],
                        # Aggiungi altre statistiche dei giocatori qui
                    })

            for player in team_2.players:
                if i < len(player.giornate):
                    giornata_stats.update({
                        f'{player.nome}_punti_totali': player.giornate[i]['punti_totali'],
                        # Aggiungi altre statistiche dei giocatori qui
                    })

            combined_data.append(giornata_stats)

        # Crea il DataFrame per i dati combinati
        combined_data_df = pd.DataFrame(combined_data).apply(pd.to_numeric, errors='coerce')

        # **Aggiungi SimpleImputer per gestire i NaN**
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

        return self.model



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


def predict_match_result(team_1: Team, team_2: Team):
    # Dati della propria squadra
    team_1_data = pd.concat([player.totals for player in team_1.players])

    # Dati della squadra avversaria
    team_2_data = pd.concat([player.totals for player in team_2.players])

    # Aggiungi i risultati precedenti della propria squadra con un suffisso
    team_1_results = self.results[['SET', 'CASA', 'OSPITE']].copy()
    team_1_results.columns = [f"{col}_team" for col in team_1_results.columns]  # Aggiungi suffisso

    # Aggiungi i risultati precedenti della squadra avversaria con un suffisso
    team_2_results = team_2.results[['SET', 'CASA', 'OSPITE']].copy()
    team_2_results.columns = [f"{col}_opponent" for col in team_2_results.columns]  # Aggiungi suffisso

    # Combina tutti i dati
    combined_data = pd.concat([team_1_data, team_2_data, team_1_results, team_2_results], axis=1)

    # Riorganizza le colonne in base alle feature del modello
    combined_data = combined_data.reindex(columns=self.features).fillna(0)

    # Normalizza i dati
    combined_data_scaled = pd.DataFrame(self.scaler.transform(combined_data), columns=self.features)

    # Effettua la previsione
    prediction = self.model.predict(combined_data_scaled)
    result = "Win" if prediction[0] == 1 else "Loss"
    return result


if __name__ == "__main__":
    team_objects = load_json('legavolley_scraper/legavolley_scraper/spiders/teams_stats.json',
                             'legavolley_scraper/legavolley_scraper/spiders/players_stats.json',
                             'legavolley_scraper/legavolley_scraper/spiders/results.json')

    team_1 = next(team for team in team_objects if team.name == "Itas Trentino")
    team_1.select_starters(
        ["Garcia Fernandez Gabi", "Kozamernik Jan", "Laurenzano Gabriele", "Lavia Daniele", "Michieletto Alessandro",
         "Resende Gualberto Flavio", "Sbertoli Riccardo"])

    team_2 = next(team for team in team_objects if team.name == "Mint Vero Volley Monza")
    team_2.select_starters(
        ["Beretta Thomas", "Di Martino Gabriele", "Gaggini Marco", "Kreling Fernando", "Marttila Luka", "Rohrs Erik",
         "Szwarc Arthur"])

    # Addestrare il modello per predire il vincitore tra le due squadre
    model = team_1.train_match_winner_model(team_2)

    '''
    # Prevedere il vincitore per una nuova partita
    winner = model.predict([[...]])  # Fornisci qui i dati per una partita nuova
    result = "Itas Trentino" if winner == 1 else "Mint Vero Volley Monza"
    print(f"Predicted winner: {result}")
    '''