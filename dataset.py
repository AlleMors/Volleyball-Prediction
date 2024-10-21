import json

import numpy as np
import pandas as pd

from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
import logging

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


def aggregate_past_data(self, team_2):
    combined_data = []
    # Aggrega le statistiche passate dei titolari di entrambe le squadre in una singola riga per partita
    for team in [self, team_2]:
        # Inizializza un dizionario per sommare le statistiche per ogni giornata
        for giornata_index in range(len(team.starters[0].giornate)):
            giornata_totals = {'squadra': team.name, 'set_giocati': 0, 'punti_totali': 0, 'punti_bp': 0,
                               'battuta_totale': 0, 'ace': 0,
                               'errori_battuta': 0, 'ace_per_set': 0, 'battuta_efficienza': 0,
                               'ricezione_totale': 0, 'errori_ricezione': 0, 'ricezione_negativa': 0,
                               'ricezione_perfetta': 0, 'ricezione_perfetta_perc': 0,
                               'ricezione_efficienza': 0, 'attacco_totale': 0, 'errori_attacco': 0,
                               'attacco_murati': 0, 'attacco_perfetti': 0, 'attacco_perfetti_perc': 0,
                               'attacco_efficienza': 0, 'muro_perfetti': 0, 'muro_per_set': 0}

            # Itera attraverso ogni giocatore per quella giornata
            for player in team.starters:
                if giornata_index < len(player.giornate):
                    giornata = player.giornate[giornata_index]

                    # Somma le statistiche per quella giornata
                    for key in giornata_totals:
                        if key != 'squadra':  # Skip the 'squadra' key
                            giornata_totals[key] += giornata[key]
            # Calcola la media delle statistiche
            for key in giornata_totals:
                if key != 'squadra':  # Skip the 'squadra' key
                    giornata_totals[key] /= len(team.starters)  # Cambiato da 7 a len(team.starters)

            combined_data.append(giornata_totals)

    return combined_data


class Team:
    players = []
    totals = []

    def __init__(self, team_data):
        self.model = None
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

    def train_match_winner_model(self, team_2, test_sizes=[0.1, 0.2, 0.3, 0.4, 0.5]):
        combined_data = aggregate_past_data(self, team_2)

        # Crea il DataFrame per i dati combinati
        combined_data_df = pd.DataFrame(combined_data)

        # Aggiungi una colonna per identificare le squadre
        combined_data_df['is_team_1'] = combined_data_df['squadra'].apply(lambda x: 1 if x == self.name else 0)

        # Rimuovi la colonna 'squadra' originale
        combined_data_df = combined_data_df.drop(columns=['squadra'])

        # Gestisci i NaN
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        combined_data_df_imputed = pd.DataFrame(imputer.fit_transform(combined_data_df),
                                                columns=combined_data_df.columns)

        # Normalizzazione dei dati
        scaler = StandardScaler()
        combined_data_df_scaled = pd.DataFrame(scaler.fit_transform(combined_data_df_imputed),
                                               columns=combined_data_df_imputed.columns)

        # Prepara l'output Y: 1 per vittoria di self, 0 per vittoria di team_2
        outcome_map = {'3-0': 1, '3-1': 1, '3-2': 1, '2-3': 0, '1-3': 0, '0-3': 0}
        y = pd.Series([outcome_map[result] for result in self.results], name='Match_Outcome')
        y_team_2 = pd.Series([outcome_map[result] for result in team_2.results], name='Match_Outcome')

        # Sommare le due serie
        combined_y = pd.concat([y, y_team_2], ignore_index=True)

        # Inizializza il RandomForestClassifier
        rf = RandomForestClassifier(random_state=42)

        # Lista per salvare i risultati
        best_accuracy = 0
        best_test_size = None

        # Loop sulle diverse proporzioni test/train
        for test_size in test_sizes:
            # Dividi il dataset in training e test
            X_train, X_test, y_train, y_test = train_test_split(combined_data_df_scaled, combined_y,
                                                                test_size=test_size, random_state=42)

            # Check se ci sono abbastanza campioni per fare cross-validation
            min_class_count = y_train.value_counts().min()
            min_samples_threshold = 2

            if min_class_count < min_samples_threshold:
                # Addestra direttamente il modello se i campioni sono insufficienti
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
            else:
                # Usa StratifiedKFold per cross-validation
                n_splits = min(5, max(2, min_class_count))
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                scores = cross_val_score(rf, X_train, y_train, cv=skf, scoring='accuracy')
                accuracy = np.mean(scores)

            print(f"Test size {test_size}, Accuracy: {accuracy:.2f}")

            # Aggiorna la miglior proporzione
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_test_size = test_size

        print(f"La miglior proporzione test/train è {best_test_size} con un'accuratezza del {best_accuracy:.2f}")

        # Dopo aver trovato la proporzione migliore, usa quella per addestrare il modello finale
        X_train, X_test, y_train, y_test = train_test_split(combined_data_df_scaled, combined_y,
                                                            test_size=best_test_size, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        final_accuracy = accuracy_score(y_test, y_pred)
        print(f"Final accuracy con la miglior proporzione test/train: {final_accuracy:.2f}")

        # Predici il vincitore per la nuova partita
        winner = rf.predict(X_test)
        result = self.name if winner[0] == 1 else team_2.name
        print(f"Predicted winner: {result}")
        return rf


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

    team_1 = next(team for team in team_objects if team.name == "Sir Susa Vim Perugia")
    team_2 = next(team for team in team_objects if team.name == "Valsa Group Modena")

    modena_starters = ["Sanguinetti Giovanni", "Anzani Simone", "Davyskiba Vlad", "De Cecco Luciano", "Buchegger Paul",
                       "Rinaldi Tommaso", "Federici Filippo"]
    trento_starters = ["Garcia Fernandez Gabi", "Kozamernik Jan", "Laurenzano Gabriele", "Lavia Daniele",
                       "Michieletto Alessandro",
                       "Resende Gualberto Flavio", "Sbertoli Riccardo"]
    perugia_starters = ["Giannelli Simone", "Loser Agustin", "Ben Tara Wassim", "Russo Roberto", "Colaci Massimo",
                        "Ishikawa Yuki", "Semeniuk Kamil"]
    cisterna_starters = ["Baranowicz Michele", "Bayram Efe", "Faure Theo", "Nedeljkovic Aleksandar", "Pace Domenico",
                         "Ramon Jordi", "Mazzone Daniele"]

    team_1.select_starters(perugia_starters)
    team_2.select_starters(modena_starters)

    # Addestra il modello per predire il vincitore tra le due squadre
    model = team_1.train_match_winner_model(team_2)
