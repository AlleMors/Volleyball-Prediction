import json

import numpy as np
import pandas as pd
from IPython.core.macro import coding_declaration
from sklearn.ensemble import RandomForestClassifier
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
        self.players = [Player(player) for player in team_data['players']]  # Converti in istanze Player
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
        combined_data = pd.DataFrame([player.__dict__ for player in self.starters])
        print(combined_data)

        # Genera l'outcome della partita (vittoria/sconfitta)
        outcome_map = {'3-0': 1, '3-1': 1, '3-2': 1, '2-3': 0, '1-3': 0, '0-3': 0}
        y = pd.Series([outcome_map[result] for result in self.results], name='Match_Outcome')

        # Prepara i dati per il training
        X_train, X_test, y_train, y_test = train_test_split(combined_data, y, test_size=test_size, random_state=42)
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        # Usa RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        self.features = list(X_train.columns)

        # Calcola le importanze delle feature
        feature_importance_df = pd.DataFrame({
            'Feature': self.features,
            'Importance': self.model.feature_importances_
        })
        logging.info(f"Feature importances for {self.name}: {feature_importance_df}")

        # Salva le importanze in un file CSV
        output_dir = 'feature_importances'
        os.makedirs(output_dir, exist_ok=True)
        feature_importance_file = os.path.join(output_dir, f'{self.name}_feature_importances.csv')
        feature_importance_df.sort_values(by='Importance', ascending=False).to_csv(feature_importance_file,
                                                                                   index=False)
        logging.info(f"Feature importances saved to {feature_importance_file}")


def optimize_model_parameters_parallel(team_1, team_2, threshold_range, test_size_range):
    def process_combination(threshold, test_size, team_1, team_2):
        # Extract numerical values from player totals
        combined_data_1 = np.array([list(player.totals.values()) for player in team_1.starters if
                                    isinstance(player.totals, dict) and len(player.totals) > 0])
        combined_data_2 = np.array([list(player.totals.values()) for player in team_2.starters if
                                    isinstance(player.totals, dict) and len(player.totals) > 0])

        y_1 = team_1.compute_points('legavolley_scraper/legavolley_scraper/spiders/results.json')
        y_2 = team_2.compute_points('legavolley_scraper/legavolley_scraper/spiders/results.json')

        print(y_1)
        print(y_2)

        # Preprocess data
        variances_1 = np.var(combined_data_1, axis=0)
        variances_2 = np.var(combined_data_2, axis=0)

        high_variance_features_1 = variances_1 > threshold
        high_variance_features_2 = variances_2 > threshold

        X_high_variance_1 = combined_data_1[:, high_variance_features_1]
        X_high_variance_2 = combined_data_2[:, high_variance_features_2]

        # Split data
        X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_high_variance_1, y_1, test_size=test_size,
                                                                    random_state=42)
        X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_high_variance_2, y_2, test_size=test_size,
                                                                    random_state=42)

        # Scale data
        scaler_1 = StandardScaler()
        X_train_1 = scaler_1.fit_transform(X_train_1)
        X_test_1 = scaler_1.transform(X_test_1)

        scaler_2 = StandardScaler()
        X_train_2 = scaler_2.fit_transform(X_train_2)
        X_test_2 = scaler_2.transform(X_test_2)

        # Train model
        model_1 = RandomForestClassifier(n_estimators=100, random_state=42)
        model_1.fit(X_train_1, y_train_1)

        model_2 = RandomForestClassifier(n_estimators=100, random_state=42)
        model_2.fit(X_train_2, y_train_2)

        # Evaluate model
        test_accuracy_1 = accuracy_score(y_test_1, model_1.predict(X_test_1))
        train_accuracy_1 = accuracy_score(y_train_1, model_1.predict(X_train_1))

        test_accuracy_2 = accuracy_score(y_test_2, model_2.predict(X_test_2))
        train_accuracy_2 = accuracy_score(y_train_2, model_2.predict(X_train_2))

        return {
            'Team 1': team_1.name,
            'Team 2': team_2.name,
            'Variance Threshold': threshold,
            'Test Size': test_size,
            'Train Accuracy Team 1': train_accuracy_1,
            'Test Accuracy Team 1': test_accuracy_1,
            'Train Accuracy Team 2': train_accuracy_2,
            'Test Accuracy Team 2': test_accuracy_2
        }

    results = []
    for threshold in threshold_range:
        for test_size in test_size_range:
            result = process_combination(threshold, test_size, team_1, team_2)
            results.append(result)

    return pd.DataFrame(results)


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

    team_1.train_model(threshold=1, test_size=0.2)
    team_2.train_model(threshold=1, test_size=0.2)

    # optimize_model_parameters_parallel(team_1, team_1, [1, 10], [0.1, 0.9])
