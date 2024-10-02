import json

import numpy as np
import pandas as pd
from IPython.core.macro import coding_declaration
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import warnings
import logging
import os

# Configuration for warnings and logging
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO)


class Giornata:

    def __init__(self, giornata_data):
        self.giornata = giornata_data['giornata']
        self.set_giocati = giornata_data['set_giocati']
        self.punti_totali = giornata_data['punti_totali']
        self.punti_bp = giornata_data['punti_bp']
        self.battuta_totale = giornata_data['battuta_totale']
        self.ace = giornata_data['ace']
        self.errori_battuta = giornata_data['errori_battuta']
        self.ace_per_set = giornata_data['ace_per_set']
        self.battuta_efficienza = giornata_data['battuta_efficienza']
        self.ricezione_totale = giornata_data['ricezione_totale']
        self.errori_ricezione = giornata_data['errori_ricezione']
        self.ricezione_negativa = giornata_data['ricezione_negativa']
        self.ricezione_perfetta = giornata_data['ricezione_perfetta']
        self.ricezione_perfetta_perc = giornata_data['ricezione_perfetta_perc']
        self.ricezione_efficienza = giornata_data['ricezione_efficienza']
        self.attacco_totale = giornata_data['attacco_totale']
        self.errori_attacco = giornata_data['errori_attacco']
        self.attacco_murati = giornata_data['attacco_murati']
        self.attacco_perfetti = giornata_data['attacco_perfetti']
        self.attacco_perfetti_perc = giornata_data['attacco_perfetti_perc']
        self.attacco_efficienza = giornata_data['attacco_efficienza']
        self.muro_perfetti = giornata_data['muro_perfetti']
        self.muro_per_set = giornata_data['muro_per_set']


class Player:
    giornate=[Giornata]

    def __init__(self, player_data):
        self.nome = player_data['atleta']
        self.partite_giocate = player_data['partite_giocate']
        self.set_giocati = player_data['set_giocati']
        self.punti_totali = player_data['punti_totali']
        self.punti_bp = player_data['punti_bp']
        self.battuta_totale = player_data['battuta_totale']
        self.ace = player_data['ace']
        self.errori_battuta = player_data['errori_battuta']
        self.ace_per_set = player_data['ace_per_set']
        self.battuta_efficienza = player_data['battuta_efficienza']
        self.ricezione_totale = player_data['ricezione_totale']
        self.errori_ricezione = player_data['errori_ricezione']
        self.ricezione_negativa = player_data['ricezione_negativa']
        self.ricezione_perfetta = player_data['ricezione_perfetta']
        self.ricezione_perfetta_perc = player_data['ricezione_perfetta_perc']
        self.ricezione_efficienza = player_data['ricezione_efficienza']
        self.attacco_totale = player_data['attacco_totale']
        self.errori_attacco = player_data['errori_attacco']
        self.attacco_murati = player_data['attacco_murati']
        self.attacco_perfetti = player_data['attacco_perfetti']
        self.attacco_perfetti_perc = player_data['attacco_perfetti_perc']
        self.attacco_efficienza = player_data['attacco_efficienza']
        self.muro_perfetti = player_data['muro_perfetti']
        self.muro_per_set = player_data['muro_per_set']
        self.giornate = [Giornata]


class Team:
    players = []

    def __init__(self, team_data):
        self.name = team_data['squadra']
        self.codename = team_data['codice']
        self.players = team_data['players']

        for player in self.players:
            player = Player(player, giorate_file_path)

    def compute_points(self):
        points_map = {'3-0': 3, '3-1': 3, '3-2': 2, '2-3': 1, '1-3': 0, '0-3': 0}
        reverse_points_map = {'3-0': 0, '3-1': 0, '3-2': 1, '2-3': 2, '1-3': 3, '0-3': 3}
        is_home_team = self.results['CASA'] == self.name
        points = np.where(is_home_team, self.results['SET'].map(points_map),
                          self.results['SET'].map(reverse_points_map))
        return points

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
        combined_data = pd.concat([self.stats[player] for player in self.starters], axis=1)
        combined_data['Match_Outcome'] = self.compute_match_outcome(self.results, self.name)

        variances = combined_data.var()
        high_variance_features = variances[variances > threshold].index
        X_high_variance = combined_data[high_variance_features]
        y = combined_data['Match_Outcome']

        X_train, X_test, y_train, y_test = train_test_split(X_high_variance, y, test_size=test_size, random_state=42)

        # Assicura che la directory 'data' esista
        os.makedirs('data', exist_ok=True)

        # Salva i dati su file
        X_train.to_csv(f'data/{self.name}_X_train.csv', index=False)
        X_test.to_csv(f'data/{self.name}_X_test.csv', index=False)
        y_train.to_csv(f'data/{self.name}_y_train.csv', index=False)
        y_test.to_csv(f'data/{self.name}_y_test.csv', index=False)

        self.scaler = StandardScaler()
        X_train = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        self.features = list(X_train.columns)
        logging.info(f"Features during model training for {self.name}: {self.features}")

        # Salva le importanze in un file CSV
        output_dir = 'feature_importances'
        os.makedirs(output_dir, exist_ok=True)  # Crea la cartella se non esiste
        feature_importance_file = os.path.join(output_dir, f'{self.name}_feature_importances.csv')
        feature_importance_df = pd.DataFrame({
            'Feature': self.features,
            'Importance': self.model.feature_importances_
        })
        feature_importance_df.sort_values(by='Importance', ascending=False).to_csv(feature_importance_file, index=False)

        logging.info(f"Feature importances saved to {feature_importance_file}")

    def predict_match_result(self, opponent_team):
        # Dati della propria squadra
        team1_data = pd.concat([self.stats[player] for player in self.starters], axis=1)

        # Dati della squadra avversaria
        opponent_data = pd.concat([opponent_team.stats[player] for player in opponent_team.starters], axis=1)

        # Aggiungi i risultati precedenti della propria squadra con un suffisso
        team_results = self.results[['SET', 'CASA', 'OSPITE']].copy()
        team_results.columns = [f"{col}_team" for col in team_results.columns]  # Aggiungi suffisso

        # Aggiungi i risultati precedenti della squadra avversaria con un suffisso
        opponent_results = opponent_team.results[['SET', 'CASA', 'OSPITE']].copy()
        opponent_results.columns = [f"{col}_opponent" for col in opponent_results.columns]  # Aggiungi suffisso

        # Combina tutti i dati
        combined_data = pd.concat([team1_data, opponent_data, team_results, opponent_results], axis=1)

        # Riorganizza le colonne in base alle feature del modello
        combined_data = combined_data.reindex(columns=self.features).fillna(0)

        # Normalizza i dati
        combined_data_scaled = pd.DataFrame(self.scaler.transform(combined_data), columns=self.features)

        # Effettua la previsione
        prediction = self.model.predict(combined_data_scaled)
        result = "Win" if prediction[0] == 1 else "Loss"
        return result

    def select_starters(self):
        print(f"Select the starters for {self.name} by typing the corresponding number:")
        for i, player in enumerate(self.roster):
            print(f"{i + 1}. {player}")

        selected_indices = input("Enter the numbers of the starters separated by a comma: ")
        selected_indices = [int(x.strip()) - 1 for x in selected_indices.split(',')]
        starters = [self.roster[i] for i in selected_indices]
        return starters


def optimize_model_parameters_parallel(team_objects, threshold_range, test_size_range):
    def process_combination(threshold, test_size, team):
        combined_data = pd.concat([team.dataframe for team in team_objects], axis=1)
        x_high_variance = team.preprocess_data(combined_data, threshold)
        y = team.points

        x_train, x_test, y_train, y_test = train_test_split(x_high_variance, y, test_size=test_size, random_state=42)

        scaler = StandardScaler()
        x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
        x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)

        test_accuracy = accuracy_score(y_test, model.predict(x_test))
        train_accuracy = accuracy_score(y_train, model.predict(x_train))

        return {
            'Team': team.name,
            'Variance Threshold': threshold,
            'Test Size': test_size,
            'Train Accuracy': train_accuracy,
            'Test Accuracy': test_accuracy
        }

    results = Parallel(n_jobs=-1)(delayed(process_combination)(threshold, test_size, team)
                                  for threshold in threshold_range
                                  for test_size in test_size_range
                                  for team in team_objects)
    return pd.DataFrame(results)


'''
def load_team_data(excel_file):
    roster = [sheet for sheet in excel_file.sheet_names if sheet not in ['Performance', 'Risultati']]
    stats = {player: excel_file.parse(sheet_name=player).drop('Giornata', axis=1).rename(
        columns=lambda col: f"{player}_{col}") for player in roster}
    performance = excel_file.parse(sheet_name="Performance").drop('Giornata', axis=1)
    results = excel_file.parse(sheet_name="Risultati")
    return roster, stats, performance, results
'''


def load_json(teams_file_path, giornate_file_path):
    teams = [Team]
    giornate=[Giornata]
    file_teams = json.load(open(teams_file_path))
    for data in file_teams:
        teams.append(Team(data))
    file_giornate=json.load(open(giornate_file_path))
    for data in file_giornate:
        giornate.append(Giornata(data))

    for giornata in giornate:



if __name__ == "__main__":
    '''
    teams = ["Valsa Group Modena", "Pallavolo Padova"]
    file_paths = ["Teams/Old/Modena_2023_2024.xlsx", "Teams/Old/Padova_2023_2024.xlsx"]
    
    
    excel_files = [pd.ExcelFile(file) for file in file_paths]
    team_data = [load_team_data(excel_file) for excel_file in excel_files]
    team_objects = [Team(team, *data) for team, data in zip(teams, team_data)]
    '''
    load_json('legavolley_scraper/legavolley_scraper/spiders/teams_stats.json')

    threshold_range = np.arange(0, 5, 0.05)
    test_size_range = np.arange(0.5, 0.9, 0.05)

    logging.info("Starting model parameter optimization...")
    results_df = optimize_model_parameters_parallel(team_objects, threshold_range, test_size_range)
    logging.info("Optimization complete!")

    best_result = results_df.sort_values(by='Test Accuracy', ascending=False).iloc[0]
    logging.info("Best parameter combination:")
    logging.info(best_result)

    for team in team_objects:
        logging.info(f"Training model for team: {team.name}")
        team.train_model(best_result['Variance Threshold'], 0.5)

    team1, team2 = team_objects
    prediction = team1.predict_match_result(team2)
    logging.info(f"Predicted result between {team1.name} and {team2.name}: {prediction}")
