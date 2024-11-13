import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

from visualization import print_feature_importance


class PlayerMatchDay:
    def __init__(self, giornata_data):
        self.matchday = pd.to_numeric(giornata_data['giornata'].replace(" ", "0"), errors='coerce')
        self.played_sets = pd.to_numeric(giornata_data['set_giocati'].replace(" ", "0"), errors='coerce')
        self.total_points = pd.to_numeric(giornata_data['punti_totali'].replace(" ", "0"), errors='coerce')
        self.bp_points = pd.to_numeric(giornata_data['punti_bp'].replace(" ", "0"), errors='coerce')
        self.total_serves = pd.to_numeric(giornata_data['battuta_totale'].replace(" ", "0"), errors='coerce')
        self.ace = pd.to_numeric(giornata_data['ace'].replace(" ", "0"), errors='coerce')
        self.serve_errors = pd.to_numeric(giornata_data['errori_battuta'].replace(" ", "0"), errors='coerce')
        self.ace_per_set = pd.to_numeric(giornata_data['ace_per_set'].replace(" ", "0"), errors='coerce')
        self.serve_eff = pd.to_numeric(giornata_data['battuta_efficienza'].replace(" ", "0"), errors='coerce')
        self.total_reception = pd.to_numeric(giornata_data['ricezione_totale'].replace(" ", "0"), errors='coerce')
        self.reception_err = pd.to_numeric(giornata_data['errori_ricezione'].replace(" ", "0"), errors='coerce')
        self.reception_neg = pd.to_numeric(giornata_data['ricezione_negativa'].replace(" ", "0"), errors='coerce')
        self.reception_prf = pd.to_numeric(giornata_data['ricezione_perfetta'].replace(" ", "0"), errors='coerce')
        self.reception_prf_perc = pd.to_numeric(giornata_data['ricezione_perfetta_perc'].replace(" ", "0"),
                                                errors='coerce')
        self.reception_eff = pd.to_numeric(giornata_data['ricezione_efficienza'].replace(" ", "0"),
                                           errors='coerce')
        self.total_att = pd.to_numeric(giornata_data['attacco_totale'].replace(" ", "0"), errors='coerce')
        self.attack_errors = pd.to_numeric(giornata_data['errori_attacco'].replace(" ", "0"), errors='coerce')
        self.attack_blocked = pd.to_numeric(giornata_data['attacco_murati'].replace(" ", "0"), errors='coerce')
        self.attack_prf = pd.to_numeric(giornata_data['attacco_perfetti'].replace(" ", "0"), errors='coerce')
        self.attack_prf_perc = pd.to_numeric(giornata_data['attacco_perfetti_perc'].replace(" ", "0"),
                                             errors='coerce')
        self.attack_eff = pd.to_numeric(giornata_data['attacco_efficienza'].replace(" ", "0"), errors='coerce')
        self.block_prf = pd.to_numeric(giornata_data['muro_perfetti'].replace(" ", "0"), errors='coerce')
        self.block_per_set = pd.to_numeric(giornata_data['muro_per_set'].replace(" ", "0"), errors='coerce')


class Player:
    match_days = []
    totals = []
    averages = []

    def __init__(self, player_data):
        self.name = player_data['atleta']
        self.played_matches = pd.to_numeric(player_data['partite_giocate'].replace(" ", "0"), errors='coerce')
        self.played_sets = pd.to_numeric(player_data['set_giocati'].replace(" ", "0"), errors='coerce')
        self.total_points = pd.to_numeric(player_data['punti_totali'].replace(" ", "0"), errors='coerce')
        self.bp_points = pd.to_numeric(player_data['punti_bp'].replace(" ", "0"), errors='coerce')
        self.total_serves = pd.to_numeric(player_data['battuta_totale'].replace(" ", "0"), errors='coerce')
        self.ace = pd.to_numeric(player_data['ace'].replace(" ", "0"), errors='coerce')
        self.serve_errors = pd.to_numeric(player_data['errori_battuta'].replace(" ", "0"), errors='coerce')
        self.ace_per_set = pd.to_numeric(player_data['ace_per_set'].replace(" ", "0"), errors='coerce')
        self.serve_eff = pd.to_numeric(player_data['battuta_efficienza'].replace(" ", "0"), errors='coerce')
        self.total_reception = pd.to_numeric(player_data['ricezione_totale'].replace(" ", "0"), errors='coerce')
        self.reception_err = pd.to_numeric(player_data['errori_ricezione'].replace(" ", "0"), errors='coerce')
        self.reception_neg = pd.to_numeric(player_data['ricezione_negativa'].replace(" ", "0"), errors='coerce')
        self.reception_prf = pd.to_numeric(player_data['ricezione_perfetta'].replace(" ", "0"), errors='coerce')
        self.reception_prf_perc = pd.to_numeric(player_data['ricezione_perfetta_perc'].replace(" ", "0"),
                                                errors='coerce')
        self.reception_eff = pd.to_numeric(player_data['ricezione_efficienza'].replace(" ", "0"),
                                           errors='coerce')
        self.total_att = pd.to_numeric(player_data['attacco_totale'].replace(" ", "0"), errors='coerce')
        self.attack_err = pd.to_numeric(player_data['errori_attacco'].replace(" ", "0"), errors='coerce')
        self.attack_blocked = pd.to_numeric(player_data['attacco_murati'].replace(" ", "0"), errors='coerce')
        self.attack_prf = pd.to_numeric(player_data['attacco_perfetti'].replace(" ", "0"), errors='coerce')
        self.attack_prf_perc = pd.to_numeric(player_data['attacco_perfetti_perc'].replace(" ", "0"),
                                             errors='coerce')
        self.attack_eff = pd.to_numeric(player_data['attacco_efficienza'].replace(" ", "0"), errors='coerce')
        self.block_prf = pd.to_numeric(player_data['muro_perfetti'].replace(" ", "0"), errors='coerce')
        self.block_per_set = pd.to_numeric(player_data['muro_per_set'].replace(" ", "0"), errors='coerce')


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
import json


class Team:
    players = []
    totals = []

    def __init__(self, team_data):
        from data_processing import convert_to_numeric

        self.model = None
        self.results = []
        self.name = team_data['squadra']
        self.codename = team_data['codice']
        self.players = [Player(player) for player in team_data['players']]
        self.starters = []
        self.totals = convert_to_numeric(team_data['totali'])

    def select_starters(self, starter_names):
        self.starters = [player for player in self.players if player.name in starter_names]

        if len(self.starters) != 7:
            raise ValueError("The number of starters must be 7")

    def load_results(self, results_data):
        for matchday in results_data:
            for result in matchday['results']:
                if result['team'] == self.name:
                    self.results.append(result['result'])
                elif result['opponent'] == self.name:
                    self.results.append(result['result'][2] + '-' + result['result'][0])

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
            if player.name == name:
                return player
        return None

    def calculate_home_advantage(self, team_2):
        """
        Calcola il vantaggio casalingo come la differenza tra la percentuale di vittorie casalinghe
        della squadra di casa (self) e la percentuale di vittorie in trasferta dell'avversario (team_2).
        """
        home_performance = self.results.count('win') / len(self.results)
        away_performance = team_2.results.count('win') / len(team_2.results)

        return home_performance - away_performance

    def train_and_predict_match_winner_symmetric(self, team_2):
        from data_processing import aggregate_past_data_symmetric, get_dynamic_n_splits, get_cross_validator

        best_accuracy = 0
        best_advantage = 0

        # Calcola il vantaggio casalingo dinamico
        home_advantage = self.calculate_home_advantage(team_2)

        # Ottieni i dati aggregati come differenziali
        combined_data = aggregate_past_data_symmetric(self, team_2)

        combined_data_df = pd.DataFrame(combined_data)

        with open('combined_data_df.json', 'w') as f:
            combined_data_df.to_json(f, orient='records', indent=4)

        # Rimuove 'is_team_1' e altre colonne non necessarie
        combined_data_df = combined_data_df.drop(columns=['squadra'], errors='ignore')

        # Gestione dei valori mancanti
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        combined_data_df_imputed = pd.DataFrame(imputer.fit_transform(combined_data_df),
                                                columns=combined_data_df.columns)

        # Mappatura dei risultati delle partite in valori binari (vittoria o sconfitta)
        outcome_map = {'3-0': 1, '3-1': 1, '3-2': 1, '2-3': 0, '1-3': 0, '0-3': 0}
        filtered_results = [result for result in self.results if result != '0-0']
        filtered_results_team_2 = [result for result in team_2.results if result != '0-0']

        # Mappatura dei risultati per entrambe le squadre, con il nome della squadra nella colonna
        y_team_1 = pd.Series([outcome_map[result] for result in filtered_results], name=f'Match_Outcome_{self.name}')
        y_team_2 = pd.Series([outcome_map[result] for result in filtered_results_team_2],
                             name=f'Match_Outcome_{team_2.name}')

        # Combinazione delle serie in un DataFrame, con una colonna per ciascuna squadra
        combined_y = pd.concat([y_team_1, y_team_2], axis=1)

        with open('combined_y.json', 'w') as f:
            combined_y.to_json(f, orient='records', indent=4)

        # Cross-validation dinamica
        # n_splits = get_dynamic_n_splits(combined_y)
        skf = get_cross_validator(combined_y)

        # Parametri per l'ottimizzazione
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Modello Random Forest
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
        grid_search.fit(combined_data_df_imputed, combined_y)

        # Modello migliore da GridSearch
        best_rf = grid_search.best_estimator_

        # Calcola l'accuratezza media del modello migliore
        accuracy = np.mean(
            cross_val_score(best_rf, combined_data_df_imputed, combined_y, cv=skf, scoring='accuracy'))

        # Salva il miglior vantaggio se l'accuratezza migliora
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_advantage = home_advantage

        print(f"Best home team advantage: {best_advantage} with accuracy: {best_accuracy:.2f}")

        # Addestra il modello finale con il miglior vantaggio casalingo trovato
        combined_data_df_imputed = pd.DataFrame(imputer.transform(combined_data_df),
                                                columns=combined_data_df.columns)

        with open('combined_data_df_imputed.json', 'w') as f:
            combined_data_df_imputed.to_json(f, orient='records', indent=4)

        best_rf.fit(combined_data_df_imputed, combined_y)

        prediction = best_rf.predict(combined_data_df_imputed)

        # Supponiamo che prediction contenga le previsioni per entrambe le squadre
        # Ad esempio, prediction = [1, 0] se la squadra 1 vince e la squadra 2 perde
        # Puoi fare un conteggio dei valori 1 per ciascuna squadra

        # Conta gli 1 nella prima colonna (squadra 1)
        team_1_wins = np.sum(prediction[:, 0] == 1)

        # Conta gli 1 nella seconda colonna (squadra 2)
        team_2_wins = np.sum(prediction[:, 1] == 1)

        # Decidi il risultato in base a quale squadra ha più vittorie
        if team_1_wins > team_2_wins:
            prediction_result = f'{self.name} wins'  # Squadra 1 ha più vittorie
        else:
            prediction_result = f'{team_2.name} wins'  # Squadra 2 ha più vittorie

        print(f"Prediction result: {prediction_result}")

        # Stampa l'importanza delle feature
        print_feature_importance(best_rf, combined_data_df_imputed.columns)

        return best_rf, prediction_result
