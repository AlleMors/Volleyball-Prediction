import json
from collections import Counter
from fpdf import FPDF
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
import logging

# Configuration for warnings and logging
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO)


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


def aggregate_past_data_symmetric(self, team_2):
    combined_data = []

    # Iterate twice: once with the normal order, once with the reverse order
    for team_a, team_b in [(self, team_2), (team_2, self)]:
        # Aggregate the past statistics of the starters of both teams into a single row for each match
        for matchday_index in range(len(team_a.starters[0].match_days)):
            matchday_totals = {'squadra': team_a.name, 'set_giocati': 0, 'punti_totali': 0, 'punti_bp': 0,
                               'battuta_totale': 0, 'ace': 0,
                               'errori_battuta': 0, 'ace_per_set': 0, 'battuta_efficienza': 0,
                               'ricezione_totale': 0, 'errori_ricezione': 0, 'ricezione_negativa': 0,
                               'ricezione_perfetta': 0, 'ricezione_perfetta_perc': 0,
                               'ricezione_efficienza': 0, 'attacco_totale': 0, 'errori_attacco': 0,
                               'attacco_murati': 0, 'attacco_perfetti': 0, 'attacco_perfetti_perc': 0,
                               'attacco_efficienza': 0, 'muro_perfetti': 0, 'muro_per_set': 0}

            # Sum the statistics for that matchday
            for player in team_a.starters:
                if matchday_index < len(player.match_days):
                    giornata = player.match_days[matchday_index]
                    for key in matchday_totals:
                        if key != 'squadra':
                            matchday_totals[key] += giornata[key]
            # Calculate the average statistics
            for key in matchday_totals:
                if key != 'squadra':
                    matchday_totals[key] /= len(team_a.starters)

            combined_data.append(matchday_totals)

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
        self.starters = [player for player in self.players if player.name in starter_names]

        if len(self.starters) != 7:
            raise ValueError("The number of starters must be 7")

    def load_results(self, results_data):
        for matchday in results_data:
            for result in matchday['results']:
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
            if player.name == name:
                return player
        return None

    def train_and_predict_match_winner_symmetric(self, team_2):
        combined_data = aggregate_past_data_symmetric(self, team_2)
        combined_data_df = pd.DataFrame(combined_data)
        combined_data_df['is_team_1'] = combined_data_df['squadra'].apply(lambda x: 1 if x == self.name else 0)
        combined_data_df = combined_data_df.drop(columns=['squadra'])

        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        combined_data_df_imputed = pd.DataFrame(imputer.fit_transform(combined_data_df),
                                                columns=combined_data_df.columns)

        scaler = StandardScaler()
        combined_data_df_scaled = pd.DataFrame(scaler.fit_transform(combined_data_df_imputed),
                                               columns=combined_data_df_imputed.columns)

        outcome_map = {'3-0': 1, '3-1': 1, '3-2': 1, '2-3': 0, '1-3': 0, '0-3': 0}
        y = pd.Series([outcome_map[result] for result in self.results], name='Match_Outcome')
        y_team_2 = pd.Series([outcome_map[result] for result in team_2.results], name='Match_Outcome')

        combined_y = pd.concat([y, y_team_2], ignore_index=True)
        combined_y = pd.concat([combined_y, combined_y], ignore_index=True)
        combined_data_df_scaled = pd.concat([combined_data_df_scaled, combined_data_df_scaled], ignore_index=True)

        n_splits = get_dynamic_n_splits(combined_y)
        rf = RandomForestClassifier(random_state=42)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(rf, combined_data_df_scaled, combined_y, cv=skf, scoring='accuracy')
        accuracy = np.mean(scores)
        print(f"Mean accuracy with {n_splits}-fold cross-validation: {accuracy:.2f}")

        rf.fit(combined_data_df_scaled, combined_y)
        prediction = rf.predict(combined_data_df_scaled)
        prediction_result = 'win' if prediction[0] == 1 else 'lose'

        # Chiamata della funzione per stampare l'importanza delle feature
        print_feature_importance(rf, combined_data_df_imputed.columns)

        return rf, prediction_result

def print_feature_importance(model, feature_names):
    # Estrazione e visualizzazione dell'importanza delle feature
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 8))
    plt.title("Feature Importance")
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()

    # Creazione del documento PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Feature Importance Report", ln=True, align="C")

    pdf.image("feature_importance.png", x=10, y=30, w=180)
    pdf.output("Feature_Importance_Report.pdf")
    print("Il documento PDF con l'importanza delle feature è stato creato con successo!")


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
                found_player.match_days = data['giornate']
                found_player.totals = data['totals']
                found_player.averages = data['averages']

    results_data = json.load(open(results_file_path))
    for team in teams:
        team.load_results(results_data)

    return teams


def get_dynamic_n_splits(y):
    class_counts = Counter(y)
    min_class_count = min(class_counts.values())
    return min_class_count


if __name__ == "__main__":
    team_objects = load_json('legavolley_scraper/legavolley_scraper/spiders/teams_stats.json',
                             'legavolley_scraper/legavolley_scraper/spiders/players_stats.json',
                             'legavolley_scraper/legavolley_scraper/spiders/results.json')

    team_names_map = {"trento": "Itas Trentino", "modena": "Valsa Group Modena", "perugia": "Sir Susa Vim Perugia",
                      "cisterna": "Cisterna Volley", "grottazzolina": "Yuasa Battery Grottazzolina",
                      "lube": "Cucine Lube Civitanova", "milano": "Allianz Milano", "monza": "Mint Vero Volley Monza",
                      "padova": "Sonepar Padova", "piacenza": "Gas Sales Bluenergy Piacenza",
                      "taranto": "Gioiella Prisma Taranto", "verona": "Rana Verona"}

    team_1 = next(team for team in team_objects if team.name == team_names_map["monza"])
    team_2 = next(team for team in team_objects if team.name == team_names_map["modena"])

    modena_starters = ["Sanguinetti Giovanni", "Anzani Simone", "Davyskiba Vlad", "De Cecco Luciano", "Buchegger Paul",
                       "Rinaldi Tommaso", "Federici Filippo"]
    trento_starters = ["Garcia Fernandez Gabi", "Kozamernik Jan", "Laurenzano Gabriele", "Lavia Daniele",
                       "Michieletto Alessandro",
                       "Resende Gualberto Flavio", "Sbertoli Riccardo"]
    perugia_starters = ["Giannelli Simone", "Loser Agustin", "Ben Tara Wassim", "Russo Roberto", "Colaci Massimo",
                        "Ishikawa Yuki", "Semeniuk Kamil"]
    cisterna_starters = ["Baranowicz Michele", "Bayram Efe", "Faure Theo", "Nedeljkovic Aleksandar", "Pace Domenico",
                         "Ramon Jordi", "Mazzone Daniele"]
    grottazzolina_starters = ["Antonov Oleg", "Demyanenko Danny", "Marchisio Andrea", "Mattei Andrea", "Tatarov Georgi",
                              "Zhukouski Tsimafei", "Marchiani Manuele"]
    lube_starters = ["Balaso Fabio", "Boninfante Mattia", "Bottolo Mattia", "Chinenyeze Barthelemy",
                     "Gargiulo Giovanni Maria", "Lagumdzija Adis", "Loeppky Eric"]
    milano_starters = ["Caneschi Edoardo", "Catania Damiano", "Kaziyski Matey", "Louati Yacine", "Porro Paolo",
                       "Reggers Ferre", "Schnitzer Jordan"]
    monza_starters = ["Beretta Thomas", "Di Martino Gabriele", "Gaggini Marco", "Kreling Fernando", "Marttila Luka",
                      "Rohrs Erik", "Szwarc Arthur"]
    padova_starters = ["Crosato Federico", "Diez Benjamin", "Falaschi Marco", "Masulovic Veljko", "Plak Fabian",
                       "Porro Luca", "Sedlacek Marko"]
    piacenza_starters = ["Scanferla Leonardo", "Brizard Antoine", "Galassi Gianluca", "Kovacevic Uros", "Maar Stephen",
                         "Simon Robertlandy", "Romanò Yuri"]
    taranto_starters = ["Alonso Roamy", "D'Heer Wout", "Lanza Filippo", "Hofer Brodie", "Rizzo Marco", "Zimmermann Jan",
                        "Gironi Fabrizio"]
    verona_starters = ["Dzavoronok Donovan", "Abaev Konstantin", "D'Amico Francesco", "Vitelli Marco", "Keita Noumory",
                       "Sani Francesco", "Cortesia Lorenzo"]

    team_starters = {"Itas Trentino": trento_starters, "Valsa Group Modena": modena_starters,
                     "Sir Susa Vim Perugia": perugia_starters, "Cisterna Volley": cisterna_starters,
                     "Yuasa Battery Grottazzolina": grottazzolina_starters, "Cucine Lube Civitanova": lube_starters,
                     "Allianz Milano": milano_starters, "Mint Vero Volley Monza": monza_starters,
                     "Sonepar Padova": padova_starters, "Gas Sales Bluenergy Piacenza": piacenza_starters,
                     "Gioiella Prisma Taranto": taranto_starters, "Rana Verona": verona_starters}

    team_1.select_starters(team_starters[team_1.name])
    team_2.select_starters(team_starters[team_2.name])

    # Trains the model and predicts the winner of the match between team_1 and team_2
    model, prediction_result = team_1.train_and_predict_match_winner_symmetric(team_2)

    # Print the prediction
    print(f"Prediction for the match between {team_1.name} and {team_2.name}: {team_1.name} will {prediction_result}")
