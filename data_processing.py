import json
from collections import Counter

from sklearn.model_selection import StratifiedKFold, KFold

from models import Team


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
        for matchday in data['giornate']:
            convert_to_numeric(matchday)
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


def get_cross_validator(y, n_splits=5):
    class_counts = Counter(y)
    min_class_count = min(class_counts.values())

    if min_class_count >= 2:
        return StratifiedKFold(n_splits=min(n_splits, min_class_count), shuffle=True, random_state=42)
    else:
        return KFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=42)


def aggregate_past_data_symmetric(self, team_2):
    combined_data = []

    # Inizializzare le liste per le statistiche di ciascuna squadra
    rolling_data_self = {key: [] for key in ['punti_totali', 'punti_bp', 'battuta_totale',
                                             'ace_per_set', 'errori_battuta', 'ricezione_totale',
                                             'ricezione_efficienza', 'attacco_totale', 'attacco_efficienza',
                                             'muro_per_set']}
    rolling_data_team_2 = {key: [] for key in rolling_data_self.keys()}

    for matchday_index in range(len(self.starters[0].match_days)):
        # Calcolare le statistiche di media mobile per la giornata corrente per entrambe le squadre
        def calculate_rolling_avg(team, rolling_data):
            matchday_totals = {key: 0 for key in rolling_data.keys()}
            player_count = 0

            for player in team.starters:
                if matchday_index < len(player.match_days):
                    giornata = player.match_days[matchday_index]
                    for key in matchday_totals:
                        matchday_totals[key] += giornata[key]
                    player_count += 1

            if player_count > 0:
                for key in matchday_totals:
                    matchday_totals[key] /= player_count

            # Aggiungere le statistiche della giornata alla lista temporanea per il rolling
            for key in rolling_data:
                rolling_data[key].append(matchday_totals[key])

            # Calcolare la media mobile delle ultime 5 giornate
            return {f"{key}_{team.name}": sum(rolling_data[key]) / len(rolling_data[key])
                    for key in rolling_data if len(rolling_data[key]) > 0}

        # Statistiche della media mobile per la squadra self
        rolling_avg_self = calculate_rolling_avg(self, rolling_data_self)

        # Statistiche della media mobile per la squadra team_2
        rolling_avg_team_2 = calculate_rolling_avg(team_2, rolling_data_team_2)

        # Creare una singola riga con le statistiche di entrambe le squadre per la giornata
        combined_row = {**rolling_avg_self, **rolling_avg_team_2}
        combined_data.append(combined_row)

    return combined_data
