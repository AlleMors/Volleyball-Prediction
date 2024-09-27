import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Team:
    def __init__(self, name, excel_file):
        self.roster = excel_file.sheet_names.remove('Performance', 'Risultati')
        self.name = name
        self.excel_file = excel_file
        self.stats = None
        self.performance = excel_file.parse(sheet_name="Performance", na_values='').drop('Giornata', axis=1)
        self.results = excel_file.parse(sheet_name="Risultati")
        self.points = self.compute_points()
        self.dataframe = None

    def set_roster(self, players_list):
        self.stats = {
            player: self.excel_file.parse(sheet_name=player) for player in
            self.roster}
        print(self.roster)
        self.dataframe = pd.concat(self.roster.values(), axis=1)

    def compute_points(self):
        points = []
        for i in range(0, len(self.results)):
            res = self.results['SET'][i]
            if self.results['CASA'][i] == self.name:
                if res == '3-0' or res == '3-1':
                    points.append(3)
                elif res == '3-2':
                    points.append(2)
                elif res == '2-3':
                    points.append(1)
                else:
                    points.append(0)
            else:
                if res == '0-3' or res == '1-3':
                    points.append(3)
                elif res == '2-3':
                    points.append(2)
                elif res == '3-2':
                    points.append(1)
                else:
                    points.append(0)
        return points


if __name__ == "__main__":
    modena_xls = pd.ExcelFile("Modena_2023_2024.xlsx")
    padova_xls = pd.ExcelFile("Padova_2023_2024.xlsx")

    modena = Team("Valsa Group Modena", modena_xls)

    padova = Team("Pallavolo Padova", padova_xls)

    modena_players = {'Juantorena', 'Sapozkhov', 'Sanguinetti', 'Mossa de Rezende', 'Davyskiba', 'Rinaldi', 'Stankovic',
                      'Brehme', 'Federici', 'Gollini'}

    modena.set_roster(modena_players)

    padova_players = {'Falaschi', 'Crosato', 'Gardini', 'Porro', 'Plak', 'Garcia', 'Zenger'}

    padova.set_roster(padova_players)

    # Unione dei dati dei giocatori con le prestazioni delle squadre
    modena_performance_with_players = pd.concat([modena.performance, modena.dataframe],
                                                axis=1)
    padova_performance_with_players = pd.concat([padova.performance, padova.dataframe],
                                                axis=1)

    # Unione dei dati delle due squadre
    combined_data = pd.concat([modena.dataframe, padova.dataframe], axis=1)

    # Seleziona le caratteristiche (variabili indipendenti) e il target (variabile dipendente)
    X = combined_data
    y = modena.points

    # Suddivisione del dataset in set di addestramento e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

    # Standardizzazione dei dati
    scaler = StandardScaler()
    X_train.columns = X_train.columns.astype(str)
    X_train = scaler.fit_transform(X_train)
    X_test.columns = X_test.columns.astype(str)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Valutazione del modello
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    print(f'Accuracy sul set di addestramento: {train_accuracy}')
    print(f'Accuracy sul set di test: {test_accuracy}')
