import tkinter as tk
from tkinter import ttk, messagebox

import main


def run_prediction(team_1_name, team_2_name):
    # Trova gli oggetti team corrispondenti ai nomi selezionati
    team_1 = next(team for team in main.team_objects if team.name == team_1_name)
    team_2 = next(team for team in main.team_objects if team.name == team_2_name)

    # Seleziona i titolari delle squadre
    team_1.select_starters(main.team_starters[team_1.name])
    team_2.select_starters(main.team_starters[team_2.name])

    # Allena il modello e predice il vincitore
    model, prediction_result = team_1.train_and_predict_match_winner_symmetric(team_2)

    # Mostra il risultato della previsione
    messagebox.showinfo("Risultato della previsione",
                        f"Previsione per la partita tra {team_1.name} e {team_2.name}: {team_1.name} vincerà" if prediction_result == "win" else f"Previsione per la partita tra {team_1.name} e {team_2.name}: {team_2.name} vincerà")


def create_gui():
    # Finestra principale
    root = tk.Tk()
    root.title("Seleziona le partite")

    # Label per la selezione delle squadre
    label = tk.Label(root, text="Seleziona le due squadre per la previsione:")
    label.pack(pady=10)

    # Crea una combobox per selezionare la prima squadra
    team_names = list(main.team_names)
    combobox_team_1 = ttk.Combobox(root, values=team_names, state="readonly")
    combobox_team_1.set("Seleziona la prima squadra")
    combobox_team_1.pack(pady=5)

    # Crea una combobox per selezionare la seconda squadra
    combobox_team_2 = ttk.Combobox(root, values=team_names, state="readonly")
    combobox_team_2.set("Seleziona la seconda squadra")
    combobox_team_2.pack(pady=5)

    # Funzione per il bottone predizione
    def on_predict_button_click():
        team_1_name = combobox_team_1.get()
        team_2_name = combobox_team_2.get()

        # Verifica se sono state selezionate entrambe le squadre
        if team_1_name == "Seleziona la prima squadra" or team_2_name == "Seleziona la seconda squadra":
            messagebox.showwarning("Selezione mancante", "Seleziona entrambe le squadre!")
            return

        # Previsione della partita
        run_prediction(team_1_name, team_2_name)

    # Bottone per eseguire la previsione
    predict_button = tk.Button(root, text="Predici", command=on_predict_button_click)
    predict_button.pack(pady=10)

    # Avvio dell'interfaccia grafica
    root.mainloop()
