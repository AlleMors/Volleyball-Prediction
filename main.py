from data_processing import load_json

team_objects = load_json('legavolley_scraper/legavolley_scraper/spiders/teams_stats.json',
                         'legavolley_scraper/legavolley_scraper/spiders/players_stats.json',
                         'legavolley_scraper/legavolley_scraper/spiders/results.json')

# Definizione dei giocatori titolari per ogni squadra (già presente nel tuo codice)
modena_starters = ["Sanguinetti Giovanni", "Anzani Simone", "Davyskiba Vlad", "De Cecco Luciano", "Buchegger Paul",
                   "Rinaldi Tommaso", "Federici Filippo"]
trento_starters = ["Garcia Fernandez Gabi", "Kozamernik Jan", "Laurenzano Gabriele", "Lavia Daniele",
                   "Michieletto Alessandro", "Resende Gualberto Flavio", "Sbertoli Riccardo"]
perugia_starters = ["Giannelli Simone", "Loser Agustin", "Ben Tara Wassim", "Russo Roberto", "Colaci Massimo",
                    "Plotnytskyi Oleh", "Semeniuk Kamil"]
cisterna_starters = ["Baranowicz Michele", "Bayram Efe", "Faure Theo", "Nedeljkovic Aleksandar", "Pace Domenico",
                     "Ramon Jordi", "Diamantini Enrico"]
grottazzolina_starters = ["Antonov Oleg", "Demyanenko Danny", "Marchisio Andrea", "Comparoni Francesco",
                          "Tatarov Georgi", "Zhukouski Tsimafei", "Cvanciger Gabrijel"]
lube_starters = ["Balaso Fabio", "Boninfante Mattia", "Bottolo Mattia", "Chinenyeze Barthelemy",
                 "Nikolov Aleksandar", "Lagumdzija Adis", "Podrascanin Marko"]
milano_starters = ["Caneschi Edoardo", "Staforini Matteo", "Gardini Davide", "Louati Yacine", "Porro Paolo",
                   "Reggers Ferre", "Piano Matteo"]
monza_starters = ["Beretta Thomas", "Di Martino Gabriele", "Gaggini Marco", "Kreling Fernando", "Marttila Luka",
                  "Zaytsev Ivan", "Szwarc Arthur"]
padova_starters = ["Crosato Federico", "Diez Benjamin", "Falaschi Marco", "Masulovic Veljko", "Plak Fabian",
                   "Porro Luca", "Sedlacek Marko"]
piacenza_starters = ["Scanferla Leonardo", "Brizard Antoine", "Ricci Fabio", "Mandiraci Ramazan Efe",
                     "Maar Stephen", "Simon Robertlandy", "Romanò Yuri"]
taranto_starters = ["Alonso Roamy", "D'Heer Wout", "Lanza Filippo", "Hofer Brodie", "Rizzo Marco", "Zimmermann Jan",
                    "Gironi Fabrizio"]
verona_starters = ["Dzavoronok Donovan", "Abaev Konstantin", "D'Amico Francesco", "Vitelli Marco", "Keita Noumory",
                   "Mozic Rok", "Cortesia Lorenzo"]

team_starters = {"Itas Trentino": trento_starters, "Valsa Group Modena": modena_starters,
                 "Sir Susa Vim Perugia": perugia_starters, "Cisterna Volley": cisterna_starters,
                 "Yuasa Battery Grottazzolina": grottazzolina_starters, "Cucine Lube Civitanova": lube_starters,
                 "Allianz Milano": milano_starters, "Mint Vero Volley Monza": monza_starters,
                 "Sonepar Padova": padova_starters, "Gas Sales Bluenergy Piacenza": piacenza_starters,
                 "Gioiella Prisma Taranto": taranto_starters, "Rana Verona": verona_starters}

team_names = [team.name for team in team_objects]

if __name__ == "__main__":
    from gui import create_gui

    # Avvia la GUI per la selezione delle partite
    create_gui()
