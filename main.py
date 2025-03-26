# %%
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
###########                                                         ############
########### #     # ####### #####   #     # #       #######  ###### ############
########### ##   ## #     # #    #  #     # #       #       #       ############
########### #  #  # #     # #     # #     # #       #######  ####   ############
########### #     # #     # #    #  #     # #       #             # ############
########### #     # ####### #####   #######  ###### ####### ######  ############
###########                                                         ############
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
# %%
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import re
from itertools import combinations
import datetime
from math import comb
from str_to_ascii import *

np.random.seed(0)

# %%
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
######                                                                    ######
###### #     # ####### ##    #         ####### #     # #     # #######    ######
###### ##   ##    #    # #   #         #        #   #  ##   ## #     #    ######
###### #  #  #    #    #  #  #         #######    #    #  #  # #######    ######
###### #     #    #    #   # #         #        #   #  #     # #          ######
###### #     # ####### #    ## ##      ####### #     # #     # #       ## ######
######                                                                    ######
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
# %%
df_minimal_example = pd.DataFrame(
    {
        "Name": [
            "Alice",
            "Bob",
            "Charlie",
            "David",
            "Eve",
            "Frank",
            "Grace",
            "Hugo",
            "Ivy",
            "Jack",
            "Katherine",
            "Liam",
            "Mia",
            "Noah",
            "Olivia",
            "Penelope",
            "Quinn",
            "Riley",
            "Sophia",
            "Theo",
            "Uma",
            "Violet",
            "Wyatt",
            "Xavier",
            "Yara",
            "Zane",
        ],
        "Surname": [
            "Smith",
            "Johnson",
            "Williams",
            "Jones",
            "Brown",
            "Davis",
            "Miller",
            "Wilson",
            "Moore",
            "Taylor",
            "Anderson",
            "Thomas",
            "Jackson",
            "White",
            "Harris",
            "Martin",
            "Thompson",
            "Garcia",
            "Martinez",
            "Robinson",
            "Clark",
            "Rodriguez",
            "Lewis",
            "Lee",
            "Walker",
            "Hall",
        ],
        "Level": [
            0,
            1,
            2,
            3,
            0,
            1,
            2,
            3,
            0,
            1,
            1,
            2,
            3,
            0,
            1,
            2,
            3,
            0,
            1,
            2,
            3,
            0,
            1,
            2,
            3,
            2,
        ],
        "Gender": ["Female", "Male"] * 13,
        "Games played": [0] * 26,
        "Happiness": [0] * 26,
    }
).astype(
    {
        "Name": str,
        "Surname": str,
        "Level": int,
        "Gender": str,
    }
)

df_minimal_example.set_index("Name", inplace=True)
df_minimal_example["Level"].unique()


# %%
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
###############                                                 ################
############### ####### #     # #######         #####   ####### ################
###############    #    ##   ## #     #         #    #  #       ################
###############    #    #  #  # #######         #     # #####   ################
###############    #    #     # #               #    #  #       ################
############### ####### #     # #       ##      #####   #       ################
###############                                                 ################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
# %%


main_df = pd.read_excel("Adhésion au GRNA 2024-2025 (Responses).xlsx")
level_df = pd.read_excel("Niveau.xlsx")

for data_frame in [main_df, level_df]:
    data_frame["PrénomNom"] = data_frame["Prénom"].apply(
        lambda x: re.sub(" ", "", x).capitalize()
    )
    for idx_1 in data_frame.index:
        for idx_2 in data_frame.drop(idx_1, axis=0).index:
            temp_surname_1 = re.sub(" ", "", data_frame.loc[idx_1, "Nom"]).capitalize()
            temp_surname_2 = re.sub(" ", "", data_frame.loc[idx_2, "Nom"]).capitalize()
            while (
                data_frame.loc[idx_1, "PrénomNom"] == data_frame.loc[idx_2, "PrénomNom"]
            ):
                data_frame.loc[idx_1, "PrénomNom"] += temp_surname_1[0]
                data_frame.loc[idx_2, "PrénomNom"] += temp_surname_2[0]
                temp_surname_1 = temp_surname_1[1:]
                temp_surname_2 = temp_surname_2[1:]

    data_frame.set_index("PrénomNom", inplace=True)

main_fd = main_df.loc[level_df.index]

main_df["Niveau"] = level_df["Niveau moyenné"]
main_df.dropna(axis=0, subset =["Niveau"], inplace=True)
main_df.drop(
    "Le montant de la cotisation semestrielle pour la saison 2024-2025 étant en discussion, il faudra la payer ultérieurement. Le GRNA se réserve le droit d'empêcher l'accès aux entraînement à celles et ceux qui ne règleront pas la cotisation quand ce sera demandé.",
    axis=1,
    inplace=True,
)

# changing dtypes of each column
main_df["Genre"] = main_df["Genre"].apply(
    lambda x: "Male" if x == "Masculin" else "Female"
)
main_df["Êtes-vous étudiant ?"] = main_df["Êtes-vous étudiant ?"].apply(
    lambda x: True if x == "Oui" else False
)
main_df["Avez-vous l'ambition de participer à des tournois cette saison ?"] = main_df[
    "Avez-vous l'ambition de participer à des tournois cette saison ?"
].apply(lambda x: True if x == "Oui" else False)
main_df[
    "Acceptez-vous d'apparaître en photo ou en vidéo sur les réseaux de diffusion du club ?"
] = main_df[
    "Acceptez-vous d'apparaître en photo ou en vidéo sur les réseaux de diffusion du club ?"
].apply(
    lambda x: True if x == "Oui" else False
)
translated_shortened_columns = {
    "Timestamp": "Timestamp",
    "Nom": "Surname",
    "Prénom": "Name",
    "Date de naissance": "Date of birth",
    "Genre": "Gender",
    "Adresse e-mail": "Email address",
    "Code Postal": "Postal code",
    "Numéro de téléphone": "Phone number",
    "Êtes-vous étudiant ?": "student",
    "Avez-vous l'ambition de participer à des tournois cette saison ?": "tournaments participation",
    "Acceptez-vous d'apparaître en photo ou en vidéo sur les réseaux de diffusion du club ?": "video agreement",
    "Donnez votre RG-ID (identifiant roundnet playerzone, image ci-dessous) si vous en avez un!\nSi vous n'en avez pas, créez un compte (ce sera nécessaire pour participer à des tournois)": "RG-ID",
    "Niveau": "Level",
}




main_df.rename(columns=translated_shortened_columns, inplace=True)
main_df.index.name = "NameSurname"

true_dtypes = {
    "Timestamp": datetime.datetime,
    "Surname": str,
    "Name": str,
    "Date of birth": datetime.datetime,
    "Gender": str,
    "Email address": str,
    "Postal code": int,
    "Phone number": str,
    "student": bool,
    "tournaments participation": bool,
    "video agreement": bool,
    "RG-ID": str,
    "Level": int,
}


for col in true_dtypes.keys():
    if true_dtypes[col] == datetime.datetime:
        main_df[col] = pd.to_datetime(main_df[col])
    else:
        main_df[col] = main_df[col].astype(true_dtypes[col])

# setting default category as level
main_df["Category"] = main_df["Level"]
# setting default Happiness to 0
main_df["Happiness"] = 0
# setting default games played to 0
main_df["Games played"] = 0

# %%
example_df = main_df.iloc[3:18]
short_example_df = main_df.loc[["VictorDa", "David", "Enzo", "Anyel", "Nolan"]]


# %%
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
###########                                                         ############
###########  ###### #        #####   ######  ###### #######  ###### ############
########### #       #       #     # #       #       #       #       ############
########### #       #       #######  ####    ####   #######  ####   ############
########### #       #       #     #       #       # #             # ############
###########  ######  ###### #     # ######  ######  ####### ######  ############
###########                                                         ############
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
# %%
################################################################################
###############                                                 ################
############### ####### #        #####  #     # ####### ######  ################
############### #     # #       #     # #     # #       #     # ################
############### ####### #       #######   ###   ####### ######  ################
############### #       #       #     #    #    #       #   ##  ################
############### #        ###### #     #    #    ####### #    ## ################
###############                                                 ################
################################################################################
class Player:
    def __init__(self, series):
        # Initialize the Player object with a pandas Series
        self.series = series
        # Set attributes for each key in the series
        for key in series.keys():
            setattr(self, re.sub(" ", "_", key.lower()), series[key])
        # Set the name attribute
        self.name = series.name


# %% Example usage of the Player class
player = Player(df_minimal_example.iloc[0])
# Print the attributes of the player
for attr in ["Name"] + list(player.series.keys()):
    print(attr + " : ", getattr(player, re.sub(" ", "_", attr.lower())))


# %%
################################################################################
###                                                                         ####
### ####### #######  #####  #     # ####### ####### ####### #     # ####### ####
###    #    #       #     # ##   ## #     # #          #    #     # #     # ####
###    #    ####### ####### #  #  # #     # #####      #    #  #  # #     # ####
###    #    #       #     # #     # #     # #          #    ##   ## #     # ####
###    #    ####### #     # #     # ####### #          #    #     # ####### ####
###                                                                         ####
################################################################################
class TeamOfTwo:
    def __init__(self, player_A, player_B, preference=None):
        # Initialize the TeamOfTwo object with two players and a preference
        self.preference = preference
        # Create a DataFrame with the series of both players
        self.df = pd.concat([player_A.series, player_B.series], axis=1).T
        self.player_A = player_A
        self.player_B = player_B
        # Store the names of the players in a set
        self.players = {player_A, player_B}
        self.players_name = {player_A.name, player_B.name}

        # Calculate the level difference between the two players
        self.level_difference = abs(self.player_A.level - self.player_B.level)
        # calculate the mean level of the team
        self.mean_level = (self.player_A.level + self.player_B.level) / 2
        # Determine if the team is mixed gender
        self.mixed = True if self.player_A.gender != self.player_B.gender else False
        # Determine if both players are male
        self.male = (
            True
            if self.player_A.gender == "Male" and self.player_B.gender == "Male"
            else False
        )
        # Determine if both players are female
        self.female = (
            True
            if self.player_A.gender == "Female" and self.player_B.gender == "Female"
            else False
        )
        # Determine if both players are non-binary
        self.non_binary = (
            True
            if self.player_A.gender == "Non-binary"
            and self.player_B.gender == "Non-binary"
            else False
        )


# %%
team_of_two = TeamOfTwo(
    Player(df_minimal_example.iloc[0]),
    Player(df_minimal_example.iloc[1]),
    preference="Level",
)
for attr in [
    "preference",
    "players",
    "level_difference",
    "mixed",
    "male",
    "female",
    "non_binary",
]:
    print(attr + " : ", getattr(team_of_two, attr))


# %%
###################################################################################
#                                                                                 #
# #######  #####  #     # ####### ####### ####### ####### ####### #     # ######  #
# #       #     # ##   ## #       #     # #       #       #     # #     # #     # #
# #  #### ####### #  #  # ####### #     # #####   #####   #     # #     # ######  #
# #     # #     # #     # #       #     # #       #       #     # #     # #   ##  #
# ####### #     # #     # ####### ####### #       #       ####### ####### #    ## #
#                                                                                 #
###################################################################################
# %%
class GameOfFour:
    def __init__(self, team_A, team_B, preference=None):
        # Initialize the GameOfFour object with two teams and a preference
        self.preference = preference

        # Create a DataFrame with the data of both teams
        self.df = pd.concat([team_A.df, team_B.df], axis=0)

        self.team_A = team_A
        self.team_B = team_B

        # Store the teams in a set of frozensets
        self.teams = set([team_A, team_B])
        # Store the participants of the game
        self.participants = frozenset(self.team_A.players.union(self.team_B.players))
        self.level_difference = abs(self.team_A.mean_level - self.team_B.mean_level)


# %% Example usage of the GameOfFour class
example_one_game_df = df_minimal_example.iloc[:4]
game_of_four = GameOfFour(
    TeamOfTwo(Player(example_one_game_df.iloc[0]), Player(example_one_game_df.iloc[1])),
    TeamOfTwo(Player(example_one_game_df.iloc[2]), Player(example_one_game_df.iloc[3])),
    preference="Level",
)
# Print the attributes of the game
for attr in ["preference", "teams", "participants", "level_difference"]:
    print(attr + " : ", getattr(game_of_four, attr))


# %%
###################################################################################
#                                                                                 #
# #######  #####  #     # #######  ###### ######  ####### #     # ##    # #####   #
# #       #     # ##   ## #       #       #     # #     # #     # # #   # #    #  #
# #  #### ####### #  #  # #######  ####   ######  #     # #     # #  #  # #     # #
# #     # #     # #     # #             # #   ##  #     # #     # #   # # #    #  #
# ####### #     # #     # ####### ######  #    ## ####### ####### #    ## #####   #
#                                                                                 #
###################################################################################
# %%
class GamesRound:
    def __init__(
        self,
        set_of_players,
        previous_games_rounds_anti_chron=[],
        teams_per_game=2,
        players_per_team=2,
        amount_of_games=None,
        preference=None,
        num_iter=1000,
        level_gap_tol=0.5,
    ):
        self.df = pd.DataFrame([player.series for player in set_of_players])

        if amount_of_games:
            self.amount_of_games = amount_of_games
        else:
            self.amount_of_games = len(set_of_players) // (
                teams_per_game * players_per_team
            )

        self.preference = preference
        self.people_present = {player for player in set_of_players}
        self.people_present_names = {player.name for player in set_of_players}
        self.previous_games = previous_games_rounds_anti_chron
        self.teams_per_game = teams_per_game
        self.players_per_team = players_per_team
        self.num_iter = num_iter
        self.level_gap_tol = level_gap_tol

        self.games = []
        self.create_games()

        self.not_playing = self.people_present - set().union(
            *{game.participants for game in self.games}
        )

    def create_set_of_all_possible_teams(self):
        # function that creates all possible teams of <players_per_team> players from a set of players
        return {
            TeamOfTwo(*team)
            for team in combinations(self.people_playing, self.players_per_team)
        }

    def create_games(self):
        import random

        ####removing players amongst the ones that had played the most##########
        # amount of players that wil not play
        amount_non_playing = len(self.people_present_names) % (
            self.players_per_team * self.teams_per_game
        )

        # we split the players by amount of games played
        all_amounts_of_games_played = {
            person.games_played for person in self.people_present
        }

        dic_amount_of_games_played_set_of_players = {
            amount_of_games_played: set()
            for amount_of_games_played in all_amounts_of_games_played
        }

        for player in random.sample(
            list(self.people_present), len(self.people_present)
        ):
            dic_amount_of_games_played_set_of_players[player.games_played].add(player)

        list_descending_priority = [
            player
            for i in sorted(
                dic_amount_of_games_played_set_of_players.keys(), reverse=True
            )
            for player in dic_amount_of_games_played_set_of_players[i]
        ]
        self.people_playing = set(list_descending_priority[amount_non_playing:])

        for player in self.people_playing:
            player.games_played += 1
        ########################################################################

        self.set_of_all_possible_teams = self.create_set_of_all_possible_teams()
        ######preference == none################################################
        # if preference is none, we create random games, trying not to recreate the same games
        if self.preference == None:
            pass
        if self.preference == "balanced":

            # we create the games
            for iter in range(self.num_iter):
                for i in range(self.amount_of_games):
                    people_left_to_play = self.people_playing - set().union(
                        *{game.participants for game in self.games}
                    )
                    self.games.append(self.create_balanced_game(people_left_to_play))

                if max([game.level_difference for game in self.games]) == 0:
                    break
                if (
                    max([game.level_difference for game in self.games])
                    <= self.level_gap_tol
                ) and (iter > self.num_iter // 2):
                    break
                self.games = []

            if self.games == []:
                print("could not find a game, because the tolerance is too low")

        # ####preference == level#################################################
        # if preference is level, we sort the players by level and create games by level
        if self.preference == "level":
            # we find all the possible levels
            self.create_games_by_level()
            ########################################################################

    def create_balanced_game(self, people_left_to_play, balanced=True):

        # setting maximal number of iterations in each case
        if balanced:
            level_diff = 0
            while level_diff < 3:
                for i in range(self.num_iter):
                    teams = []
                    temp_set_of_teams = self.set_of_all_possible_teams.copy()
                    # print([(player.name for player in team.players) for team in temp_set_of_teams])
                    # print([player.name for player in people_left_to_play])
                    temp_set_of_teams = {
                        team
                        for team in temp_set_of_teams
                        if all(player in people_left_to_play for player in team.players)
                    }
                    # print(temp_set_of_teams)
                    set_of_chosen_teams = set()
                    for team_iter in range(self.teams_per_game):
                        team = np.random.choice(list(temp_set_of_teams))
                        if (
                            set()
                            .union(
                                *(
                                    other_team.players
                                    for other_team in set_of_chosen_teams
                                )
                            )
                            .intersection(set(team.players))
                            == set()
                        ):
                            set_of_chosen_teams.add(team)
                            temp_set_of_teams = temp_set_of_teams.difference({team})
                    if len(set_of_chosen_teams) == self.teams_per_game:

                        game_of_four = GameOfFour(
                            *set_of_chosen_teams, preference=self.preference
                        )

                        if game_of_four.level_difference <= level_diff:

                            return game_of_four

                level_diff += 1

        else:
            # we pick the first two teams with no intersection###############
            # we randomize the set of teams
            list_of_teams = list(self.set_of_all_possible_teams)
            np.random.shuffle(list_of_teams)

            for i in range(self.num_iter):
                possible_team_combination = combinations(
                    list_of_teams, self.teams_per_game
                )
                for team_combination in possible_team_combination:
                    if (
                        set().union(*set(possible_team_combination))
                        == self.set_of_all_possible_teams
                    ):
                        break
            return GameOfFour(*team_combination, preference="None")
        return "could not find a game, because there are not enough players"

    def create_games_by_level(self):
        # we first sort the players by level
        dic_level_players = {level: [] for level in self.df["Level"].unique()}
        for player in self.people_playing:
            dic_level_players[player.level].append(player)
        # we randomize the players in each level
        for level in dic_level_players.keys():
            np.random.shuffle(dic_level_players[level])
        # we flatten the list of players
        list_of_players_decreasing_order = [
            player
            for level in sorted(dic_level_players.keys(), reverse=True)
            for player in dic_level_players[level]
        ]
        people_per_game = self.teams_per_game * self.players_per_team
        for i in range(self.amount_of_games):
            self.games.append(
                self.create_balanced_game(
                    list_of_players_decreasing_order[
                        i * people_per_game : (i + 1) * people_per_game
                    ]
                )
            )


# %%

set_of_players = {Player(df_minimal_example.iloc[i]) for i in range(19)}
round_of_games = GamesRound(
    set_of_players, preference="level", num_iter=40, level_gap_tol=2
)
for attr in ["amount_of_games", "preference"]:
    print(attr + " : ", getattr(round_of_games, attr))
print("not playing:", [player.name for player in round_of_games.not_playing])
i = 1
for game in round_of_games.games:
    print("_______", "game", i, "_______")
    print([team.players_name for team in game.teams])
    for attr in ["preference", "level_difference"]:
        print(attr + " : ", getattr(game, attr))
    i += 1

# %%

set_of_players = {Player(main_df.iloc[i]) for i in range(12)}
round_of_games = GamesRound(
    set_of_players, preference="level", level_gap_tol=2, num_iter=40
)

for attr in ["amount_of_games", "preference"]:
    print(attr + " : ", getattr(round_of_games, attr))
print("not playing:", [player.name for player in round_of_games.not_playing])
i = 1
for game in round_of_games.games:
    print("_______", "game", i, "_______")
    print([team.players_name for team in game.teams])
    for attr in ["level_difference"]:
        print(attr + " : ", getattr(game, attr))
    i += 1
# %%
for player in set_of_players:
    print(player.games_played)
# %%
###########################################################################################
#                                                                                         #
#  ###### ####### ####### ####### ####### ######  ####### #     # ##    # #####    ###### #
# #       #          #    #     # #       #     # #     # #     # # #   # #    #  #       #
#  ####   #######    #    #     # #####   ######  #     # #     # #  #  # #     #  ####   #
#       # #          #    #     # #       #   ##  #     # #     # #   # # #    #        # #
# ######  #######    #    ####### #       #    ## ####### ####### #    ## #####   ######  #
#                                                                                         #
###########################################################################################


class SessionOfRounds:
    def __init__(
        self,
        set_of_players,
        amount_of_rounds=1,
        games_per_round_each_round=None,
        players_per_team_each_round=None,
        preferences=[None],
        level_gap_tol=0.5,
        num_iter=40,
    ):
        self.amount_of_rounds = amount_of_rounds
        self.games_per_round_each_round = games_per_round_each_round
        self.players_per_team_each_round = players_per_team_each_round
        self.preferences = preferences
        self.level_gap_tol = level_gap_tol
        self.num_iter = num_iter

        self.players = set_of_players
        self.players_name = {player.name for player in set_of_players}
        self.rounds = []

        # reformatting preferences to the amount of preferences wanted
        if preferences is None:
            preferences = [None] * amount_of_rounds
        elif isinstance(preferences, str):
            preferences = [preferences] * amount_of_rounds
        elif len(preferences) < amount_of_rounds:
            preferences = preferences + [None] * (amount_of_rounds - len(preferences))
        elif len(preferences) > amount_of_rounds:
            return "too many preferences"
        self.preferences = preferences

        # reformatting games_per_round_at_each_round to the amount of rounds wanted
        if games_per_round_each_round is None:
            maximal_games_per_round = len(set_of_players) // 4
            games_per_round_each_round = [maximal_games_per_round] * amount_of_rounds
        elif isinstance(games_per_round_each_round, int):
            games_per_round_each_round = [games_per_round_each_round] * amount_of_rounds
        elif len(games_per_round_each_round) < amount_of_rounds:
            games_per_round_each_round = games_per_round_each_round + [
                games_per_round_each_round[-1]
            ] * (amount_of_rounds - len(games_per_round_each_round))

        self.games_per_round_each_round = games_per_round_each_round

        # reformatting players_per_team_each_round to the amount of rounds wanted
        if players_per_team_each_round is None:
            players_per_team_each_round = [2] * self.amount_of_rounds

        elif isinstance(players_per_team_each_round, int):
            players_per_team_each_round = [
                players_per_team_each_round
            ] * amount_of_rounds
        elif len(players_per_team_each_round) < amount_of_rounds:
            players_per_team_each_round = players_per_team_each_round + [2] * (
                amount_of_rounds - len(players_per_team_each_round)
            )

        self.players_per_team_each_round = players_per_team_each_round

        self.create_rounds()

    def create_rounds(self):
        # function that creates a list of rounds
        rounds = []
        for i in range(self.amount_of_rounds):
            rounds.append(
                GamesRound(
                    set_of_players=self.players,
                    amount_of_games=self.games_per_round_each_round[i],
                    players_per_team=self.players_per_team_each_round[i],
                    preference=self.preferences[i],
                    level_gap_tol=self.level_gap_tol,
                    num_iter=self.num_iter,
                )
            )
        self.rounds = rounds


# %%
set_of_players = {Player(main_df.iloc[i]) for i in range(15)}
session_of_rounds = SessionOfRounds(
    set_of_players,
    amount_of_rounds=4,
    preferences=["balanced", "balanced", "level", "level"],
    level_gap_tol=1,
    num_iter=40,
)
print("all players:", session_of_rounds.players_name)


i = 1
for round in session_of_rounds.rounds:
    print("")
    print("_______", "round", i, "_______")
    print("preference : ", getattr(round, "preference"))
    print("not playing:", [player.name for player in round.not_playing])
    j = 1
    for game in round.games:
        print("-------", "game", j, "-------")
        print([team.players_name for team in game.teams])
        if game.preference == "balanced":
            print("level_difference : ", getattr(game, "level_difference"))
        if game.preference == "level":
            for player in game.participants:
                print("name : ", player.name, "level : ", player.level)
        j += 1
    i += 1
print("")
print("##########STATS END OF SESSION##########")
for player in set_of_players:
    print(player.name, "played", player.games_played, "games")
# %%
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
###                                                                         ####
### ####### #     # ##    #  ###### ####### ####### ####### ##    #  ###### ####
### #       #     # # #   # #          #       #    #     # # #   # #       ####
### #####   #     # #  #  # #          #       #    #     # #  #  #  ####   ####
### #       #     # #   # # #          #       #    #     # #   # #       # ####
### #       ####### #    ##  ######    #    ####### ####### #    ## ######  ####
###                                                                         ####
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
# %%
df_25_03_25 = main_df.loc[[
    "VictorDa",
    "Linda",
    "Marc",
    "Nathan",
    "Luciano",
    "Luca",
    "Manon",
    "Felix",
    "Gabriel",
    "Leo",
    "Colin",
    "David",
    "Aliénor",
    "VictorDi"
]]
# %%
df_25_03_25.sort_values("Level", inplace=True,ascending=False)
team_DLMLM = {Player(df_25_03_25.loc[name]) for name in ["Leo","David","Marc","Luca","Manon"]}

team_FLAVN = {Player(df_25_03_25.loc[name]) for name in ["VictorDi","Nathan","Felix","Luciano","Aliénor"]}
team_LCGV = {Player(df_25_03_25.loc[name]) for name in ["VictorDa","Linda","Colin","Gabriel"]}

team_DLMLM_level = np.mean([player.level for player in team_DLMLM])
team_FLAVN_level = np.mean([player.level for player in team_FLAVN])
team_LCGV_level = np.mean([player.level for player in team_LCGV])
for level in [team_DLMLM_level,team_FLAVN_level,team_LCGV_level]:
    print(level)
# %%