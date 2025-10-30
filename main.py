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
import random

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
        "Noisy level": [0] * 26,
        "Gender": ["Female", "Male"] * 13,
        "Games played": [0] * 26,
        "Happiness": [0] * 26,
    }
).astype(
    {
        "Name": str,
        "Surname": str,
        "Level": float,
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
###############    #    ##   ## #     #         #        #   #  ################
###############    #    #  #  # #######         #     #    #   ################
###############    #    #     # #               #        #   # ################
############### ####### #     # #       ##      #####   #     ################
###############                                                 ################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
# %%


main_df = pd.read_excel("Adhésion au GRNA 2025-2026 (Responses).xlsx")
level_df = pd.read_excel("4_Niveau.xlsx")
spectrum_df = pd.read_excel("spectre.xlsx")
for data_frame in [main_df, level_df, spectrum_df]:
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
for ind in level_df.index:
    if ind not in main_df.index:
        print(f"Could not find {ind} in main_df, adding it")
        main_df.loc[ind] = 0
        for col in main_df.columns:
            if col in level_df.columns:
                main_df.loc[ind, col] = level_df.loc[ind, col]


main_df = main_df.loc[level_df.index]

main_df["Niveau"] = level_df["Niveau moyenné"]
main_df.dropna(axis=0, subset=["Niveau"], inplace=True)
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
    "Level": float,
}


for col in true_dtypes.keys():
    if true_dtypes[col] == datetime.datetime:
        main_df[col] = pd.to_datetime(main_df[col])
    else:
        try:
            main_df[col] = main_df[col].astype(true_dtypes[col])
        except Exception:
            print(f"Could not convert column {col} to {true_dtypes[col]}")

# setting default category as level
main_df["Category"] = main_df["Level"]
# setting default Happiness to 0
main_df["Happiness"] = 0
# setting default games played to 0
main_df["Games played"] = 0

# setting default Noisy level to 0
main_df["Noisy level"] = 0

main_df[["Masochiste", "Équilibré", "Challenger", "Chill", "Sadique"]] = spectrum_df[
    ["Masochiste", "Équilibré", "Challenger", "Chill", "Sadique"]
].fillna(0)

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
        """
        creating Player class from series. The Series should contain at least the following columns:
        - Level : float. The level of the player
        - Gender : str. The gender of the player (currently "Male" and "Female")
        - Happiness: float. The happiness of the player (normally at 0)
        - Games played: int. The number of games played by the player (normally at 0)
        """
        # Initialize the Player object with a pandas Series
        self.series = series
        # Set attributes for each key in the series
        for key in series.keys():
            setattr(self, re.sub(" ", "_", key.lower()), series[key])
        # Set the name attribute
        self.name = series.name
        # Sets a "noisy level" to randomize a bit the level of the player. Especially useful for games by level
        # the initial noisy level is set to the level of the player, and will change depending on what type of games the player will engage in
        self.noisy_level = series["Level"]
        # we also created a rounded noisy level, to avoid too many categories of levels
        # default is the level of the player
        self.rounded_noisy_level = series["Level"]
        self.happiness = series.get("Happiness", 0)
        self.previous_happiness = self.happiness  # <-- Add this line
        self.maso = series.get("Masochiste", 0)
        self.equilibre = series.get("Équilibré", 0)
        self.challenger = series.get("Challenger", 0)
        self.chill = series.get("Chill", 0)
        self.sadique = series.get("Sadique", 0)
        self.games_played = series.get("Games played", 0)

    def update_happiness(
        self,
        game_mean_level,
        teammates_levels,
        opponents_levels,
        level_gap_tol,
        players_chill,
        session_median,
        same_teammate=False,
        spectrum=False,
    ):
        if spectrum:
            team_level = np.mean(teammates_levels + [self.level])
            opponents_mean_level = np.mean(opponents_levels)
            normalization_factor = sum(
                [self.maso, self.equilibre, self.challenger, self.chill, self.sadique]
            )
            temp_hapiness = 0
            temp_hapiness += self.maso * (
                1 if np.mean(opponents_levels) >= 0.7 * self.level else 0
            )
            temp_hapiness += self.equilibre * (
                1
                if abs(team_level - opponents_mean_level) <= 0.5 * level_gap_tol
                else 0
            )
            temp_hapiness += self.challenger * (
                1
                if abs(0.9 * opponents_mean_level - team_level) <= 0.5 * level_gap_tol
                else 0
            )
            temp_hapiness += self.chill * (1 if players_chill >= 10 else 0)
            temp_hapiness += self.sadique * (
                1 if np.mean(opponents_levels) <= team_level else 0
            )
            self.happiness += temp_hapiness / normalization_factor
            self.happiness = round(self.happiness, 2)
        else:
            # More nuanced happiness calculation for higher level players
            high_level_teammates = sum(
                1 for level in teammates_levels if level >= (self.level * 0.85)
            )
            high_level_opponents = sum(
                1 for level in opponents_levels if level >= (self.level * 0.85)
            )

            # Higher level players are happier with competitive matches
            self.happiness += high_level_teammates + high_level_opponents

            # Penalize if consistently playing with much lower level players
            # if np.mean(teammates_levels + opponents_levels) < (self.level * 0.85):
            #     self.happiness -= 1
        if same_teammate:
            self.happiness -= 1


# %% Example usage of the Player class
if __name__ == "__main__":
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

    def same_players(self, other_team):
        # Check if the two teams have the same players
        return self.players == other_team.players


# %%
team_of_two = TeamOfTwo(
    Player(df_minimal_example.iloc[0]),
    Player(df_minimal_example.iloc[1]),
    preference="Level",
)
# for attr in [
#     "preference",
#     "players_name",
#     "level_difference",
#     "mixed",
#     "male",
#     "female",
#     "non_binary",
# ]:
#     print(attr + " : ", getattr(team_of_two, attr))


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

        # Calculate and store mean levels
        self.team_A_mean_level = self.team_A.mean_level
        self.team_B_mean_level = self.team_B.mean_level
        self.overall_mean_level = (self.team_A_mean_level + self.team_B_mean_level) / 2

        self.level_difference = np.round(
            abs(self.team_A_mean_level - self.team_B_mean_level), 2
        )

    def update_players_happiness(
        self, session_median_level, level_gap_tol, spectrum, teammate_history
    ):
        """Update happiness for all players in the game, with penalty for repeated teammates"""
        for team in [self.team_A, self.team_B]:
            players = list(team.players)
            for i, player in enumerate(players):
                teammates_levels = [p.level for j, p in enumerate(players) if j != i]
                other_team = self.team_B if team == self.team_A else self.team_A
                opponents_levels = [p.level for p in other_team.players]
                same_teammate = any(
                    frozenset([player.name, teammate.name]) in teammate_history
                    for teammate in team.players
                    if teammate != player
                )
                total_players_chill = sum(p.chill for p in players)
                player.update_happiness(
                    game_mean_level=self.overall_mean_level,
                    teammates_levels=teammates_levels,
                    opponents_levels=opponents_levels,
                    level_gap_tol=level_gap_tol,
                    players_chill=total_players_chill,
                    session_median=session_median_level,
                    same_teammate=same_teammate,
                    spectrum=spectrum,
                )
                # Penalize if this pair has already played together


# %% Example usage of the GameOfFour class
if __name__ == "__main__":
    example_one_game_df = df_minimal_example.iloc[:4]
    game_of_four = GameOfFour(
        TeamOfTwo(
            Player(example_one_game_df.iloc[0]), Player(example_one_game_df.iloc[1])
        ),
        TeamOfTwo(
            Player(example_one_game_df.iloc[2]), Player(example_one_game_df.iloc[3])
        ),
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
        list_of_players,
        previous_games_rounds_anti_chron=[],
        teams_per_game=2,
        players_per_team=2,
        amount_of_games=None,
        preference=None,
        num_iter=40,
        level_gap_tol=0.5,
        seed=None,
        spectrum=False,
    ):

        if seed is not None:
            random.seed(seed)

        self.df = pd.DataFrame([player.series for player in list_of_players])

        if amount_of_games:
            self.amount_of_games = amount_of_games
        else:
            self.amount_of_games = len(list_of_players) // (
                teams_per_game * players_per_team
            )

        self.preference = preference
        self.people_present = list_of_players
        self.people_present_names = [player.name for player in list_of_players]
        self.previous_games = previous_games_rounds_anti_chron
        self.previous_teams = set(
            team for round in previous_games_rounds_anti_chron for team in round.teams
        )
        self.teams_per_game = teams_per_game
        self.players_per_team = players_per_team
        self.num_iter = num_iter
        self.level_gap_tol = level_gap_tol
        self.spectrum = spectrum

        # Create a set of player pairs that have played together
        self.teammate_history = []
        for round in previous_games_rounds_anti_chron:
            for game in round.games:
                for team in game.teams:
                    for player1, player2 in combinations(
                        [p.name for p in team.players], 2
                    ):
                        self.teammate_history.append(frozenset([player1, player2]))
        self.games = []
        self.session_median_level = np.median(
            [player.level for player in list_of_players]
        )
        self.create_games()

        self.not_playing = [
            person for person in list_of_players if person not in self.people_playing
        ]

    def create_set_of_all_possible_teams(self, remove_previous_teams=True):
        # function that creates all possible teams of <players_per_team> players from a set of players
        set_of_all_possible_teams = set(
            TeamOfTwo(*team)
            for team in combinations(self.people_playing, self.players_per_team)
        )
        # remove teams that have already played together
        if remove_previous_teams:
            set_of_all_possible_teams = set(
                team
                for team in set_of_all_possible_teams
                if not any([team.same_players(team2) for team2 in self.previous_teams])
            )
        return set_of_all_possible_teams

    def create_games(self, level_gap_tol=0.5, spectrum=False):

        ####removing players amongst the ones that had played the most##########
        # amount of players that will not play
        amount_non_playing = len(self.people_present_names) % (
            self.players_per_team * self.teams_per_game
        )

        # we split the players by amount of games played
        all_amounts_of_games_played = {
            person.games_played for person in self.people_present
        }

        dic_amount_of_games_played_list_of_players = {
            amount_of_games_played: []
            for amount_of_games_played in all_amounts_of_games_played
        }
        shuffled_players = self.people_present.copy()
        random.shuffle(shuffled_players)

        for player in shuffled_players:
            dic_amount_of_games_played_list_of_players[player.games_played].append(
                player
            )

        # Sort each group of players with the same games played by increasing happiness
        for k in dic_amount_of_games_played_list_of_players:
            dic_amount_of_games_played_list_of_players[k].sort(
                key=lambda p: p.happiness
            )

        list_descending_priority = [
            player
            for i in sorted(
                dic_amount_of_games_played_list_of_players.keys(), reverse=True
            )
            for player in dic_amount_of_games_played_list_of_players[i]
        ]
        self.people_playing = set(list_descending_priority[amount_non_playing:])

        for player in self.people_playing:
            player.previous_happiness = player.happiness
            player.games_played += 1
        ########################################################################

        self.set_of_all_possible_teams = self.create_set_of_all_possible_teams()
        ######preference == none################################################
        # if preference is none, we create random games, trying not to recreate the same games
        if isinstance(self.preference, dict):
            preference_type = self.preference.get("type")
            kwargs = self.preference.get("kwargs")
        else:
            preference_type = self.preference
            kwargs = {}
        if self.preference == None:
            pass
        if preference_type == "balanced":

            # we create the games
            for iter in range(self.num_iter):
                for i in range(self.amount_of_games):
                    people_left_to_play = [
                        people
                        for people in self.people_playing
                        if people
                        not in set().union(*{game.participants for game in self.games})
                    ]
                    self.games.append(
                        self.create_balanced_game(
                            people_left_to_play,
                            level_gap_tol=self.level_gap_tol,
                            spectrum=self.spectrum,
                            **kwargs,
                        )
                    )

                if (
                    max([game.level_difference for game in self.games])
                    <= self.level_gap_tol
                ):
                    break
                if (
                    max([game.level_difference for game in self.games])
                    <= 3 * self.level_gap_tol
                ) and (iter > self.num_iter // 2):
                    break
                self.games = []

            if self.games == []:
                print("could not find a game, because the tolerance is too low")

        if preference_type == "level":
            self.create_games_by_level(**kwargs)
        self.teams = set()
        for game in self.games:
            self.teams = self.teams.union(game.teams)

    def create_balanced_game(
        self, people_left_to_play, level_gap_tol, spectrum, mixed=False, **kwargs
    ):
        available_players = list(people_left_to_play)
        if len(available_players) < self.teams_per_game * self.players_per_team:
            return "could not find a game, because there are not enough players"

        best_game = None
        best_happiness_score = -1

        current_gap_tol = self.level_gap_tol
        max_attempts = 3

        for attempt in range(max_attempts):
            for _ in range(self.num_iter):
                random.shuffle(available_players)
                teams = []
                for i in range(
                    0,
                    self.teams_per_game * self.players_per_team,
                    self.players_per_team,
                ):
                    team_players = available_players[i : i + self.players_per_team]
                    team = TeamOfTwo(*team_players)
                    teams.append(team)

                game = GameOfFour(*teams, preference=self.preference)

                if game.level_difference <= current_gap_tol:
                    # --- Save previous happiness ---
                    for team in teams:
                        for player in team.players:
                            player.previous_happiness = player.happiness

                    # --- Simulate happiness update ---
                    game.update_players_happiness(
                        self.session_median_level,
                        level_gap_tol,
                        spectrum,
                        self.teammate_history,
                    )

                    # --- Calculate happiness score ---
                    happiness_score = sum(
                        player.happiness for team in teams for player in team.players
                    )

                    # --- Revert happiness ---
                    for team in teams:
                        for player in team.players:
                            player.happiness = player.previous_happiness

                    if happiness_score > best_happiness_score:
                        best_happiness_score = happiness_score
                        best_game = game

        if best_game:
            best_game.update_players_happiness(
                self.session_median_level,
                self.level_gap_tol,
                self.spectrum,
                self.teammate_history,
            )
            return best_game
        else:
            # fallback: just create a random game
            random.shuffle(available_players)
            teams = []
            for i in range(
                0, self.teams_per_game * self.players_per_team, self.players_per_team
            ):
                team_players = available_players[i : i + self.players_per_team]
                team = TeamOfTwo(*team_players)
                teams.append(team)
            game = GameOfFour(*teams, preference=self.preference)
            return game

    def create_games_by_level(self, alternate=False, mixed=False, **kwargs):

        sorted_players = sorted(
            self.people_playing,
            key=lambda p: (round(p.level * 2) / 2, -p.happiness),
            reverse=True,
        )
        all_players_in_each_game = [
            [
                player
                for player in sorted_players[
                    i : i + self.players_per_team * self.teams_per_game
                ]
            ]
            for i in range(
                0, len(sorted_players), self.players_per_team * self.teams_per_game
            )
        ]

        for game_num, players_in_game in enumerate(all_players_in_each_game):
            start_idx = game_num * self.players_per_team * self.teams_per_game

            if alternate:
                # Divide players into teams by alternating
                team1_players = players_in_game[0::2][: self.players_per_team]
                team2_players = players_in_game[1::2][: self.players_per_team]
            else:
                # Divide players into teams by putting 0 and 4 together, and 2 and 3 together
                # WORKS ONLY FOR 4 PLAYERS
                if len(players_in_game) == 4:
                    team1_players = [players_in_game[i] for i in [0, 3]]
                    team2_players = [players_in_game[i] for i in [1, 2]]
                else:
                    print(
                        "Error: Game creation by level requires exactly 4 players per game."
                        "please put alternate to True if you want to create games with more than 4 players"
                    )

            # Generate all possible (team1, team2) combinations as TeamOfTwo objects
            alternative_possible_teams = list(
                combinations(players_in_game, self.players_per_team)
            )
            possible_team_pairs_with_level_diff = []
            for team1_players in alternative_possible_teams:
                team2_players = tuple(
                    player for player in players_in_game if player not in team1_players
                )
                if len(team2_players) == self.players_per_team:
                    team1 = TeamOfTwo(*team1_players)
                    team2 = TeamOfTwo(*team2_players)
                    level_diff = abs(team1.mean_level - team2.mean_level)
                    possible_team_pairs_with_level_diff.append(
                        (team1, team2, level_diff)
                    )

            # Sort by difference of levels
            possible_team_pairs_with_level_diff.sort(key=lambda x: x[2])
            # Pick the first one where neither team was in previous_teams
            found = False
            prev_team_sets = [team.players for team in self.previous_teams]
            for team1, team2, _ in possible_team_pairs_with_level_diff:
                if (
                    team1.players not in prev_team_sets
                    and team2.players not in prev_team_sets
                ):
                    team1_obj, team2_obj = team1, team2
                    found = True
                    break
            else:
                # fallback: just use the first possible pair
                team1_obj, team2_obj, _ = possible_team_pairs_with_level_diff[0]
            team1_players = list(team1_obj.players)
            team2_players = list(team2_obj.players)

            if (
                len(team1_players) == self.players_per_team
                and len(team2_players) == self.players_per_team
            ):
                team1 = TeamOfTwo(*team1_players)
                team2 = TeamOfTwo(*team2_players)
                game = GameOfFour(team1, team2, preference=self.preference)
                self.games.append(game)
                game.update_players_happiness(
                    self.session_median_level,
                    self.level_gap_tol,
                    self.spectrum,
                    self.teammate_history,
                )


# %%
if __name__ == "__main__":
    list_of_players = [Player(df_minimal_example.iloc[i]) for i in range(19)]
    round_of_games = GamesRound(
        list_of_players,
        preference={"type": "level", "kwargs": {"randomize": False}},
        num_iter=40,
        level_gap_tol=2,
        seed=0,
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
if __name__ == "__main__":
    list_of_players = [Player(main_df.loc[name]) for name in main_df.iloc[10:21].index]
    round_of_games = GamesRound(
        list_of_players, preference="balanced", level_gap_tol=2, num_iter=40, seed=1
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
###########################################################################################
#                                                                                         #
#  ###### ####### ####### ####### ####### ######  ####### #     # ##    # #####    ###### #
# #       #          #    #     # #       #     # #     # #     # # #   # #    #  #       #
#  ####   #######    #    #     # #####   ######  #     # #     # #  #  # #     #  ####   #
#       # #     #    #    #     # #       #   ##  #     # #     # #   # # #    #        # #
# ######  #######    #    ####### #       #    ## ####### ####### #    ## #####   ######  #
#                                                                                         #
###########################################################################################


class SessionOfRounds:
    def __init__(
        self,
        list_of_players,
        amount_of_rounds=1,
        games_per_round_each_round=None,
        players_per_team_each_round=None,
        preferences=[None],
        level_gap_tol=0.5,
        num_iter=40,
        spectrum=False,
        seed=None,
    ):
        self.amount_of_rounds = amount_of_rounds
        self.games_per_round_each_round = games_per_round_each_round
        self.players_per_team_each_round = players_per_team_each_round
        self.preferences = preferences
        self.level_gap_tol = level_gap_tol
        self.num_iter = num_iter

        self.players = list_of_players
        self.players_name = [player.name for player in list_of_players]
        self.rounds = []
        self.spectrum = spectrum

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
            maximal_games_per_round = len(list_of_players) // 4
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

        self.create_rounds(seed=seed)

    def create_rounds(self, seed=None):
        # function that creates a list of rounds
        rounds = []
        for i in range(self.amount_of_rounds):
            if seed is not None:
                round_seed = seed + i
            else:
                round_seed = None
            rounds.append(
                GamesRound(
                    list_of_players=self.players,
                    amount_of_games=self.games_per_round_each_round[i],
                    players_per_team=self.players_per_team_each_round[i],
                    previous_games_rounds_anti_chron=rounds,
                    preference=self.preferences[i],
                    level_gap_tol=self.level_gap_tol,
                    num_iter=self.num_iter,
                    seed=round_seed,
                    spectrum=self.spectrum,
                )
            )
        # Calculate happiness inequality before each round
        for i in range(self.amount_of_rounds):
            # Sort players by happiness to prioritize less happy players
            sorted_players = sorted(self.players, key=lambda p: p.happiness)

            # Give priority to players with lower happiness scores
            if i > 0:  # Skip for first round since all start at 0 happiness
                # Assign temporary boost to level for less happy players
                for idx, player in enumerate(sorted_players):
                    boost_factor = 1 + (
                        0.2 * (len(sorted_players) - idx) / len(sorted_players)
                    )
                    player.temp_boost = boost_factor

            # Reset temporary boosts
            for player in self.players:
                player.temp_boost = 1.0
        self.rounds = rounds
        self.mean_happiness = np.mean([player.happiness for player in self.players])
        self.max_and_min_happiness = (
            np.max([player.happiness for player in self.players]),
            np.min([player.happiness for player in self.players]),
        )
        self.max_happiness_difference = (
            self.max_and_min_happiness[0] - self.max_and_min_happiness[1]
        )
        self.std_happiness = np.std([player.happiness for player in self.players])

    # %%
    ################################################################################
    #####                                                                      #####
    ##### ####### ######  ####### ##    # #######       #####  #     # #     # #####
    ##### #     # #     #    #    # #   #    #         #     # #     #  #   #  #####
    ##### ####### ######     #    #  #  #    #         ####### #     #    #    #####
    ##### #       #   ##     #    #   # #    #         #     # #     #  #   #  #####
    ##### #       #    ## ####### #    ##    #         #     # ####### #     # #####
    #####                                                                      #####
    ################################################################################

    def count_all_pairs(self, order_num_list=None):
        # Find and display players who played at least twice with each other and record the rounds
        player_pairs = {}
        pair_rounds = {}
        # Use the possibly reordered rounds for analysis
        round_copy = self.rounds.copy()
        if order_num_list is not None:
            round_copy = [round_copy[i - 1] for i in order_num_list]
        for round_index, round in enumerate(round_copy, start=1):
            for game in round.games:
                for team in game.teams:
                    for player_a, player_b in combinations(team.players, 2):
                        pair = frozenset([player_a.name, player_b.name])
                        player_pairs[pair] = player_pairs.get(pair, 0) + 1
                        if pair not in pair_rounds:
                            pair_rounds[pair] = []
                        pair_rounds[pair].append(round_index)
        return player_pairs, pair_rounds

    def add_team_repetition_to_output(
        self, output, player_pairs, pair_rounds, minimum=2
    ):
        teams_to_repetitions = {}
        output.append(
            f"\n#######PLAYERS WHO PLAYED TOGETHER AT LEAST {minimum} TIMES #######"
        )
        for pair, count in player_pairs.items():
            if count >= minimum:
                rounds = ", ".join(map(str, pair_rounds[pair]))
                output.append(
                    f"{', '.join(pair)} played together {count} times in rounds: {rounds}"
                )
                teams_to_repetitions[pair] = count
        return output, teams_to_repetitions

    # %%
    ################################################################################
    #                                                                              #
    # ####### ######  ####### ##    # #######      #     #  #####  ####### ##    # #
    # #     # #     #    #    # #   #    #         ##   ## #     #    #    # #   # #
    # ####### ######     #    #  #  #    #         #  #  # #######    #    #  #  # #
    # #       #   ##     #    #   # #    #         #     # #     #    #    #   # # #
    # #       #    ## ####### #    ##    #         #     # #     # ####### #    ## #
    #                                                                              #
    ################################################################################

    def print_all_results(self, print_levels=True, order_num_list=None):
        import pyperclip

        # Collect all printed information
        output = []
        output.append("all players: " + str(self.players_name))
        i = 1
        round_copy = self.rounds.copy()
        if order_num_list is not None:
            round_copy = [round_copy[i - 1] for i in order_num_list]
        for round in round_copy:
            output.append("\n")
            output.append(f"{i} " * 8 + "ROUND " + f"{i} " * 8)
            output.append("preference : " + str(getattr(round, "preference")))
            output.append(
                "not playing: " + str([player.name for player in round.not_playing])
            )
            j = 1
            for game in round.games:
                output.append(f"------- game {j} -------")
                output.append(str([team.players_name for team in game.teams]))
                if game.preference == "balanced":
                    output.append(
                        "level_difference : "
                        + str(np.round(getattr(game, "level_difference"), 2))
                    )
                    if print_levels:
                        for player in game.participants:
                            output.append(
                                f"name : {player.name}, level : {player.level}, happiness gained : {np.round(player.happiness - getattr(player, 'previous_happiness', 0), 2)}"
                            )
                if print_levels:
                    if game.preference == "level":
                        for player in game.participants:
                            output.append(
                                f"name : {player.name}, level : {player.level}, happiness gained : {np.round(player.happiness - getattr(player, 'previous_happiness', 0), 2)}"
                            )
                        output.append(
                            f"level difference : {np.round(game.level_difference, 2)}"
                        )
                j += 1
            output.append(f"{i} " * 8 + "ROUND END " + f"{i} " * 8)
            output.append("\n\n\n")
            i += 1

        output.append("\n#####AMOUNT OF GAMES PLAYED#####")
        for player in self.players:
            output.append(
                f"{player.name} played {player.games_played} games, happiness: {np.round(player.happiness, 2)}"
            )
        #####ADDD THE COUNT OF PLAYERS THAT PLAYED TOGETHER AT LEAST TWICE AND SO ON, REPLACE FROM PRINT_ALL_RESULTS#####
        player_pairs, pair_rounds = self.count_all_pairs(order_num_list)

        output, _ = self.add_team_repetition_to_output(
            output, player_pairs, pair_rounds, minimum=2
        )
        # Find and display players who played against each other at least twice and record the rounds
        opponent_pairs = {}
        opponent_pair_rounds = {}
        for round_index, round in enumerate(round_copy, start=1):
            for game in round.games:
                team_A, team_B = list(game.teams)
                for player_a in team_A.players:
                    for player_b in team_B.players:
                        pair = frozenset([player_a.name, player_b.name])
                        opponent_pairs[pair] = opponent_pairs.get(pair, 0) + 1
                        if pair not in opponent_pair_rounds:
                            opponent_pair_rounds[pair] = []
                        opponent_pair_rounds[pair].append(round_index)

        output.append(
            "\n##########PLAYERS WHO PLAYED AGAINST EACH OTHER AT LEAST THRICE##########"
        )
        for pair, count in opponent_pairs.items():
            if count >= 3:
                rounds = ", ".join(map(str, opponent_pair_rounds[pair]))
                output.append(
                    f"{', '.join(pair)} played against each other {count} times in rounds: {rounds}"
                )

        # Add happiness analytics section
        output.append("\n#####HAPPINESS ANALYTICS#####")

        # Calculate happiness statistics
        happiness_values = [player.happiness for player in self.players]
        output.append(f"Average happiness: {np.round(np.mean(happiness_values), 2)}")
        output.append(
            f"Happiness standard deviation: {np.round(np.std(happiness_values), 2)}"
        )
        output.append(f"Min happiness: {np.round(min(happiness_values), 2)}")
        output.append(f"Max happiness: {np.round(max(happiness_values), 2)}")

        # Identify happiest and least happy players
        happiest_players = sorted(
            self.players, key=lambda p: p.happiness, reverse=True
        )[:3]
        least_happy_players = sorted(self.players, key=lambda p: p.happiness)[:3]

        output.append("\nHappiest players:")
        for player in happiest_players:
            output.append(f"{player.name}: {np.round(player.happiness, 2)}")

        output.append("\nLeast happy players:")
        for player in least_happy_players:
            output.append(f"{player.name}: {np.round(player.happiness, 2)}")

        pyperclip.copy("\n".join(output))
        print("\n".join(output))
        print("\n\n\n\nALL RESULTS HAVE BEEN COPIED TO THE CLIPBOARD.")


# %%
# choose randomly 12 numbers between 0 and 31, without repetition
import random

# numbers = random.sample(range(0, 31), 12)
good_numbers = [1, 3, 4, 5, 7, 9]  # , 12, 15, 16, 17, 23, 27]
if __name__ == "__main__":
    list_of_players = [
        Player(main_df.loc[name]) for name in main_df.iloc[good_numbers].index
    ]
    session_of_rounds = SessionOfRounds(
        list_of_players,
        amount_of_rounds=6,
        preferences=["balanced"] * 3 + ["level"] * 3,
        level_gap_tol=1.5,
        num_iter=50,
        seed=3,
    )
    # %%
    # session_of_rounds.print_all_results()
