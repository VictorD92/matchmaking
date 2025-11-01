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
from unidecode import unidecode

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
for name in main_df.index:
    if (
        sum(
            main_df.loc[
                name, ["Masochiste", "Équilibré", "Challenger", "Chill", "Sadique"]
            ]
        )
        == 0
    ):
        main_df.loc[
            name, ["Masochiste", "Équilibré", "Challenger", "Chill", "Sadique"]
        ] = 5

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
        self.alchimiste = series.get("Alchimiste", 0)
        self.last_spec_chosen = None
        self.spec_chosen_history = []

        self.last_happiness_gained = 0
        self.happiness_gained_history = []

        self.games_played = series.get("Games played", 0)
        self.teammate_history = []
        self.other_players_in_game_history = []

    def update_happiness(
        self,
        game_mean_level,
        teammates_levels,
        opponents_levels,
        level_gap_tol,
        is_gender_preference_satisfied,
        players_chill,
        session_median,
        weight_same_teammate=4,
        amount_same_other_players=0,
        spectrum=False,
        seed=None,
    ):
        initial_happiness = self.happiness

        if spectrum:
            teammates_mean_level = np.mean(teammates_levels)
            team_level = np.mean(teammates_levels + [self.level])
            opponents_mean_level = np.mean(opponents_levels)
            spectrum_game = {
                "Maso": 1 if 0.7 * opponents_mean_level >= self.level else 0,
                "Équilibré": (
                    1
                    if abs(team_level - opponents_mean_level) <= 0.5 * level_gap_tol
                    else 0
                ),
                "Challenger": (
                    1
                    if abs(0.9 * opponents_mean_level - team_level)
                    <= 0.5 * level_gap_tol
                    else 0
                ),
                "Chill": 1 if players_chill >= 10 else 0,
                "Sadique": 1 if opponents_mean_level <= team_level else 0,
                "Alchimiste": (
                    1
                    if abs(self.level - teammates_mean_level) <= 0.5 * level_gap_tol
                    else 0
                ),
            }
            best_gain = 0
            specs_with_best_gain = []
            # Sort specs to ensure consistent ordering
            for spec in sorted(spectrum_game.keys()):
                spec_gain = getattr(self, unidecode(spec.lower())) * spectrum_game[spec]
                if spec_gain > best_gain:
                    specs_with_best_gain = [spec]
                    best_gain = spec_gain
                elif spec_gain == best_gain:
                    specs_with_best_gain.append(spec)

            spec_chosen = random.choice(specs_with_best_gain)
            self.happiness += (
                getattr(self, unidecode(spec_chosen.lower()))
                * spectrum_game[spec_chosen]
            )
            self.last_spec_chosen = spec_chosen

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
            self.last_spec_chosen = None

        self.happiness -= weight_same_teammate

        self.happiness -= weight_same_teammate / 3 * amount_same_other_players

        if not is_gender_preference_satisfied:
            self.happiness -= 2
        # Store the happiness gained for this game
        self.last_happiness_gained = self.happiness - initial_happiness

        return self.last_happiness_gained


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
    def __init__(
        self, player_A, player_B, type_preference=None, gender_preference=None
    ):
        # Initialize the TeamOfTwo object with two players and a type_preference
        self.type_preference = type_preference
        self.gender_preference = gender_preference
        # Create a DataFrame with the series of both players
        self.df = pd.concat([player_A.series, player_B.series], axis=1).T
        self.player_A = player_A
        self.player_B = player_B
        # Store the players in a list to maintain order
        self.players = [player_A, player_B]
        self.players_set = {player_A, player_B}  # Keep set for membership tests
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
        return self.players_set == other_team.players_set


# %%
team_of_two = TeamOfTwo(
    Player(df_minimal_example.iloc[0]),
    Player(df_minimal_example.iloc[1]),
    type_preference="Level",
    gender_preference=None,
)
# for attr in [
#     "type_preference",
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
    def __init__(
        self,
        team_A,
        team_B,
        type_preference=None,
        gender_preference=None,
        weight_same_teammate=4,
    ):
        # Initialize the GameOfFour object with two teams and a type_preference
        self.type_preference = type_preference
        self.gender_preference = gender_preference

        # Create a DataFrame with the data of both teams
        self.df = pd.concat([team_A.df, team_B.df], axis=0)

        self.team_A = team_A
        self.team_B = team_B

        # Store the teams in a set of frozensets
        self.teams = set([team_A, team_B])
        # Store the participants of the game
        self.participants = frozenset(
            self.team_A.players_set.union(self.team_B.players_set)
        )

        # Calculate and store mean levels
        self.team_A_mean_level = self.team_A.mean_level
        self.team_B_mean_level = self.team_B.mean_level
        self.overall_mean_level = (self.team_A_mean_level + self.team_B_mean_level) / 2

        self.level_difference = np.round(
            abs(self.team_A_mean_level - self.team_B_mean_level), 2
        )
        self.is_gender_preference_satisfied = (
            False if self.compute_gender_preference_score() == 0 else True
        )
        self.weight_same_teammate = weight_same_teammate

    def compute_gender_preference_score(self):
        # return 1 if gender preference is satisfied, 0 otherwise
        if self.gender_preference == "mixed":
            if self.team_A.mixed and self.team_B.mixed:
                return 1
            else:
                return 0
        elif self.gender_preference == "same":
            if self.team_A.mixed == False and self.team_B.mixed == False:
                return 1
            else:
                return 0
        else:
            return 1

    def update_players_happiness(
        self, session_median_level, level_gap_tol, spectrum, seed=None
    ):
        """Update happiness for all players in the game, with penalty for repeated teammates"""
        for team in [self.team_A, self.team_B]:
            players = list(team.players)
            for i, player in enumerate(players):
                teammates_levels = [p.level for j, p in enumerate(players) if j != i]
                other_team = self.team_B if team == self.team_A else self.team_A
                opponents_levels = [p.level for p in other_team.players]
                weight_same_teammate = self.weight_same_teammate * any(
                    frozenset([player, teammate]) in player.teammate_history
                    for teammate in team.players
                )
                amount_same_other_players = 0
                for other_player in team.players:
                    if other_player != player:
                        for game in player.other_players_in_game_history:
                            if other_player in game:
                                amount_same_other_players += 1

                total_players_chill = sum(p.chill for p in players)
                player.update_happiness(
                    game_mean_level=self.overall_mean_level,
                    teammates_levels=teammates_levels,
                    opponents_levels=opponents_levels,
                    level_gap_tol=level_gap_tol,
                    is_gender_preference_satisfied=self.is_gender_preference_satisfied,
                    players_chill=total_players_chill,
                    session_median=session_median_level,
                    weight_same_teammate=weight_same_teammate,
                    amount_same_other_players=amount_same_other_players,
                    spectrum=spectrum,
                    seed=seed,
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
        type_preference="Level",
        gender_preference="mixed",
    )
    # Print the attributes of the game
    for attr in ["type_preference", "teams", "participants", "level_difference"]:
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
        type_preference=None,
        gender_preference=None,
        num_iter=40,
        level_gap_tol=0.5,
        lambda_weight=2,
        weight_same_teammate=4,
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

        self.type_preference = type_preference
        self.gender_preference = gender_preference
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
        self.lambda_weight = lambda_weight
        # Create a set of player pairs that have played together
        self.teammate_history = []
        for round in previous_games_rounds_anti_chron:
            for game in round.games:
                for team in game.teams:
                    for player1, player2 in combinations([p for p in team.players], 2):
                        self.teammate_history.append(frozenset([player1, player2]))
        self.games = []
        self.session_median_level = np.median(
            [player.level for player in list_of_players]
        )
        self.weight_same_teammate = weight_same_teammate
        self.create_games(seed=seed)

        self.not_playing = [
            person for person in list_of_players if person not in self.people_playing
        ]
        for player in self.not_playing:
            player.spec_chosen_history.append(None)
            player.happiness_gained_history.append(None)

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

    def create_games(self, seed=None):

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
        self.people_playing = list_descending_priority[amount_non_playing:]

        for player in self.people_playing:
            player.previous_happiness = player.happiness
            player.games_played += 1
        ########################################################################

        self.set_of_all_possible_teams = self.create_set_of_all_possible_teams()
        ######type_preference == none################################################
        # if type_preference is none, we create random games, trying not to recreate the same games
        if isinstance(self.type_preference, dict):
            preference_type = self.type_preference.get("type")
            kwargs = self.type_preference.get("kwargs")
        else:
            preference_type = self.type_preference
            kwargs = {}
        if self.type_preference == None:
            pass
        if preference_type == "balanced":

            # Create all games together optimizing global happiness score
            self.games = self.create_all_balanced_games(
                self.people_playing,
                level_gap_tol=self.level_gap_tol,
                spectrum=self.spectrum,
                lambda_weight=self.lambda_weight,
                seed=seed,
                **kwargs,
            )

            if self.games == []:
                print("could not find a game, because the tolerance is too low")

        if preference_type == "level":
            self.create_games_by_level(seed=seed, **kwargs)
        self.teams = set()
        for game in self.games:
            self.teams = self.teams.union(game.teams)

        # Update history for playing players
        for player in self.people_playing:
            player.spec_chosen_history.append(player.last_spec_chosen)
            player.happiness_gained_history.append(player.last_happiness_gained)
            for team in self.teams:
                if player in team.players_set:
                    player.teammate_history.append(frozenset(team.players_set))

            player.other_players_in_game_history.append(
                [p for p in game.participants if p != player]
            )

        # Update history for non-playing players (already done in __init__ method)
        # No need to duplicate the history update here

    def create_balanced_game(
        self,
        people_left_to_play,
        level_gap_tol,
        spectrum,
        lambda_weight=2,
        seed=None,
        **kwargs,
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

                game = GameOfFour(
                    *teams,
                    type_preference=self.type_preference,
                    gender_preference=self.gender_preference,
                    weight_same_teammate=self.weight_same_teammate,
                )

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
                        seed=seed,
                    )

                    # --- Calculate happiness score ---
                    all_happiness = [
                        player.happiness for team in teams for player in team.players
                    ]
                    happiness_mean = np.mean(all_happiness)
                    happiness_std = np.std(all_happiness)
                    happiness_score = happiness_mean - lambda_weight * happiness_std

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
                seed=seed,
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
            game = GameOfFour(
                *teams,
                type_preference=self.type_preference,
                gender_preference=self.gender_preference,
                weight_same_teammate=self.weight_same_teammate,
            )
            return game

    def create_all_balanced_games(
        self,
        people_playing,
        level_gap_tol,
        spectrum,
        lambda_weight=2,
        seed=None,
        **kwargs,
    ):
        """
        Create all games for a round by considering all possible combinations
        and choosing the one with the best overall happiness score.
        """
        if (
            len(people_playing)
            < self.amount_of_games * self.teams_per_game * self.players_per_team
        ):
            return []

        # Generate all possible ways to divide players into games
        all_game_combinations = self.generate_all_game_combinations(people_playing)

        if not all_game_combinations:
            return []

        best_games = None
        best_overall_score = float("-inf")

        # Sample a subset if there are too many combinations (performance optimization)
        max_combinations_to_test = min(len(all_game_combinations), self.num_iter * 10)
        if len(all_game_combinations) > max_combinations_to_test:
            random.shuffle(all_game_combinations)
            all_game_combinations = all_game_combinations[:max_combinations_to_test]

        for game_combination in all_game_combinations:
            # Check if all games meet level gap tolerance
            if not all(
                game.level_difference <= level_gap_tol for game in game_combination
            ):
                continue

            # Save current happiness state
            happiness_backup = {
                player.name: player.happiness for player in people_playing
            }

            # Simulate happiness updates for all games
            all_players_in_games = []
            for game in game_combination:
                game.update_players_happiness(
                    self.session_median_level,
                    level_gap_tol,
                    spectrum,
                    seed=seed,
                )
                all_players_in_games.extend(game.participants)

            # Calculate overall happiness score for this combination
            all_happiness = [player.happiness for player in all_players_in_games]
            happiness_mean = np.mean(all_happiness)
            happiness_std = np.std(all_happiness) if len(all_happiness) > 1 else 0
            overall_score = happiness_mean - lambda_weight * happiness_std

            # Restore happiness state
            for player in people_playing:
                player.happiness = happiness_backup[player.name]

            # Track best combination
            if overall_score > best_overall_score:
                best_overall_score = overall_score
                best_games = game_combination

        # Apply the happiness updates for the best combination
        if best_games:
            for game in best_games:
                game.update_players_happiness(
                    self.session_median_level,
                    level_gap_tol,
                    spectrum,
                    seed=seed,
                )

            return list(best_games)

        return []

    def generate_all_game_combinations(self, people_playing):
        """
        Generate all possible ways to divide players into games.
        Returns a list of game combinations, where each combination is a list of GameOfFour objects.
        Uses sampling and early stopping to avoid combinatorial explosion.
        """
        from itertools import combinations

        players = list(people_playing)
        n_players = len(players)
        players_per_game = self.teams_per_game * self.players_per_team

        if n_players != self.amount_of_games * players_per_game:
            return []

        # Strict limit on total combinations to avoid infinite loops
        max_combinations = min(1000, self.num_iter * 50)
        all_combinations = []

        def generate_games_recursive(remaining_players, current_games, depth=0):
            # Early stopping if we have enough combinations
            if len(all_combinations) >= max_combinations:
                return

            if len(remaining_players) == 0:
                # All players assigned, add this combination
                all_combinations.append(current_games.copy())
                return

            if len(remaining_players) < players_per_game:
                # Not enough players for another game
                return

            # Limit the number of player combinations we consider
            game_player_combos = list(combinations(remaining_players, players_per_game))

            # Sample combinations if there are too many
            if depth == 0:  # First level
                max_combos = min(len(game_player_combos), 20)
            else:  # Deeper levels
                max_combos = min(len(game_player_combos), 10)

            if len(game_player_combos) > max_combos:
                random.shuffle(game_player_combos)
                game_player_combos = game_player_combos[:max_combos]

            for game_players in game_player_combos:
                if len(all_combinations) >= max_combinations:
                    break

                game_players_list = list(game_players)

                # Limit team arrangements - only try 3 different arrangements
                team_combos = list(
                    combinations(range(players_per_game), self.players_per_team)
                )
                max_team_combos = min(len(team_combos), 3)
                team_combos = team_combos[:max_team_combos]

                for team1_indices in team_combos:
                    if len(all_combinations) >= max_combinations:
                        break

                    team2_indices = [
                        i for i in range(players_per_game) if i not in team1_indices
                    ]

                    team1_players = [game_players_list[i] for i in team1_indices]
                    team2_players = [game_players_list[i] for i in team2_indices]

                    team1 = TeamOfTwo(*team1_players)
                    team2 = TeamOfTwo(*team2_players)

                    game = GameOfFour(
                        team1,
                        team2,
                        type_preference=self.type_preference,
                        gender_preference=self.gender_preference,
                        weight_same_teammate=self.weight_same_teammate,
                    )

                    # Recursively generate the rest
                    new_remaining = [
                        p for p in remaining_players if p not in game_players
                    ]
                    current_games.append(game)
                    generate_games_recursive(new_remaining, current_games, depth + 1)
                    current_games.pop()

        generate_games_recursive(players, [])

        return all_combinations

    def create_games_by_level(self, alternate=False, mixed=False, seed=None, **kwargs):

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
            prev_team_sets = [team.players_set for team in self.previous_teams]
            for team1, team2, _ in possible_team_pairs_with_level_diff:
                if (
                    team1.players_set not in prev_team_sets
                    and team2.players_set not in prev_team_sets
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
                game = GameOfFour(
                    team1,
                    team2,
                    type_preference=self.type_preference,
                    gender_preference=self.gender_preference,
                )
                self.games.append(game)
                game.update_players_happiness(
                    self.session_median_level,
                    self.level_gap_tol,
                    self.spectrum,
                    seed=seed,
                )


# %%
if __name__ == "__main__":
    list_of_players = [Player(df_minimal_example.iloc[i]) for i in range(19)]
    round_of_games = GamesRound(
        list_of_players,
        type_preference={"type": "level", "kwargs": {"randomize": False}},
        gender_preference=None,
        num_iter=40,
        level_gap_tol=2,
        seed=0,
    )
    for attr in ["amount_of_games", "type_preference"]:
        print(attr + " : ", getattr(round_of_games, attr))
    print("not playing:", [player.name for player in round_of_games.not_playing])
    i = 1
    for game in round_of_games.games:
        print("_______", "game", i, "_______")
        print([team.players_name for team in game.teams])
        for attr in ["type_preference", "level_difference"]:
            print(attr + " : ", getattr(game, attr))
        i += 1

# %%
if __name__ == "__main__":
    list_of_players = [Player(main_df.loc[name]) for name in main_df.iloc[10:21].index]
    round_of_games = GamesRound(
        list_of_players,
        type_preference="balanced",
        gender_preference=None,
        level_gap_tol=2,
        num_iter=40,
        seed=1,
    )

    for attr in ["amount_of_games", "type_preference"]:
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
        type_preferences=[None],
        gender_preferences=[None],
        level_gap_tol=0.5,
        num_iter=40,
        spectrum=False,
        lambda_weight=2,
        weight_same_teammate=4,
        seed=None,
    ):
        self.amount_of_rounds = amount_of_rounds
        self.games_per_round_each_round = games_per_round_each_round
        self.players_per_team_each_round = players_per_team_each_round
        self.type_preferences = type_preferences
        self.gender_preferences = gender_preferences
        self.level_gap_tol = level_gap_tol
        self.num_iter = num_iter

        self.players = list_of_players
        self.players_name = [player.name for player in list_of_players]
        self.rounds = []
        self.spectrum = spectrum
        self.lambda_weight = lambda_weight
        self.weight_same_teammate = weight_same_teammate

        # reformatting type_preferences to the amount of type_preferences wanted
        if type_preferences is None:
            type_preferences = [None] * amount_of_rounds
        elif isinstance(type_preferences, str):
            type_preferences = [type_preferences] * amount_of_rounds
        elif len(type_preferences) < amount_of_rounds:
            type_preferences = type_preferences + [None] * (
                amount_of_rounds - len(type_preferences)
            )
        elif len(type_preferences) > amount_of_rounds:
            return "too many type_preferences"
        self.type_preferences = type_preferences

        # reformatting gender_preferences to the amount of gender_preferences wanted
        if gender_preferences is None:
            gender_preferences = [None] * amount_of_rounds
        elif isinstance(gender_preferences, str):
            gender_preferences = [gender_preferences] * amount_of_rounds
        elif len(gender_preferences) < amount_of_rounds:
            gender_preferences = gender_preferences + [None] * (
                amount_of_rounds - len(gender_preferences)
            )
        elif len(gender_preferences) > amount_of_rounds:
            return "too many gender_preferences"
        self.gender_preferences = gender_preferences

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
                    type_preference=self.type_preferences[i],
                    gender_preference=self.gender_preferences[i],
                    level_gap_tol=self.level_gap_tol,
                    num_iter=self.num_iter,
                    lambda_weight=self.lambda_weight,
                    weight_same_teammate=self.weight_same_teammate,
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

    def count_all_opponent_pairs(self, order_num_list=None):
        # Find and display players who played against each other at least twice and record the rounds
        opponent_pairs = {}
        opponent_pair_rounds = {}
        # Use the possibly reordered rounds for analysis
        round_copy = self.rounds.copy()
        if order_num_list is not None:
            round_copy = [round_copy[i - 1] for i in order_num_list]
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
        return opponent_pairs, opponent_pair_rounds

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

    def add_opponent_team_repetition_to_output(
        self, output, opponent_pairs, opponent_pair_rounds, minimum=3
    ):
        teams_to_repetitions = {}
        output.append(
            f"\n#######PLAYERS WHO PLAYED AGAINST EACH OTHER AT LEAST {minimum} TIMES #######"
        )
        for pair, count in opponent_pairs.items():
            if count >= minimum:
                rounds = ", ".join(map(str, opponent_pair_rounds[pair]))
                output.append(
                    f"{', '.join(pair)} played against each other {count} times in rounds: {rounds}"
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

    def print_all_results(
        self,
        print_levels=True,
        order_num_list=None,
        minimum_team_repeats=2,
        minimum_opponent_repeats=3,
    ):
        import pyperclip

        # ANSI color codes
        class Colors:
            HEADER = "\033[95m"
            OKBLUE = "\033[94m"
            OKCYAN = "\033[96m"
            OKGREEN = "\033[92m"
            WARNING = "\033[93m"
            FAIL = "\033[91m"
            ENDC = "\033[0m"
            BOLD = "\033[1m"
            UNDERLINE = "\033[4m"
            MAGENTA = "\033[35m"
            YELLOW = "\033[33m"
            RED = "\033[31m"
            GREEN = "\033[32m"
            BLUE = "\033[34m"
            CYAN = "\033[36m"

        # Spectrum colors mapping
        spec_colors = {
            "Maso": Colors.RED,
            "Équilibré": Colors.GREEN,
            "Challenger": Colors.YELLOW,
            "Chill": Colors.CYAN,
            "Sadique": Colors.MAGENTA,
            "Alchimiste": Colors.BLUE,
            None: Colors.ENDC,
        }

        # Collect all printed information
        output = []
        colored_output = []  # For terminal display with colors

        header_text = "all players: " + str(self.players_name)
        output.append(header_text)
        colored_output.append(Colors.HEADER + Colors.BOLD + header_text + Colors.ENDC)

        i = 1
        round_copy = self.rounds.copy()
        if order_num_list is not None:
            round_copy = [round_copy[i - 1] for i in order_num_list]
        for round in round_copy:
            output.append("\n")
            colored_output.append("\n")

            round_header = f"{i} " * 8 + "ROUND " + f"{i} " * 8
            output.append(round_header)
            colored_output.append(
                Colors.OKBLUE + Colors.BOLD + round_header + Colors.ENDC
            )

            pref_text = "type preference : " + str(getattr(round, "type_preference"))
            pref_text += ",\ngender preference : " + str(
                getattr(round, "gender_preference")
            )
            output.append(pref_text)
            colored_output.append(Colors.OKCYAN + pref_text + Colors.ENDC)

            not_playing_text = "not playing: " + str(
                [player.name for player in round.not_playing]
            )
            output.append(not_playing_text)
            colored_output.append(Colors.WARNING + not_playing_text + Colors.ENDC)

            j = 1
            for game in round.games:
                game_header = f"------- game {j} -------"
                output.append(game_header)
                colored_output.append(Colors.OKGREEN + game_header + Colors.ENDC)

                teams_text = str([team.players_name for team in game.teams])
                output.append(teams_text)
                colored_output.append(Colors.BOLD + teams_text + Colors.ENDC)

                # Always print happiness gained for all players
                for player in game.participants:
                    # Get the correct round index for spec_chosen_history
                    round_idx = (
                        order_num_list[i - 1] - 1
                        if order_num_list is not None
                        else i - 1
                    )
                    spec_chosen = player.spec_chosen_history[round_idx]
                    happiness_gained = player.happiness_gained_history[round_idx]

                    if print_levels:
                        # Color code based on happiness gained
                        color = ""
                        if happiness_gained is None:
                            color = Colors.ENDC
                        else:
                            if happiness_gained >= 3:
                                color = (
                                    spec_colors.get(spec_chosen, Colors.ENDC)
                                    + Colors.GREEN
                                )
                            elif 0 <= happiness_gained < 3:
                                color = (
                                    spec_colors.get(spec_chosen, Colors.ENDC)
                                    + Colors.YELLOW
                                )
                            else:  # negative
                                color = (
                                    spec_colors.get(spec_chosen, Colors.ENDC)
                                    + Colors.RED
                                )

                        player_text = f"name : {player.name}, level : {player.level}, happiness gained ({spec_chosen}): {color}{happiness_gained}{Colors.ENDC}"
                    else:
                        # Color code based on happiness gained
                        color = ""
                        if happiness_gained is None:
                            color = Colors.ENDC
                        else:
                            if happiness_gained >= 3:
                                color = (
                                    spec_colors.get(spec_chosen, Colors.ENDC)
                                    + Colors.GREEN
                                )
                            elif 0 <= happiness_gained < 3:
                                color = (
                                    spec_colors.get(spec_chosen, Colors.ENDC)
                                    + Colors.YELLOW
                                )
                            else:  # negative
                                color = (
                                    spec_colors.get(spec_chosen, Colors.ENDC)
                                    + Colors.RED
                                )
                        player_text = f"name : {player.name}, happiness gained ({spec_chosen}): {color}{happiness_gained}{Colors.ENDC}"

                    output.append(player_text)
                    colored_output.append(player_text)

                if game.type_preference == "balanced":
                    level_diff_text = "level_difference : " + str(
                        np.round(getattr(game, "level_difference"), 2)
                    )
                    output.append(level_diff_text)
                    colored_output.append(level_diff_text)

                if game.type_preference == "level":
                    level_diff_text = (
                        f"level difference : {np.round(game.level_difference, 2)}"
                    )
                    output.append(level_diff_text)
                    colored_output.append(level_diff_text)
                j += 1
            round_end_text = f"{i} " * 8 + "ROUND END " + f"{i} " * 8
            output.append(round_end_text)
            colored_output.append(
                Colors.OKBLUE + Colors.BOLD + round_end_text + Colors.ENDC
            )
            output.append("\n\n\n")
            colored_output.append("\n\n\n")
            i += 1

        games_played_header = "\n#####AMOUNT OF GAMES PLAYED#####"
        output.append(games_played_header)
        colored_output.append(
            Colors.HEADER + Colors.BOLD + games_played_header + Colors.ENDC
        )

        for player in self.players:
            player_stats_text = f"{player.name} played {player.games_played} games, happiness: {np.round(player.happiness, 2)}"
            output.append(player_stats_text)

            # Color based on happiness level
            happiness = player.happiness
            if happiness >= np.mean([p.happiness for p in self.players]) + np.std(
                [p.happiness for p in self.players]
            ):
                color = Colors.OKGREEN  # High happiness
            elif happiness <= np.mean([p.happiness for p in self.players]) - np.std(
                [p.happiness for p in self.players]
            ):
                color = Colors.FAIL  # Low happiness
            else:
                color = Colors.ENDC  # Normal happiness

            colored_player_stats = f"{Colors.BOLD}{player.name}{Colors.ENDC} played {player.games_played} games, happiness: {color}{np.round(player.happiness, 2)}{Colors.ENDC}"
            colored_output.append(colored_player_stats)
        #####ADDD THE COUNT OF PLAYERS THAT PLAYED TOGETHER AT LEAST TWICE AND SO ON, REPLACE FROM PRINT_ALL_RESULTS#####
        player_pairs, pair_rounds = self.count_all_pairs(order_num_list)

        output, _ = self.add_team_repetition_to_output(
            output, player_pairs, pair_rounds, minimum=minimum_team_repeats
        )
        # Also add to colored output
        temp_output = []
        temp_output, _ = self.add_team_repetition_to_output(
            temp_output, player_pairs, pair_rounds, minimum=minimum_team_repeats
        )
        for line in temp_output:
            if line.startswith("\n#######PLAYERS WHO PLAYED TOGETHER"):
                colored_output.append(Colors.WARNING + Colors.BOLD + line + Colors.ENDC)
            else:
                colored_output.append(Colors.WARNING + line + Colors.ENDC)

        # Find and display players who played against each other at least twice and record the rounds
        opponent_pairs, opponent_pair_rounds = self.count_all_opponent_pairs(
            order_num_list
        )

        output, _ = self.add_opponent_team_repetition_to_output(
            output,
            opponent_pairs,
            opponent_pair_rounds,
            minimum=minimum_opponent_repeats,
        )
        # Also add to colored output
        temp_output = []
        temp_output, _ = self.add_opponent_team_repetition_to_output(
            temp_output,
            opponent_pairs,
            opponent_pair_rounds,
            minimum=minimum_opponent_repeats,
        )
        for line in temp_output:
            if line.startswith("\n#######PLAYERS WHO PLAYED AGAINST"):
                colored_output.append(Colors.FAIL + Colors.BOLD + line + Colors.ENDC)
            else:
                colored_output.append(Colors.FAIL + line + Colors.ENDC)

        # Add happiness analytics section
        analytics_header = "\n#####HAPPINESS ANALYTICS#####"
        output.append(analytics_header)
        colored_output.append(
            Colors.HEADER + Colors.BOLD + analytics_header + Colors.ENDC
        )

        # Calculate happiness statistics
        happiness_values = [player.happiness for player in self.players]

        avg_text = f"Average happiness: {np.round(np.mean(happiness_values), 2)}"
        output.append(avg_text)
        colored_output.append(Colors.OKCYAN + avg_text + Colors.ENDC)

        std_text = (
            f"Happiness standard deviation: {np.round(np.std(happiness_values), 2)}"
        )
        output.append(std_text)
        colored_output.append(Colors.OKCYAN + std_text + Colors.ENDC)

        min_text = f"Min happiness: {np.round(min(happiness_values), 2)}"
        output.append(min_text)
        colored_output.append(Colors.FAIL + min_text + Colors.ENDC)

        max_text = f"Max happiness: {np.round(max(happiness_values), 2)}"
        output.append(max_text)
        colored_output.append(Colors.OKGREEN + max_text + Colors.ENDC)

        # Identify happiest and least happy players
        happiest_players = sorted(
            self.players, key=lambda p: p.happiness, reverse=True
        )[:3]
        least_happy_players = sorted(self.players, key=lambda p: p.happiness)[:3]

        happiest_header = "\nHappiest players:"
        output.append(happiest_header)
        colored_output.append(
            Colors.OKGREEN + Colors.BOLD + happiest_header + Colors.ENDC
        )

        for player in happiest_players:
            player_happy_text = f"{player.name}: {np.round(player.happiness, 2)}"
            output.append(player_happy_text)
            colored_output.append(Colors.OKGREEN + player_happy_text + Colors.ENDC)

        least_happy_header = "\nLeast happy players:"
        output.append(least_happy_header)
        colored_output.append(
            Colors.FAIL + Colors.BOLD + least_happy_header + Colors.ENDC
        )

        for player in least_happy_players:
            player_sad_text = f"{player.name}: {np.round(player.happiness, 2)}"
            output.append(player_sad_text)
            colored_output.append(Colors.FAIL + player_sad_text + Colors.ENDC)

        pyperclip.copy("\n".join(output))
        print("\n".join(colored_output))  # Print colored version to terminal
        print(
            Colors.OKGREEN
            + Colors.BOLD
            + "\n\n\n\nALL RESULTS HAVE BEEN COPIED TO THE CLIPBOARD."
            + Colors.ENDC
        )


# %%
################################################################################
################################################################################
################################################################################
###############################                 ################################
###############################  #####  #   #  ################################
############################### #        ###   ################################
###############################  ####     #    ################################
###############################      #    #    ################################
############################### #####     #    ################################
###############################                 ################################
################################################################################
################################################################################
################################################################################


def plot_session_charts(session_of_rounds, save_path=None):
    """
    Create essential visualizations for a roundnet session.

    Parameters:
    - session_of_rounds: SessionOfRounds object
    - save_path: Optional path to save plots (if None, plots are just displayed)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from collections import defaultdict

    # Set style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Prepare data
    happiness_values = [player.happiness for player in session_of_rounds.players]
    player_names = [player.name for player in session_of_rounds.players]
    levels = [player.level for player in session_of_rounds.players]
    games_played = [player.games_played for player in session_of_rounds.players]

    # 1. Happiness Distribution (top-left)
    ax1 = axes[0, 0]
    ax1.hist(
        happiness_values,
        bins=max(1, len(set(happiness_values))),
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    ax1.set_title("Happiness Distribution", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Happiness Score")
    ax1.set_ylabel("Number of Players")
    ax1.grid(axis="y", alpha=0.3)

    # Add statistics text
    stats_text = f"Mean: {np.mean(happiness_values):.2f}\n"
    stats_text += f"Std: {np.std(happiness_values):.2f}\n"
    stats_text += f"Min: {min(happiness_values):.2f}\n"
    stats_text += f"Max: {max(happiness_values):.2f}"
    ax1.text(
        0.02,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # 2. Level vs Happiness Scatter (top-right)
    ax2 = axes[0, 1]
    scatter = ax2.scatter(
        levels,
        happiness_values,
        s=[p.games_played * 50 for p in session_of_rounds.players],
        alpha=0.7,
        c=happiness_values,
        cmap="RdYlGn",
    )
    ax2.set_title(
        "Level vs Happiness\n(Size = Games Played)", fontsize=14, fontweight="bold"
    )

    ax2.set_xlabel("Player Level")
    ax2.set_ylabel("Happiness Score")
    ax2.grid(alpha=0.3)

    # Add player name annotations
    for i, (level, happiness, name) in enumerate(
        zip(levels, happiness_values, player_names)
    ):
        ax2.annotate(
            name,
            (level, happiness),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
            alpha=0.9,
            ha="left",
            va="bottom",
        )

    # 3. Happiness by Gender (bottom-left)
    ax3 = axes[1, 0]

    # Group happiness by gender
    genders = [
        getattr(player, "gender", "Unknown") for player in session_of_rounds.players
    ]
    happiness_by_gender = defaultdict(list)
    for player in session_of_rounds.players:
        gender = getattr(player, "gender", "Unknown")
        happiness_by_gender[gender].append(player.happiness)

    # Prepare data for boxplot
    box_data = []
    box_labels = []
    box_colors = {"Homme": "lightblue", "Femme": "lightpink", "Unknown": "lightgray"}
    colors_list = []

    for gender in sorted(happiness_by_gender.keys()):
        box_data.append(happiness_by_gender[gender])
        box_labels.append(f"{gender}\n(n={len(happiness_by_gender[gender])})")
        colors_list.append(box_colors.get(gender, "lightgray"))

    # Create boxplot
    bp = ax3.boxplot(
        box_data, labels=box_labels, patch_artist=True, showmeans=True, meanline=True
    )

    # Color the boxes
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax3.set_title("Happiness Distribution by Gender", fontsize=14, fontweight="bold")
    ax3.set_ylabel("Happiness Score")
    ax3.grid(axis="y", alpha=0.3)

    # Add mean values as text
    for i, (gender, happiness_list) in enumerate(sorted(happiness_by_gender.items())):
        mean_val = np.mean(happiness_list)
        ax3.text(
            i + 1,
            mean_val,
            f"{mean_val:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    # 4. Session Summary Stats (bottom-right)
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Calculate summary statistics
    total_games = sum(len(round_obj.games) for round_obj in session_of_rounds.rounds)
    avg_level = np.mean(levels)
    happiness_inequality = np.std(happiness_values)

    summary_text = f"""SESSION SUMMARY STATISTICS
    
Total Rounds: {len(session_of_rounds.rounds)}
Total Games: {total_games}
Total Players: {len(session_of_rounds.players)}

Average Player Level: {avg_level:.2f}
Level Range: {min(levels):.1f} - {max(levels):.1f}

Happiness Mean: {np.mean(happiness_values):.2f}
Happiness Std Dev: {happiness_inequality:.2f}
Happiness Range: {min(happiness_values):.1f} - {max(happiness_values):.1f}

Most Happy: {max(session_of_rounds.players, key=lambda p: p.happiness).name}
Least Happy: {min(session_of_rounds.players, key=lambda p: p.happiness).name}

Games Played Range: {min(games_played)} - {max(games_played)}
"""

    ax4.text(
        0.1,
        0.9,
        summary_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=1", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Charts saved to {save_path}")
    else:
        plt.show()

    return fig


def plot_team_analysis(session_of_rounds, save_path=None):
    """
    Create visualizations focused on team compositions and partnerships.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import networkx as nx
    from collections import defaultdict

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Partnership Network Graph
    ax1 = axes[0, 0]
    G = nx.Graph()

    # Add all players as nodes
    for player in session_of_rounds.players:
        G.add_node(player.name, level=player.level, happiness=player.happiness)

    # Add edges for partnerships
    partnership_counts = defaultdict(int)
    for round_obj in session_of_rounds.rounds:
        for game in round_obj.games:
            for team in game.teams:
                players_list = list(team.players)
                if len(players_list) == 2:
                    p1, p2 = players_list[0].name, players_list[1].name
                    partnership_counts[(p1, p2)] += 1
                    G.add_edge(p1, p2, weight=partnership_counts[(p1, p2)])

    # Create layout
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Draw nodes with size based on happiness
    node_sizes = [max(50, G.nodes[node]["happiness"] * 20) for node in G.nodes()]
    node_colors = [G.nodes[node]["level"] for node in G.nodes()]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap="viridis",
        alpha=0.8,
        ax=ax1,
    )

    # Draw edges with thickness based on partnership frequency
    edges = G.edges()
    weights = [G[u][v]["weight"] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w * 2 for w in weights], alpha=0.6, ax=ax1)

    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax1)

    ax1.set_title(
        "Partnership Network\n(Node size = Happiness, Color = Level, Edge thickness = Partnerships)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.axis("off")

    # 2. Team Level Balance Analysis
    ax2 = axes[0, 1]
    level_differences = []
    game_types = []

    for round_obj in session_of_rounds.rounds:
        for game in round_obj.games:
            level_differences.append(game.level_difference)
            game_types.append(str(game.type_preference))

    # Group by game type
    unique_types = list(set(game_types))
    for i, game_type in enumerate(unique_types):
        type_diffs = [
            diff for diff, typ in zip(level_differences, game_types) if typ == game_type
        ]
        ax2.boxplot(
            type_diffs,
            positions=[i],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=plt.cm.Set3(i), alpha=0.7),
        )

    ax2.set_title("Level Differences by Game Type", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Level Difference")
    ax2.set_xticks(range(len(unique_types)))
    ax2.set_xticklabels(unique_types, rotation=45)
    ax2.grid(axis="y", alpha=0.3)

    # 3. Round-by-Round Happiness Evolution
    ax3 = axes[1, 0]

    # This requires tracking happiness per round (simplified visualization)
    for player in session_of_rounds.players:
        # Simulate happiness progression (this could be enhanced with actual tracking)
        happiness_progression = [0]
        current_happiness = 0
        for round_idx, round_obj in enumerate(session_of_rounds.rounds):
            # Check if player participated in this round
            participated = any(player in game.participants for game in round_obj.games)
            if participated:
                # Estimate happiness gain (this could be improved with actual tracking)
                estimated_gain = player.happiness / max(1, player.games_played)
                current_happiness += estimated_gain
            happiness_progression.append(current_happiness)

        ax3.plot(
            range(len(happiness_progression)),
            happiness_progression,
            marker="o",
            label=player.name,
            alpha=0.7,
        )

    ax3.set_title("Happiness Evolution by Round", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Round Number")
    ax3.set_ylabel("Cumulative Happiness")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax3.grid(alpha=0.3)

    # 4. Gender and Level Distribution
    ax4 = axes[1, 1]

    # Get gender and level data
    genders = [
        getattr(player, "gender", "Unknown") for player in session_of_rounds.players
    ]
    levels_by_gender = defaultdict(list)
    for player in session_of_rounds.players:
        gender = getattr(player, "gender", "Unknown")
        levels_by_gender[gender].append(player.level)

    # Create violin plot
    positions = []
    data_to_plot = []
    labels = []
    for i, (gender, levels_list) in enumerate(levels_by_gender.items()):
        positions.append(i)
        data_to_plot.append(levels_list)
        labels.append(f"{gender}\n(n={len(levels_list)})")

    parts = ax4.violinplot(
        data_to_plot, positions=positions, showmeans=True, showmedians=True
    )

    # Color the violin plots
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(plt.cm.Set2(i))
        pc.set_alpha(0.7)

    ax4.set_title("Level Distribution by Gender", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Player Level")
    ax4.set_xticks(positions)
    ax4.set_xticklabels(labels)
    ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Team analysis charts saved to {save_path}")
    else:
        plt.show()

    return fig


def plot_spectrum_analysis(session_of_rounds, save_path=None):
    """
    Create visualizations for spectrum (personality types) analysis if spectrum data is available.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from collections import defaultdict

    # Check if spectrum data is available
    spectrum_attrs = [
        "maso",
        "equilibre",
        "challenger",
        "chill",
        "sadique",
        "alchimiste",
    ]
    has_spectrum = any(
        hasattr(player, attr) and getattr(player, attr, 0) > 0
        for player in session_of_rounds.players
        for attr in spectrum_attrs
    )

    if not has_spectrum:
        print("No spectrum data available for analysis.")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Spectrum Profile Radar Chart
    ax1 = axes[0, 0]

    # Create radar chart for average spectrum values
    spectrum_names = [
        "Maso",
        "Équilibré",
        "Challenger",
        "Chill",
        "Sadique",
        "Alchimiste",
    ]
    spectrum_values = []

    for attr in spectrum_attrs:
        values = [getattr(player, attr, 0) for player in session_of_rounds.players]
        spectrum_values.append(np.mean(values))

    # Number of variables
    N = len(spectrum_names)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    # Add the first value at the end to close the polygon
    spectrum_values += spectrum_values[:1]

    # Plot
    ax1 = plt.subplot(2, 2, 1, projection="polar")
    ax1.plot(angles, spectrum_values, "o-", linewidth=2, color="blue", alpha=0.7)
    ax1.fill(angles, spectrum_values, alpha=0.25, color="blue")
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(spectrum_names)
    ax1.set_title("Average Spectrum Profile", fontsize=12, fontweight="bold", pad=20)

    # 2. Spectrum vs Happiness
    ax2 = axes[0, 1]

    # Calculate dominant spectrum for each player
    player_dominant_spectrums = []
    player_happiness = []

    for player in session_of_rounds.players:
        spectrum_scores = {
            name: getattr(player, attr, 0)
            for name, attr in zip(spectrum_names, spectrum_attrs)
        }
        dominant = (
            max(spectrum_scores, key=spectrum_scores.get)
            if max(spectrum_scores.values()) > 0
            else "None"
        )
        player_dominant_spectrums.append(dominant)
        player_happiness.append(player.happiness)

    # Group happiness by dominant spectrum
    spectrum_happiness = defaultdict(list)
    for spectrum, happiness in zip(player_dominant_spectrums, player_happiness):
        spectrum_happiness[spectrum].append(happiness)

    # Create box plot
    box_data = []
    box_labels = []
    for spectrum, happiness_list in spectrum_happiness.items():
        if happiness_list:  # Only include if there's data
            box_data.append(happiness_list)
            box_labels.append(f"{spectrum}\n(n={len(happiness_list)})")

    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)

    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(bp["boxes"])))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_title("Happiness by Dominant Spectrum", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Happiness Score")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.3)

    # 3. Spectrum Chosen History (if available)
    ax3 = axes[1, 0]

    # Analyze spec_chosen_history
    all_specs_chosen = []
    for player in session_of_rounds.players:
        if hasattr(player, "spec_chosen_history"):
            all_specs_chosen.extend(
                [spec for spec in player.spec_chosen_history if spec is not None]
            )

    if all_specs_chosen:
        spec_counts = {}
        for spec in all_specs_chosen:
            spec_counts[spec] = spec_counts.get(spec, 0) + 1

        # Create pie chart
        wedges, texts, autotexts = ax3.pie(
            spec_counts.values(),
            labels=spec_counts.keys(),
            autopct="%1.1f%%",
            startangle=90,
        )
        ax3.set_title(
            "Spectrum Types Chosen During Games", fontsize=12, fontweight="bold"
        )
    else:
        ax3.text(
            0.5,
            0.5,
            "No spectrum choices recorded",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=14,
        )
        ax3.set_title(
            "Spectrum Types Chosen During Games", fontsize=12, fontweight="bold"
        )

    # 4. Player Spectrum Heatmap
    ax4 = axes[1, 1]

    # Create matrix of player spectrum values
    spectrum_matrix = []
    player_names = [player.name for player in session_of_rounds.players]

    for player in session_of_rounds.players:
        player_spectrum = [getattr(player, attr, 0) for attr in spectrum_attrs]
        spectrum_matrix.append(player_spectrum)

    spectrum_matrix = np.array(spectrum_matrix)

    # Create heatmap
    im = ax4.imshow(spectrum_matrix, cmap="YlOrRd", aspect="auto")
    ax4.set_title("Player Spectrum Profiles", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Spectrum Types")
    ax4.set_ylabel("Players")
    ax4.set_xticks(range(len(spectrum_names)))
    ax4.set_xticklabels(spectrum_names, rotation=45)
    ax4.set_yticks(range(len(player_names)))
    ax4.set_yticklabels(player_names, fontsize=8)

    # Add colorbar
    plt.colorbar(im, ax=ax4, shrink=0.8)

    # Add text annotations
    for i in range(len(player_names)):
        for j in range(len(spectrum_names)):
            text = ax4.text(
                j,
                i,
                f"{spectrum_matrix[i, j]:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Spectrum analysis charts saved to {save_path}")
    else:
        plt.show()

    return fig


# Example usage function
def create_all_session_charts(session_of_rounds, base_filename=None):
    """
    Create all available charts for a session.

    Parameters:
    - session_of_rounds: SessionOfRounds object
    - base_filename: Base filename for saving (without extension). If None, charts are displayed.
    """
    print("Creating session overview charts...")
    fig1 = plot_session_charts(
        session_of_rounds, f"{base_filename}_overview.png" if base_filename else None
    )

    print("Creating team analysis charts...")
    fig2 = plot_team_analysis(
        session_of_rounds, f"{base_filename}_teams.png" if base_filename else None
    )

    print("Creating spectrum analysis charts...")
    fig3 = plot_spectrum_analysis(
        session_of_rounds, f"{base_filename}_spectrum.png" if base_filename else None
    )

    print("All charts created successfully!")
    return fig1, fig2, fig3


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
        type_preferences=["balanced"] * 3 + ["level"] * 3,
        level_gap_tol=1.5,
        num_iter=50,
        seed=3,
    )
    # %%
    # session_of_rounds.print_all_results()

    # Example of how to use the chart functions:
    # create_all_session_charts(session_of_rounds, "my_session")
    # or just display without saving:
    # create_all_session_charts(session_of_rounds)
