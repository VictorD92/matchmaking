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
        main_df[col] = main_df[col].astype(true_dtypes[col])

# setting default category as level
main_df["Category"] = main_df["Level"]
# setting default Happiness to 0
main_df["Happiness"] = 0
# setting default games played to 0
main_df["Games played"] = 0

# setting default Noisy level to 0
main_df["Noisy level"] = 0

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
        - Hapiness: float. The happiness of the player (normally at 0)
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
        self.level_difference = np.round(
            abs(self.team_A.mean_level - self.team_B.mean_level), 2
        )


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
        num_iter=1000,
        level_gap_tol=0.5,
        seed=None,
    ):
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
        self.teams_per_game = teams_per_game
        self.players_per_team = players_per_team
        self.num_iter = num_iter
        self.level_gap_tol = level_gap_tol

        self.games = []
        self.create_games(seed=seed)

        self.not_playing = [
            person for person in list_of_players if person not in self.people_playing
        ]

    # Modify the create_set_of_all_possible_teams method to include a seed parameter
    def create_set_of_all_possible_teams(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # function that creates all possible teams of <players_per_team> players from a set of players
        return {
            TeamOfTwo(*team)
            for team in combinations(self.people_playing, self.players_per_team)
        }

    # Modify the create_games method to include a seed parameter
    def create_games(self, seed=None):
        import random

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

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
            amount_of_games_played: set()
            for amount_of_games_played in all_amounts_of_games_played
        }

        for player in random.sample(
            list(self.people_present), len(self.people_present)
        ):
            dic_amount_of_games_played_list_of_players[player.games_played].add(player)

        list_descending_priority = [
            player
            for i in sorted(
                dic_amount_of_games_played_list_of_players.keys(), reverse=True
            )
            for player in dic_amount_of_games_played_list_of_players[i]
        ]
        self.people_playing = set(list_descending_priority[amount_non_playing:])

        for player in self.people_playing:
            player.games_played += 1
        ########################################################################

        self.set_of_all_possible_teams = self.create_set_of_all_possible_teams(
            seed=seed
        )
        ######preference == none################################################
        # if preference is none, we create random games, trying not to recreate the same games
        if isinstance(self.preference, dict):
            preference_type = self.preference.get("type")
        else:
            preference_type = self.preference
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
                        self.create_balanced_game(people_left_to_play, seed=seed)
                    )

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
        if preference_type == "level":
            # we find all the possible levels
            if isinstance(self.preference, dict):
                kwargs = self.preference.get("kwargs")
            else:
                kwargs = {}
            self.create_games_by_level(seed=seed, **kwargs)
            ########################################################################

    # Modify the create_balanced_game method to include a seed parameter
    def create_balanced_game(self, people_left_to_play, balanced=True, seed=None):
        if seed is not None:
            np.random.seed(seed)

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

                level_diff += 0.1

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

    # Modify the create_games_by_level method to include a seed parameter
    def create_games_by_level(self, seed=None, randomize=True):
        if seed is not None:
            np.random.seed(seed)

        # we randomize a bit the level of the players, to avoid having the same players in the same level
        # we first find the minimal and maximal levels of the players
        min_level = min([player.level for player in self.people_playing])
        max_level = max([player.level for player in self.people_playing])
        level_gap = max_level - min_level
        if randomize:
            # we add a random number with value uniformly distributed, up to 1/2 * level_gap if the preceeding noisy level was below the level of the player,
            # and up to -1/2* level_gap if the preceeding noisy level was above the level of the player

            for player in self.people_playing:
                bias = 1 if player.level > player.noisy_level else -1
                player.noisy_level = player.level + bias * np.random.uniform(
                    0, level_gap / 2
                )

        else:
            for player in self.people_playing:
                player.noisy_level = player.level
        # we first sort the players by level

        # finding all the possible levels, rounded up to 1/4 of the level gap
        levels = []
        for player in self.people_playing:
            rounded_noisy_level = player.noisy_level - player.noisy_level % (
                level_gap / 8
            )
            player.rounded_noisy_level = rounded_noisy_level
            if rounded_noisy_level not in levels:
                levels.append(rounded_noisy_level)
        levels = sorted(levels, reverse=True)

        dic_level_players = {level: [] for level in levels}
        for player in self.people_playing:
            dic_level_players[player.rounded_noisy_level].append(player)
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
                    ],
                    balanced=True,
                    seed=seed,
                )
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
    list_of_players = [Player(main_df.loc[name]) for name in main_df.iloc[10:22].index]
    round_of_games = GamesRound(
        list_of_players, preference="level", level_gap_tol=2, num_iter=40, seed=0
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
#       # #          #    #     # #       #   ##  #     # #     # #   # # #    #        # #
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
            rounds.append(
                GamesRound(
                    list_of_players=self.players,
                    amount_of_games=self.games_per_round_each_round[i],
                    players_per_team=self.players_per_team_each_round[i],
                    preference=self.preferences[i],
                    level_gap_tol=self.level_gap_tol,
                    num_iter=self.num_iter,
                    seed=seed,
                )
            )
        self.rounds = rounds

    def print_all_results(self):
        import pyperclip

        # Collect all printed information
        output = []
        output.append("all players: " + str(self.players_name))
        i = 1
        for round in self.rounds:
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
                if game.preference == "level":
                    for player in game.participants:
                        output.append(f"name : {player.name}, level : {player.level}")
                j += 1
            output.append(f"{i} " * 8 + "ROUND END " + f"{i} " * 8)
            output.append("\n\n\n")
            i += 1

        output.append("\n#####AMOUNT OF GAMES PLAYED#####")
        for player in self.players:
            output.append(f"{player.name} played {player.games_played} games")

        # Find and display players who played at least twice with each other and record the rounds
        player_pairs = {}
        pair_rounds = {}
        for round_index, round in enumerate(self.rounds, start=1):
            for game in round.games:
                for team in game.teams:
                    for player_a, player_b in combinations(team.players, 2):
                        pair = frozenset([player_a.name, player_b.name])
                        player_pairs[pair] = player_pairs.get(pair, 0) + 1
                        if pair not in pair_rounds:
                            pair_rounds[pair] = []
                        pair_rounds[pair].append(round_index)

        output.append(
            "\n##########PLAYERS WHO PLAYED TOGETHER AT LEAST TWICE##########"
        )
        for pair, count in player_pairs.items():
            if count >= 2:
                rounds = ", ".join(map(str, pair_rounds[pair]))
                output.append(
                    f"{', '.join(pair)} played together {count} times in rounds: {rounds}"
                )

        # Copy the output to the clipboard
        pyperclip.copy("\n".join(output))
        print("\n".join(output))
        print("\n\n\n\nALL RESULTS HAVE BEEN COPIED TO THE CLIPBOARD.")


# %%
# choose randomly 12 numbers between 0 and 31, without repetition
import random

numbers = random.sample(range(0, 31), 12)
good_numbers = [1, 3, 4, 5, 7, 9, 12, 15, 16, 17, 23, 27]
if __name__ == "__main__":
    list_of_players = [
        Player(main_df.loc[name]) for name in main_df.iloc[good_numbers].index
    ]
    session_of_rounds = SessionOfRounds(
        list_of_players,
        amount_of_rounds=8,
        preferences=["level"] * 8,
        level_gap_tol=1,
        num_iter=40,
        # seed=0,
    )
    # %%
    session_of_rounds.print_all_results()

# %%


# %%
