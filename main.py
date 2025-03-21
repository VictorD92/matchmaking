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
        ],
        "Level": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 1, 2],
        "Gender": ["Female", "Male"] * 6,
        "Games played": [0] * 12,
        "Happiness": [0] * 12,
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
            set_all_levels_people_playing = {
                player.level for player in self.people_playing
            }
            list_decreasing_levels = sorted(set_all_levels_people_playing, reverse=True)
            # we sort the players by level
            dic_level_players = {
                level: set() for level in set_all_levels_people_playing
            }
            for player in self.people_playing:
                dic_level_players[player.level].add(player)

            # we check if the amount of players in each level is a multiple of the amount of self.players_per_team * self.teams_per_game
            # if not, we randomly add a player from one level down
            for level in list_decreasing_levels:
                while (
                    len(dic_level_players[level])
                    % (self.players_per_team * self.teams_per_game)
                    != 0
                ):
                    random_player = random.sample(
                        list(dic_level_players[level - 1]), 1
                    )[0]
                    dic_level_players[level].add(random_player)
                    dic_level_players[level - 1].remove(random_player)
            

            for level, players in dic_level_players.items():
                print(level, [player.name for player in players])
            # we create the games in each level
            list_levels_not_empty = [
                level
                for level in list_decreasing_levels
                if dic_level_players[level] != set()
            ]
            for level in list_levels_not_empty:
                people_left_to_play_in_level = dic_level_players[level]
                print("people left to play in level at start:")
                print([player.name for player in people_left_to_play_in_level])
                for iter in range(self.num_iter):
                    for i in range(
                        len(people_left_to_play_in_level)
                        // (self.players_per_team * self.teams_per_game)
                    ):
                        people_left_to_play_in_level = (
                            people_left_to_play_in_level
                            - set().union(*{game.participants for game in self.games})
                        )
                        print("people removed:")
                        print([player.name for player in set().union(*{game.participants for game in self.games})])
                        print("people left to play in level:")
                        print([player.name for player in people_left_to_play_in_level])
                        self.games.append(
                            self.create_balanced_game(people_left_to_play_in_level)
                        )

                    if max([game.level_difference for game in self.games]) == 0:
                        print("game players:")
                        print([participant.name for participant in game.participants])
                        break
                    if (
                        max([game.level_difference for game in self.games])
                        <= self.level_gap_tol
                    ) and (iter > self.num_iter // 2):
                        print("game players:")
                        print([participant.name for participant in game.participants])
                        break
                    self.games = []

                if self.games == []:
                    print("could not find a game, because the tolerance is too low")
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
            list_of_teams = list(set_of_teams)
            np.random.shuffle(list_of_teams)

            for i in range(self.num_iter):
                possible_team_combination = combinations(
                    list_of_teams, self.teams_per_game
                )
                for team_combination in possible_team_combination:
                    if set().union(*set(possible_team_combination)) == set_of_teams:
                        break
            return GameOfFour(*team_combination, preference="None")
        return "could not find a game, because there are not enough players"


# %%
!!!!no error but only level 0 is done!!!!
set_of_players = {Player(df_minimal_example.iloc[i]) for i in range(12)}
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
!!!!ValueError: Sample larger than population or is negative!!!
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
        elif len(games_per_round_each_round) > amount_of_rounds:
            return "too many rounds chosen for variable games_per_round_each_round"
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
        elif len(players_per_team_each_round) > amount_of_rounds:
            return "too many rounds chosen for variable players_per_team_each_round"
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
set_of_players = {Player(main_df.iloc[i]) for i in range(7)}
session_of_rounds = SessionOfRounds(
    set_of_players,
    amount_of_rounds=5,
    preferences="balanced",
    level_gap_tol=4,
    num_iter=40,
)
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
        for attr in ["level_difference"]:
            print(attr + " : ", getattr(game, attr))
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

# %%

# %%
set_of_teams = create_set_of_all_possible_teams(set(list_of_players[:7]))
# %%
balanced_game = create_balanced_game(set_of_teams, balanced=True)
for attr in ["preference", "teams", "participants", "level_difference"]:
    print(attr + " : ", getattr(balanced_game, attr))


# %%

# %%


# %%
def create_balanced_list_of_four(out_df, ordered_criteria=None, players_per_team=2):
    # criteria_ordonnes is a list of criteria (columns of df) to sort the players by similarity
    list_criterion_types = [true_dtypes[critere] for critere in ordered_criteria]
    if ordered_criteria is None:
        return create_random_sub_dataframe_of_four(out_df)
    out_df = out_df.copy()
    team_1 = []
    team_2 = []

    out_df["Category"] = [
        [out_df.loc[ind, critere] for critere in ordered_criteria]
        for ind in out_df.index
    ]
    if list_criterion_types == []:
        temp_df = create_random_sub_dataframe_of_four(out_df)
        return GameOfFour(temp_df)
    i = 0
    while i < len(list_criterion_types):
        print("dealing with", ordered_criteria[i])
        if is_numeric_dtype(out_df[ordered_criteria[i]]):
            print("criterion is numeric")
            dist_df = pd.DataFrame(index=out_df.index, columns=out_df.index).fillna(0)
            for player_1 in out_df.index:
                for player_2 in out_df.drop(player_1).index:
                    dist_df.loc[player_1, player_2] = abs(
                        out_df.loc[player_1, ordered_criteria[i]]
                        - out_df.loc[player_2, ordered_criteria[i]]
                    )
            min_dist = dist_df.min().min()
            minimal_pairs = []
            for player_1 in dist_df.index:
                for player_2 in dist_df.drop(player_1).index:
                    if dist_df.loc[player_1, player_2] == min_dist:
                        if frozenset([player_1, player_2]) not in minimal_pairs:
                            minimal_pairs.append(frozenset([player_1, player_2]))
            minimal_games = []
            print(minimal_pairs)

            for pair_1 in minimal_pairs:
                for pair_2 in minimal_pairs:
                    if pair_1 & pair_2 == set():
                        list_1 = list(pair_1)
                        np.random.shuffle(list_1)
                        list_2 = list(pair_2)
                        np.random.shuffle(list_2)
                        for zero_one in range(2):
                            ordered_combined_players = [
                                list_1[0],
                                list_2[zero_one],
                                list_1[1],
                                list_2[1 - zero_one],
                            ]
                            temp_game_of_four = GameOfFour(
                                out_df.loc[ordered_combined_players],
                                preference=ordered_criteria[i],
                            )
                            if temp_game_of_four.teams not in [
                                minimal_game.teams for minimal_game in minimal_games
                            ]:
                                minimal_games.append(temp_game_of_four)

            return minimal_games
            if minimal_games == []:
                game = np.random.choice(possible_games)
            possible_games = minimal_games

            i += 1
        elif ordered_criteria[i] == "Mixt":
            print("criterion is mixte")
            df_different_gender = pd.DataFrame(
                index=out_df.index, columns=out_df.index
            ).fillna(False)
            for player_1 in out_df.index:
                for player_2 in out_df.drop(player_1).index:
                    if out_df.loc[player_1, "Mixt"] != out_df.loc[player_2, "Mixt"]:
                        df_different_gender.loc[player_1, player_2] = True
            diferent_gender_pairs = [
                (player_1, player_2)
                for player_1 in df_different_gender.index
                for player_2 in df_different_gender.columns
                if df_different_gender[player_1, player_2]
            ]
            i += 1

        else:
            print("criterion is categorical")
            for player_1 in out_df.index:
                for player_2 in out_df.drop(player_1).index:
                    break


# %%
games = create_balanced_list_of_four(example_df, ["Level"])
# %%
len(games)
# %%
for i in range(len(games)):
    print(games[i].teams)


# %%
def create_random_sub_dataframe_of_four(out_df, separate_by_categories=False):
    # function that picks 4 random people from the df, according to the preference
    # df is the main dataframe containing all the players

    out_df = out_df.copy()
    if separate_by_categories:
        categories_with_at_least_four = [
            cat
            for cat in out_df["Category"].unique()
            if len(out_df[out_df["Category"] == cat]) >= 4
        ]
        if categories_with_at_least_four != []:
            random_category = np.random.choice(categories_with_at_least_four)
            out_df = out_df[out_df["Category"] == random_category]

    random_four_players = np.random.choice(out_df.index, 4, replace=False)

    return out_df.loc[random_four_players]


# %%


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

main_df["Niveau"] = level_df["Niveau"]
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
short_example_df = main_df.loc[["VictorDa", "David", "Bram", "Anyel", "Nolan"]]


# %%

create_random_sub_dataframe_of_four(main_df, separate_by_categories=True)["Level"]


# %%

# %%
round_of_games = RoundOfGames(main_df, 8, preference="Level")
games = round_of_games.games

# %%
round_of_games.not_playing
# %%
result = (1 / 2) * comb(15, 2) * comb(13, 2)
print(result)
# %%
len(example_df)


# %%
# %%
for i in range(8):
    list_of_players = [Player(main_df.iloc[i]) for i in range(len(main_df))]
    set_of_teams = create_set_of_all_possible_teams(set(list_of_players))
    game = create_balanced_game(set_of_teams, 2, True)
    print(" et ".join(game.team_A.players), " VS ", " et ".join(game.team_B.players))
# %%
team = list(set_of_teams)[1]
print("joueurs: ", team.player_A.name, " et ", team.player_B.name)
print("niveau moyen: ", team.mean_level)
print("difference de niveau: ", team.level_difference)
print("mixte: ", team.mixed)
print("masculin: ", team.male)
# %%
