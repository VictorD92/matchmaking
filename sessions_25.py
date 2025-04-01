#%%
import main
import random
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import re
from itertools import combinations
import datetime
from math import comb
from str_to_ascii import *



#%%
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#                                                                               #
# ##    # ####### #     #       ####### ####### ####### ####### #       ####### #
# # #   # #       #     #       #     # #       #     # #     # #       #       #
# #  #  # ####### #  #  #       ####### ####### #     # ####### #       ####### #
# #   # # #       ##   ##       #       #       #     # #       #       #       #
# #    ## ####### #     # ##### #       ####### ####### #        ###### ####### #
#                                                                               #
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#%%
#%%
Lucas = pd.Series({"Name":"Lucas","Level":1.5,"Surname":"Gageiro","Gender":"Male","Games played":0},name="Lucas")
#%%
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
##############                                                    ##############
##############  #####   #####      #####   #####   #####  ####### ##############
############## #     #       #    #     # #     # #     # #       ##############
############## #     #  #####         ##  #     #     ##  ######  ##############
############## #     #       #      ##    #     #   ##          # ##############
##############  #####   #####  ## #######  #####  ####### ######  ##############
##############                                                    ##############
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
#%%

################################################################################
################################################################################
################################################################################
###############################                 ################################
###############################  #####  ####### ################################
############################### #     # #       ################################
###############################     ##  ######  ################################
###############################   ##          # ################################
############################### ####### ######  ################################
###############################                 ################################
################################################################################
################################################################################
################################################################################
#%%
df_25_03_25 = main.main_df.loc[[
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
]].copy()
# %%
df_25_03_25.sort_values("Level", inplace=True,ascending=False)
team_DLMLM = {main.Player(df_25_03_25.loc[name]) for name in ["Leo","David","Marc","Luca","Manon"]}

team_FLAVN = {main.Player(df_25_03_25.loc[name]) for name in ["VictorDi","Nathan","Felix","Luciano","Aliénor"]}
team_LCGV = {main.Player(df_25_03_25.loc[name]) for name in ["VictorDa","Linda","Colin","Gabriel"]}

team_DLMLM_level = np.mean([player.level for player in team_DLMLM])
team_FLAVN_level = np.mean([player.level for player in team_FLAVN])
team_LCGV_level = np.mean([player.level for player in team_LCGV])
for level in [team_DLMLM_level,team_FLAVN_level,team_LCGV_level]:
    print(level)


#%%
################################################################################
################################################################################
################################################################################
###############################                 ################################
###############################  #####   #####  ################################
############################### #     # #       ################################
###############################     ##  ######  ################################
###############################   ##    #     # ################################
############################### #######  #####  ################################
###############################                 ################################
################################################################################
################################################################################
################################################################################
#%%
# %%
df_26_03_25 = main.main_df.loc[[
    "VictorDa",
    "Florina",
    "Nathan",
    "Luca",
    "Manon",
    "Felix",
    "Gabriel",
    "Leo",
    "Dominik",
    "Colin",
    "Anyel",
    "Aliénor",
]].copy()
plus_1_enzo = pd.Series({"Name":"plus_1","Level":1,"Surname":"D'enzo","Gender":"Female","Games played":0},name="plus_1")
df_26_03_25 = pd.concat([df_26_03_25, pd.DataFrame([Lucas])])
df_26_03_25 = pd.concat([df_26_03_25, pd.DataFrame([plus_1_enzo])])
#%%

list_of_players = [main.Player(df_26_03_25.loc[name]) for name in df_26_03_25.index]
session_of_rounds = main.SessionOfRounds(
    list_of_players,
    amount_of_rounds=5,
    preferences=["balanced", "balanced", "level","level", "level"],
    level_gap_tol=1,
    num_iter=40,
    seed=26,
)
#%%
session_of_rounds.print_all_results()
# output:
# all players: ['VictorDa', 'Florina', 'Nathan', 'Luca', 'Manon', 'Felix', 'Gabriel', 'Leo', 'Dominik', 'Colin', 'Anyel', 'Aliénor', 'Lucas', 'plus_1']

# 1 1 1 1 1 1 1 1 ROUND 1 1 1 1 1 1 1 1 1
# preference :  balanced
# not playing: ['Manon', 'Leo']
# ------- game 1 -------
# [{'Anyel', 'Gabriel'}, {'Dominik', 'Felix'}]
# level_difference :  0.0
# ------- game 2 -------
# [{'Luca', 'Florina'}, {'Aliénor', 'Nathan'}]
# level_difference :  0.0
# ------- game 3 -------
# [{'Lucas', 'Colin'}, {'plus_1', 'VictorDa'}]
# level_difference :  0.4
# 1 1 1 1 1 1 1 1 1 ROUND END 1 1 1 1 1 1 1 1 1

# 2 2 2 2 2 2 2 2 ROUND 2 2 2 2 2 2 2 2 2
# preference :  balanced
# not playing: ['Florina', 'Colin']
# ------- game 1 -------
# [{'Luca', 'Nathan'}, {'Felix', 'Dominik'}]
# level_difference :  0.0
# ------- game 2 -------
# [{'Aliénor', 'Leo'}, {'Anyel', 'Gabriel'}]
# level_difference :  0.0
# ------- game 3 -------
# [{'Lucas', 'Plus_1'}, {'Manon', 'VictorDa'}]
# level_difference :  0.4
# 2 2 2 2 2 2 2 2 2 ROUND END 2 2 2 2 2 2 2 2 2

# 3 3 3 3 3 3 3 3 ROUND 3 3 3 3 3 3 3 3 3
# preference :  level
# not playing: ['VictorDa', 'Felix']
# ------- game 1 -------
# [{'Anyel', 'Leo'}, {'Dominik', 'Florina'}]
# name :  Leo level :  4.0
# name :  Florina level :  3.0
# name :  Dominik level :  3.7
# name :  Anyel level :  3.0
# ------- game 2 -------
# [{'Lucas', 'Nathan'}, {'Colin', 'Gabriel'}]
# name :  Aliénor level :  1.7
# name :  Colin level :  2.0
# name :  Gabriel level :  2.7
# name :  Nathan level :  3.0
# ------- game 3 -------
# [{'plus_1', 'Aliénor'}, {'Manon', 'Luca'}]
# name :  plus_1 level :  1.0
# name :  Manon level :  1.0
# name :  Luca level :  1.7
# name :  Lucas level :  1.5
# 3 3 3 3 3 3 3 3 3 ROUND END 3 3 3 3 3 3 3 3 3

# 4 4 4 4 4 4 4 4 ROUND 4 4 4 4 4 4 4 4 4
# preference :  level
# not playing: ['Luca', 'Lucas']
# ------- game 1 -------
# [{'Anyel', 'Leo'}, {'VictorDa', 'Dominik'}]
# name :  Leo level :  4.0
# name :  Dominik level :  3.7
# name :  Anyel level :  3.0
# name :  VictorDa level :  3.3
# ------- game 2 -------
# [{'Colin', 'Nathan'}, {'Gabriel', 'Florina'}]
# name :  Florina level :  3.0
# name :  Colin level :  2.0
# name :  Gabriel level :  2.7
# name :  Nathan level :  3.0
# ------- game 3 -------
# [{'Manon', 'Felix'}, {'plus_1', 'Aliénor'}]
# name :  plus_1 level :  1.0
# name :  Manon level :  1.0
# name :  Aliénor level :  1.7
# name :  Felix level :  2.0
# 4 4 4 4 4 4 4 4 4 ROUND END 4 4 4 4 4 4 4 4 4

# 5 5 5 5 5 5 5 5 ROUND 5 5 5 5 5 5 5 5 5
# preference :  level
# not playing: ['Nathan', 'Dominik']
# ------- game 1 -------
# [{'Leo', 'Florina'}, {'Anyel', 'VictorDa'}]
# name :  Leo level :  4.0
# name :  Florina level :  3.0
# name :  Anyel level :  3.0
# name :  VictorDa level :  3.3
# ------- game 2 -------
# [{'Aliénor', 'Gabriel'}, {'Colin', 'Felix'}]
# name :  Colin level :  2.0
# name :  Aliénor level :  1.7
# name :  Felix level :  2.0
# name :  Gabriel level :  2.7
# ------- game 3 -------
# [{'Manon', 'Lucas'}, {'plus_1', 'Luca'}]
# name :  plus_1 level :  1.0
# name :  Manon level :  1.0
# name :  Luca level :  1.7
# name :  Lucas level :  1.5
# 5 5 5 5 5 5 5 5 5 ROUND END 5 5 5 5 5 5 5 5 5

# ##########STATS END OF SESSION##########
# VictorDa played 4 games
# Florina played 4 games
# Nathan played 4 games
# Luca played 4 games
# Manon played 4 games
# Felix played 4 games
# Gabriel played 5 games
# Leo played 4 games
# Dominik played 4 games
# Colin played 4 games
# Anyel played 5 games
# Aliénor played 5 games
# Lucas played 4 games
# plus_1 played 5 games
# %%
#%%
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
##############                                                    ##############
##############  #####  #    #      #####   #####   #####  ####### ##############
############## #     # #    #     #     # #     # #     # #       ##############
############## #     # #######        ##  #     #     ##  ######  ##############
############## #     #      #       ##    #     #   ##          # ##############
##############  #####       #  ## #######  #####  ####### ######  ##############
##############                                                    ##############
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
#%%
################################################################################
################################################################################
################################################################################
################################                ################################
################################  #####    ##   ################################
################################ #     #  ####  ################################
################################ #     #   ##   ################################
################################ #     #   ##   ################################
################################  #####   ##### ################################
################################                ################################
################################################################################
################################################################################
################################################################################
#%%
# %%
df = main.main_df.loc[[
    "VictorDa",
    "David",
    "Aliénor",
    "Manon",
    "Felix",
    "VictorDi",
    "Nathan",
    "Colin",
    "Linda",
    "Dominik",
    "Sarah",
    "Nolan",
    "Leo",
    "Enzo",
]].copy()
#%%

list_of_players = [main.Player(df.loc[name]) for name in df.index]
session_of_rounds = main.SessionOfRounds(
    list_of_players,
    amount_of_rounds=3,
    preferences=["balanced", "balanced", "level"],
    level_gap_tol=1,
    num_iter=40,
    seed=1,
)
#%%
session_of_rounds.print_all_results()
# all players: ['VictorDa', 'David', 'Aliénor', 'Manon', 'Felix', 'VictorDi', 'Nathan', 'Colin', 'Linda', 'Dominik', 'Sarah', 'Nolan', 'Leo', 'Enzo']


# 1 1 1 1 1 1 1 1 ROUND1 1 1 1 1 1 1 1 
# preference : balanced
# not playing: ['David', 'Manon']
# ------- game 1 -------
# [{'Colin', 'Leo'}, {'Linda', 'Nathan'}]
# level_difference : 0.0
# ------- game 2 -------
# [{'Enzo', 'Sarah'}, {'Aliénor', 'VictorDi'}]
# level_difference : 0.0
# ------- game 3 -------
# [{'Dominik', 'Felix'}, {'Nolan', 'VictorDa'}]
# level_difference : 0.05
# 1 1 1 1 1 1 1 1 ROUND END 1 1 1 1 1 1 1 1 






# 2 2 2 2 2 2 2 2 ROUND2 2 2 2 2 2 2 2 
# preference : balanced
# not playing: ['Sarah', 'Enzo']
# ------- game 1 -------
# [{'Nolan', 'Leo'}, {'VictorDa', 'Nathan'}]
# level_difference : 0.0
# ------- game 2 -------
# [{'Manon', 'VictorDi'}, {'Aliénor', 'David'}]
# level_difference : 0.0
# ------- game 3 -------
# [{'Dominik', 'Colin'}, {'Felix', 'Linda'}]
# level_difference : 0.35
# 2 2 2 2 2 2 2 2 ROUND END 2 2 2 2 2 2 2 2 






# 3 3 3 3 3 3 3 3 ROUND3 3 3 3 3 3 3 3 
# preference : level
# not playing: ['Aliénor', 'Colin']
# ------- game 1 -------
# [{'Dominik', 'Enzo'}, {'Leo', 'VictorDi'}]
# name : Dominik, level : 3.7
# name : Enzo, level : 4.0
# name : VictorDi, level : 4.0
# name : Leo, level : 4.0
# ------- game 2 -------
# [{'VictorDa', 'Linda'}, {'David', 'Nathan'}]
# name : Nathan, level : 3.0
# name : David, level : 3.3
# name : Linda, level : 3.0
# name : VictorDa, level : 3.3
# ------- game 3 -------
# [{'Felix', 'Sarah'}, {'Nolan', 'Manon'}]
# name : Manon, level : 1.0
# name : Nolan, level : 2.3
# name : Sarah, level : 1.7
# name : Felix, level : 2.0
# 3 3 3 3 3 3 3 3 ROUND END 3 3 3 3 3 3 3 3


# ##########AMOUNT OF GAMES PLAYED##########
# VictorDa played 3 games
# David played 2 games
# Aliénor played 2 games
# Manon played 2 games
# Felix played 3 games
# VictorDi played 3 games
# Nathan played 3 games
# Colin played 2 games
# Linda played 3 games
# Dominik played 3 games
# Sarah played 2 games
# Nolan played 3 games
# Leo played 3 games
# Enzo played 2 games

# ##########PLAYERS WHO PLAYED TOGETHER AT LEAST TWICE######
# èh bah personne