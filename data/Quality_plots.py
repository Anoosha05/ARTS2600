"""
Quality of the conversations
Prof. Tristram Alexander, Anoosha Mallakanti, Jacob Stein
"""
import os 
import matplotlib as plt
import numpy as np 
import pandas as pd


# In this file, we aim to look the quality of the conversations. 
# Method 1: With the X-axis representing the total converstaions and Y axis representing the fraction 
#       of conversation being led by a userid. Points in the 0.5 region are of interest as these represent 
#       a 'dialogue' - we are trying to filter out the monologues and extremely diverse conversations. 

# For the "Analysis of Trump" file, 
# 1. the first line gives the length of the convo 
# 2. the @... gives the userid  
# 3. list_1 = [@user1, @user2, @user1....]

os.chdir("/Users/anooshamallakanti/Desktop/twitter_data")
print("The current working directory is ", os.getcwdb())

file_path = r"/Users/anooshamallakanti/Desktop/twitter_data/long_conversation_analysis_trump_01_12_2017_text.txt"

# Pseudo code 
"""
iterate through each 'From user' line and 
"""


