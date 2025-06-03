""" 
Twitter data - Conversation length Distributions 
Prof. Tristram Alexander, Anoosha Mallakanti, Jacob Stein
"""

import os 
import numpy as np 
import matplotlib.pyplot as plt

os.chdir('/Users/anooshamallakanti/Desktop/')
print(os.getcwd())

# Read the txt file 
file_path1 = r"/Users/anooshamallakanti/Desktop/twitter_data/conversations_superbowl.txt"
conversation_length = []

# iterate through each line of the file to look for the numbers/lengths
with open(file_path1, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line.isdigit():
            conversation_length.append(int(line))

# convert to an array 
conversation_length = np.array(conversation_length)
print(conversation_length)

# Compute statistics
mean_length = np.mean(conversation_length)
median_length = np.median(conversation_length)
std_dev = np.std(conversation_length)

# Print statistics
print(f"Total conversations: {len(conversation_length)}")
print(f"Mean length: {mean_length:.2f}")
print(f"Median length: {median_length}")
print(f"Standard Deviation: {std_dev:.2f}")

# Plot histogram with the frequency of the convo lengths 
bins = np.arange(min(conversation_length), max(conversation_length) + 2)  # One bin per length
print("the longest convo was: ", max(conversation_length))
print("the shortest convo was: ", min(conversation_length))

plt.figure(figsize=(8, 6))
plt.hist(conversation_length, bins=bins, color="skyblue", edgecolor="black", alpha=0.75)
plt.xlabel("Conversation Length")
plt.ylabel("Frequency")
plt.title("Distribution of Superbowl Conversation Lengths")
plt.xlim(0, np.percentile(conversation_length, 99))  # Adjust x-axis to focus on most data
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()






# plotting the Trump convos 
# Read the txt file 
file_path2 = r"/Users/anooshamallakanti/Desktop/twitter_data/long_conversation_analysis_trump_01_12_2017_text.txt"
conversation_length = []

# iterate through each line of the file to look for the numbers/lengths
with open(file_path2, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line.isdigit():
            conversation_length.append(int(line))

# convert to an array 
conversation_length = np.array(conversation_length)
#print(conversation_length)

# Compute statistics
mean_length = np.mean(conversation_length)
median_length = np.median(conversation_length)
std_dev = np.std(conversation_length)

# Print statistics
print(f"Total conversations: {len(conversation_length)}")
print(f"Mean length: {mean_length:.2f}")
print(f"Median length: {median_length}")
print(f"Standard Deviation: {std_dev:.2f}")

# Plot histogram with the frequency of the convo lengths 
bins = np.arange(min(conversation_length), max(conversation_length) + 5)  
print("the longest convo was: ", max(conversation_length))
print("the shortest convo was: ", min(conversation_length))

plt.figure(figsize=(8, 6))
plt.hist(conversation_length, bins=bins, color="skyblue", edgecolor="black", alpha=0.75)
plt.xlabel("Conversation Length")
plt.ylabel("Frequency")
plt.title("Distribution of Trump Conversation Lengths")
plt.xlim(0, np.percentile(conversation_length, 99))  # Adjust x-axis to focus on most data
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()



# Plotting State of the Union convos 
# Read the txt file 
file_path3 = r"/Users/anooshamallakanti/Desktop/twitter_data/conversations_StateOfTheUnion_2018_02_02.txt"
conversation_length = []

# iterate through each line of the file to look for the numbers/lengths
with open(file_path3, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line.isdigit():
            conversation_length.append(int(line))

# convert to an array 
conversation_length = np.array(conversation_length)
print(conversation_length)

# Compute statistics
mean_length = np.mean(conversation_length)
median_length = np.median(conversation_length)
std_dev = np.std(conversation_length)

# Print statistics
print(f"Total conversations: {len(conversation_length)}")
print(f"Mean length: {mean_length:.2f}")
print(f"Median length: {median_length}")
print(f"Standard Deviation: {std_dev:.2f}")

# Plot histogram with the frequency of the convo lengths 
bins = np.arange(min(conversation_length), max(conversation_length) + 5)  
print("the longest convo was: ", max(conversation_length))
print("the shortest convo was: ", min(conversation_length))

plt.figure(figsize=(8, 6))
plt.hist(conversation_length, bins=bins, color="skyblue", edgecolor="black", alpha=0.75)
plt.xlabel("Conversation Length")
plt.ylabel("Frequency")
plt.title("Distribution of Trump Conversation Lengths")
plt.xlim(0, np.percentile(conversation_length, 99))  # Adjust x-axis to focus on most data
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()



# Plotting Australia Day convos
# Read the txt file 
file_path4 = r"/Users/anooshamallakanti/Desktop/twitter_data/conversations_Australia_Day.txt"
conversation_length = []

# iterate through each line of the file to look for the numbers/lengths
with open(file_path4, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line.isdigit():
            conversation_length.append(int(line))

# convert to an array 
conversation_length = np.array(conversation_length)
print(conversation_length)

# Compute statistics
mean_length = np.mean(conversation_length)
median_length = np.median(conversation_length)
std_dev = np.std(conversation_length)

# Print statistics
print(f"Total conversations: {len(conversation_length)}")
print(f"Mean length: {mean_length:.2f}")
print(f"Median length: {median_length}")
print(f"Standard Deviation: {std_dev:.2f}")

# Plot histogram with the frequency of the convo lengths 
bins = np.arange(min(conversation_length), max(conversation_length) + 5)  
print("the longest convo was: ", max(conversation_length))
print("the shortest convo was: ", min(conversation_length))


plt.figure(figsize=(8, 6))
plt.hist(conversation_length, bins=bins, color="skyblue", edgecolor="black", alpha=0.75)
plt.xlabel("Conversation Length")
plt.ylabel("Frequency")
plt.title("Distribution of Trump Conversation Lengths")
plt.xlim(0, np.percentile(conversation_length, 99))  # Adjust x-axis to focus on most data
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
