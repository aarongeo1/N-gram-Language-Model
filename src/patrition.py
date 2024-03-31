import random
import os

input_file = "assn3-lm-sounds-martinmj4im\src\consolidated_data.txt"
training_file = "training.txt"
dev_file = "dev.txt"

with open(input_file, "r") as file:
    data = file.readlines()


random.shuffle(data) # Shuffle the data randomly

# Calculate the split point for 80/20 division
split = int(0.8 * len(data))

training_data = data[:split]
dev_data = data[split:]

with open(training_file, "w") as file:
    file.writelines(training_data)

with open(dev_file, "w") as file:
    file.writelines(dev_data)

# print(f"Data split into {len(training_data)} lines for training and {len(dev_data)} lines for dev.")
