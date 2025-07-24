import pandas as pd

# Correct local paths for your machine
train = pd.read_csv(r'd:\Niti\Career\Hackathons+Projects\Shell.ai Hackathon\train.csv')
test = pd.read_csv(r'd:\Niti\Career\Hackathons+Projects\Shell.ai Hackathon\test.csv')
sample_submission = pd.read_csv(r'd:\Niti\Career\Hackathons+Projects\Shell.ai Hackathon\sample_solution.csv')

# Show basic info
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample submission shape:", sample_submission.shape)

# Preview few rows
print("\nTrain Head:\n", train.head())
print("\nTest Head:\n", test.head())
print("\nSample Submission Head:\n", sample_submission.head())