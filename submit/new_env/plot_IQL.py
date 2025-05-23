import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('IQL_log_n.csv')

df['AverageReward_IQL_10'] = df['AverageReward_IQL'].rolling(window=10, min_periods=1).mean()
df['AverageReward_IQL_30'] = df['AverageReward_IQL'].rolling(window=30, min_periods=1).mean()

plt.figure(figsize=(12, 6))
plt.plot(df['Episode'], df['AverageReward_IQL'], label='Average Reward Every 10 Episode', linestyle='-')
plt.plot(df['Episode'], df['AverageReward_IQL_10'], label='Average Reward Every 100 Episode', linestyle='--')
# plt.plot(df['Episode'], df['AverageReward_IQL_30'], label='Average Reward Every 30 Episode', linestyle='--')
plt.title('Changes on Average Reward during training by IQL')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True)
plt.show()
