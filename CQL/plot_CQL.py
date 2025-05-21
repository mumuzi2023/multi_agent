import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('CQL_log.csv')

df['AverageReward_CQL_10'] = df['AverageReward_CQL'].rolling(window=10, min_periods=1).mean()
df['AverageReward_CQL_30'] = df['AverageReward_CQL'].rolling(window=30, min_periods=1).mean()

plt.figure(figsize=(12, 6))
plt.plot(df['Episode'], df['AverageReward_CQL'], label='Average Reward per Episode', linestyle='-')
plt.plot(df['Episode'], df['AverageReward_CQL_10'], label='Average Reward Every 10 Episode', linestyle='--')
plt.plot(df['Episode'], df['AverageReward_CQL_30'], label='Average Reward Every 30 Episode', linestyle='--')

plt.title('Changes on Average Reward During training by CQL')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True)
plt.show()
