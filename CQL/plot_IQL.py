import pandas as pd
import matplotlib.pyplot as plt
import io # Needed for reading string data as a file

# Recreate the CSV data from the user's snippet for demonstration
# In a real scenario, you would use:
df = pd.read_csv('pettingzoo_pursuit_CentralizedIQL_torch.csv')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Calculate the 10-point rolling average
# min_periods=1 ensures that we get an average even if there are fewer than 10 points at the beginning
df['AverageReward_CentralizedIQL_Rolling_10'] = df['AverageReward_CentralizedIQL'].rolling(window=10, min_periods=1).mean()

# Create the plot
plt.figure(figsize=(12, 6))

# Plot the original average reward
plt.plot(df['Episode'], df['AverageReward_CentralizedIQL'], label='每集平均奖励 (原始)', linestyle='-')

# Plot the 10-point rolling average
plt.plot(df['Episode'], df['AverageReward_CentralizedIQL_Rolling_10'], label='10集滚动平均奖励', linestyle='--')

# Add titles and labels in Chinese
plt.title('DQN 平均奖励随训练过程的变化')
plt.xlabel('训练集数 (Episode)')
plt.ylabel('平均奖励 (Average Reward)')
plt.legend() # Display the legend
plt.grid(True) # Add a grid for better readability
plt.show()
