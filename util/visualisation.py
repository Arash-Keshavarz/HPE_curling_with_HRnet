import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming you have a CSV file named 'your_file.csv'
file_path_1 = 'output/Person1.csv'
file_path_2 = 'output/Person2.csv'
# Read the CSV file into a Pandas DataFrame
df_1 = pd.read_csv(file_path_1)
df_2 = pd.read_csv(file_path_2)

# Convert DataFrame columns to numpy arrays
frames_1 = df_1['Frame'].values
frames_2 = df_2['Frame'].values

right_knee_1 = df_1['Right Knee Angle'].values
right_knee_2 = df_2['Right Knee Angle'].values

if np.size(frames_1) >= np.size(frames_2):
    frames = frames_2
    right_knee_1 = right_knee_1[0:np.size(frames)]
else:
    frames = frames_1
    right_knee_2 = right_knee_2[0:np.size(frames)]




plt.figure(figsize=(12,6))
# Plot 'left_arm & right_arm' against 'Frames'
plt.plot(frames, right_knee_1, label='Right Knee Angle (Person_1)', marker= '.')
plt.plot(frames, right_knee_2, label='Right Knee Angle (Person_2)', marker='.')
# Add labels and title
plt.xlabel('Frames')
plt.ylabel('Angles')
plt.title('Angles Over Frames(Person1&2)')

# Add a legend (if needed)
plt.legend()

#plt.savefig('output/angles_plot_person1.png')

# Show the plot
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming you have a CSV file named 'your_file.csv'
file_path_1 = 'output/Person1.csv'
file_path_2 = 'output/Person2.csv'
# Read the CSV file into a Pandas DataFrame
df_1 = pd.read_csv(file_path_1)
df_2 = pd.read_csv(file_path_2)

# Convert DataFrame columns to numpy arrays
frames_1 = df_1['Frame'].values
frames_2 = df_2['Frame'].values
left_arm_1 = df_1['Left Arm Angle'].values
left_arm_2 = df_2['Left Arm Angle'].values


if np.size(frames_1) >= np.size(frames_2):
    frames = frames_2
    left_arm_1 = left_arm_1[0:np.size(frames)]
else:
    frames = frames_1
    left_arm_2 = left_arm_2[0:np.size(frames)]


plt.figure(figsize=(12,6))
# Plot 'left_arm & right_arm' against 'Frames'
plt.plot(frames, left_arm_1, label='Left Arm Angle (Person_1)', marker= '.')
plt.plot(frames, left_arm_2, label='Left Arm Angle (Person_2)', marker='.')
# Add labels and title
plt.xlabel('Frames')
plt.ylabel('Angles')
plt.title('Angles Over Frames(Person1&2)')

# Add a legend (if needed)
plt.legend()

#plt.savefig('output/angles_plot_person1.png')

# Show the plot
plt.show()
