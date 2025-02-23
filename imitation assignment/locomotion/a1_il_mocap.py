import pybullet as p
import pybullet_data
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ================================
# Simple Dataset for Joint Angles from Plain Text
# ================================
# Each line in "dog_pace.txt" is expected to have 14 comma-separated values:
#   index, timestamp, angle1, angle2, ..., angle12
# We ignore the first two values and use the next 12 as our data.
class SimpleDataset:
    def __init__(self, file_path):
        self.frames = []
        with open(file_path, 'r') as f:
            for line in f:
                tokens = line.strip().split(',')
                if len(tokens) < 14:
                    continue
                try:
                    angles = [float(tok.strip()) for tok in tokens[2:14]]
                    self.frames.append(np.array(angles, dtype=np.float32))
                except ValueError as e:
                    print("Skipping line due to conversion error:", line)
        self.frames = np.array(self.frames)  # shape: (N, 12)
        self.N = self.frames.shape[0]
    
    def get_demo(self, phase):
        # Linearly interpolate between frames based on normalized phase [0,1]
        idx_float = phase * (self.N - 1)
        idx0 = int(np.floor(idx_float))
        idx1 = min(idx0 + 1, self.N - 1)
        t = idx_float - idx0
        return (1 - t) * self.frames[idx0] + t * self.frames[idx1]

# ================================
# Neural Network Model
# ================================
# Maps a 1D phase (0 to 1) to 12 joint angles.
class ImitationNet(nn.Module):
    def __init__(self, input_dim=1, output_dim=12):
        super(ImitationNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ================================
# PyBullet Setup and A1 Initialization
# ================================
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")

# Load the A1 robot.
startPos = [0, 0, 0.3]  # a reasonable standing height
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF("./a1/urdf/a1.urdf", startPos, startOrientation, useFixedBase=False)

# Get all actuated joint indices.
raw_joint_indices = []
num_joints = p.getNumJoints(robotId)
for i in range(num_joints):
    info = p.getJointInfo(robotId, i)
    jointType = info[2]
    if jointType in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        raw_joint_indices.append(i)

# Print out joint names to see the ordering.
joint_names = {}
for i in raw_joint_indices:
    info = p.getJointInfo(robotId, i)
    name = info[1].decode('utf-8')
    joint_names[name] = i
    print(f"Joint index {i}: {name}")

# Define the expected order for the 12 joints.
expected_joint_names = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
]

# Reorder the joints if possible.
sorted_actuated_indices = []
for name in expected_joint_names:
    if name in joint_names:
        sorted_actuated_indices.append(joint_names[name])
    else:
        print(f"Warning: expected joint '{name}' not found!")
# Fallback: if reordering fails, use the raw order.
if len(sorted_actuated_indices) != 12:
    print("Using raw joint indices order instead of expected order.")
    sorted_actuated_indices = raw_joint_indices

print("Ordered actuated joint indices:", sorted_actuated_indices)

# Reset all actuated joints to zero.
for joint_index in sorted_actuated_indices:
    p.resetJointState(robotId, joint_index, 0.0)

timeStep = 0.01667
p.setTimeStep(timeStep)

# ================================
# Create Dataset, Model, Optimizer, and Loss
# ================================
dataset = SimpleDataset("dog_pace.txt")
model = ImitationNet(input_dim=1, output_dim=12)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ================================
# Imitation Learning Training Loop (No Visualization During Training)
# ================================
num_episodes = 100
steps_per_episode = 100

print("Starting training...")
for episode in range(num_episodes):
    for joint_index in sorted_actuated_indices:
        p.resetJointState(robotId, joint_index, 0.0)
    
    total_loss = 0.0
    for step in range(steps_per_episode):
        phase = step / steps_per_episode
        demo_target = dataset.get_demo(phase)  # shape: (12,)
        demo_target_tensor = torch.tensor(demo_target, dtype=torch.float32).unsqueeze(0)
        phase_tensor = torch.tensor([[phase]], dtype=torch.float32)
        pred = model(phase_tensor)
        loss = criterion(pred, demo_target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        pred_np = pred.detach().numpy().flatten()
        # Use the sorted joint order to command the robot.
        for idx, joint_index in enumerate(sorted_actuated_indices):
            if idx < len(pred_np):
                p.setJointMotorControl2(robotId, joint_index, p.POSITION_CONTROL,
                                        targetPosition=pred_np[idx], force=50)
        p.stepSimulation()
    
    avg_loss = total_loss / steps_per_episode
    print(f"Episode {episode+1}/{num_episodes}, Average Loss: {avg_loss:.6f}")

# ================================
# Final Visualization: Replay the Learned Motion
# ================================
print("Training complete. Final visualization...")
num_steps_visual = 200
for step in range(num_steps_visual):
    phase = step / num_steps_visual
    phase_tensor = torch.tensor([[phase]], dtype=torch.float32)
    pred = model(phase_tensor)
    pred_np = pred.detach().numpy().flatten()
    for idx, joint_index in enumerate(sorted_actuated_indices):
        if idx < len(pred_np):
            p.setJointMotorControl2(robotId, joint_index, p.POSITION_CONTROL,
                                    targetPosition=pred_np[idx], force=50)
    p.stepSimulation()
    time.sleep(timeStep)

print("Visualization complete.")
p.disconnect()
