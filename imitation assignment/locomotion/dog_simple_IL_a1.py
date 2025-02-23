import pybullet as p
import pybullet_data
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ================================
# Simple Dataset for Joint Angles from JSON
# ================================
class SimpleDataset:
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Each frame in data["Frames"] is a list.
        # We take only the first 12 entries (ignoring the timestamp if present)
        # If the first number is a timestamp, then frame[1:13] are the joint angles.
        # Adjust accordingly if your file directly contains 12 numbers.
        self.frames = []
        for frame in data["Frames"]:
            # Check if the first token appears to be a timestamp by its position (e.g., if frame length > 12)
            if len(frame) > 12:
                joint_angles = np.array(frame[1:13], dtype=np.float32)
            else:
                joint_angles = np.array(frame[:12], dtype=np.float32)
            self.frames.append(joint_angles)
        self.frames = np.array(self.frames)  # shape: (N, 12)
        self.N = self.frames.shape[0]
    
    def get_demo(self, phase):
        """
        Given a normalized phase in [0,1], linearly interpolate between frames.
        """
        idx_float = phase * (self.N - 1)
        idx0 = int(np.floor(idx_float))
        idx1 = min(idx0 + 1, self.N - 1)
        t = idx_float - idx0
        return (1 - t) * self.frames[idx0] + t * self.frames[idx1]

# ================================
# Neural Network Model
# ================================
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

# Load the A1 model.
startPos = [0, 0, 0.3]  # a reasonable standing height
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF("./a1/urdf/a1.urdf", startPos, startOrientation, useFixedBase=False)

# Get actuated joint indices (assuming all revolute/prismatic joints are actuated).
actuated_joint_indices = []
num_joints = p.getNumJoints(robotId)
for i in range(num_joints):
    info = p.getJointInfo(robotId, i)
    jointType = info[2]
    if jointType in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        actuated_joint_indices.append(i)
print("Actuated joint indices:", actuated_joint_indices)

# Reset all actuated joints to a neutral position.
for joint_index in actuated_joint_indices:
    p.resetJointState(robotId, joint_index, 0.0)

# Set simulation time step.
timeStep = 0.01667
p.setTimeStep(timeStep)

# ================================
# Create Dataset, Model, Optimizer, and Loss Function
# ================================
dataset = SimpleDataset("dog_pace.txt")
model = ImitationNet(input_dim=1, output_dim=12)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ================================
# Imitation Learning Training Loop (No Visualization During Training)
# ================================
num_episodes = 100
steps_per_episode = 100  # simulation steps per episode

print("Starting training...")
for episode in range(num_episodes):
    # Optionally, reset the robot's joint states at the beginning of each episode.
    for joint_index in actuated_joint_indices:
        p.resetJointState(robotId, joint_index, 0.0)
    
    total_loss = 0.0
    for step in range(steps_per_episode):
        # Generate a normalized phase from 0 to 1.
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
        
        # Command the robot's joints using the predicted angles.
        pred_np = pred.detach().numpy().flatten()
        for idx, joint_index in enumerate(actuated_joint_indices):
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
    phase = step / num_steps_visual  # normalized phase over one cycle
    phase_tensor = torch.tensor([[phase]], dtype=torch.float32)
    pred = model(phase_tensor)
    pred_np = pred.detach().numpy().flatten()
    for idx, joint_index in enumerate(actuated_joint_indices):
        if idx < len(pred_np):
            p.setJointMotorControl2(robotId, joint_index, p.POSITION_CONTROL,
                                    targetPosition=pred_np[idx], force=50)
    p.stepSimulation()
    time.sleep(timeStep)

print("Visualization complete.")
p.disconnect()
