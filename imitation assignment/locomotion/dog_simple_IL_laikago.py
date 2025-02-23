import pybullet as p
import pybullet_data
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ================================
# Demonstration Data Handling
# ================================

class DogPaceDataset:
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.frames = data["Frames"]
        # Each frame has 19 numbers: [timestamp, 12 joint angles, 6 body values]
        self.total_time = self.frames[-1][0]
        # For each frame, combine the 12 joint angles and 6 body values into one 18-dim configuration.
        self.configs = np.array([frame[1:19] for frame in self.frames], dtype=np.float32)
        # Compute normalized phase for each frame.
        self.phases = np.array([[frame[0] / self.total_time] for frame in self.frames], dtype=np.float32)
    
    def get_demo(self, phase):
        """
        Given a normalized phase in [0,1], linearly interpolate the demonstration.
        Returns an 18-dim configuration.
        """
        phases = self.phases.flatten()
        if phase <= phases[0]:
            return self.configs[0]
        if phase >= phases[-1]:
            return self.configs[-1]
        idx = np.searchsorted(phases, phase)
        phase0 = phases[idx - 1]
        phase1 = phases[idx]
        config0 = self.configs[idx - 1]
        config1 = self.configs[idx]
        t = (phase - phase0) / (phase1 - phase0)
        return (1 - t) * config0 + t * config1

# ================================
# Neural Network Model
# ================================

class ImitationNet(nn.Module):
    def __init__(self, input_dim=1, output_dim=18):
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
# PyBullet Setup and Robot Initialization
# ================================

# Use the provided file paths.
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")

# Create the dataset.
dataset = DogPaceDataset("./dog_pace.txt")
# Use the first demonstration frame as the initial configuration.
initial_config = dataset.configs[0]  # shape: (18,)
# Split the configuration:
# Leg joints: indices 0 to 11
# Body: indices 12 to 17 (assume first 3 are base position, next 3 are base Euler angles)
init_leg_angles = initial_config[0:12]
init_body = initial_config[12:18]
init_base_pos = init_body[3:6].tolist()
init_base_euler = init_body[0:3].tolist()
init_base_quat = p.getQuaternionFromEuler(init_base_euler)

# Set starting pose from the demonstration.
startPos = init_base_pos
startOrientation = init_base_quat

robotId = p.loadURDF("./a1/urdf/a1.urdf", startPos, startOrientation, useFixedBase=False)

# Identify the actuated joint indices.
actuated_joint_indices = []
num_joints = p.getNumJoints(robotId)
for i in range(num_joints):
    info = p.getJointInfo(robotId, i)
    jointType = info[2]
    if jointType in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        actuated_joint_indices.append(i)
print("Actuated joint indices:", actuated_joint_indices)

# Initialize the leg joints with the demonstration values.
for idx, joint_index in enumerate(actuated_joint_indices):
    if idx < len(init_leg_angles):
        p.resetJointState(robotId, joint_index, init_leg_angles[idx])

# Set simulation time step.
timeStep = 0.01667
p.setTimeStep(timeStep)

# ================================
# Create Model, Optimizer, and Loss Function
# ================================

model = ImitationNet(input_dim=1, output_dim=18)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ================================
# Imitation Learning Training Loop (No Visualization During Training)
# ================================

num_episodes = 100
steps_per_episode = 100  # simulation steps per episode

print("Starting training...")
for episode in range(num_episodes):
    # Reset the robot to the demonstration's initial configuration.
    p.resetBasePositionAndOrientation(robotId, startPos, startOrientation)
    for idx, joint_index in enumerate(actuated_joint_indices):
        if idx < len(init_leg_angles):
            p.resetJointState(robotId, joint_index, init_leg_angles[idx])
    
    total_loss = 0.0
    for step in range(steps_per_episode):
        t_sim = step * timeStep
        # Compute normalized phase (cyclic).
        phase = (t_sim % dataset.total_time) / dataset.total_time
        
        # Get ground-truth configuration from the demonstration.
        demo_target = dataset.get_demo(phase)  # shape: (18,)
        demo_target_tensor = torch.tensor(demo_target, dtype=torch.float32).unsqueeze(0)
        
        phase_tensor = torch.tensor([[phase]], dtype=torch.float32)
        pred = model(phase_tensor)
        
        loss = criterion(pred, demo_target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Command the robot using the predicted leg joint angles (first 12 outputs).
        pred_np = pred.detach().numpy().flatten()
        leg_targets = pred_np[0:12]
        for idx, joint_index in enumerate(actuated_joint_indices):
            if idx < len(leg_targets):
                p.setJointMotorControl2(robotId, joint_index, p.POSITION_CONTROL,
                                        targetPosition=leg_targets[idx], force=50)
        p.stepSimulation()
    avg_loss = total_loss / steps_per_episode
    print(f"Episode {episode+1}/{num_episodes}, Average Loss: {avg_loss:.6f}")

# ================================
# Final Visualization: Replay the Learned Motion
# ================================

print("Training complete. Final visualization...")
num_steps_visual = 200
for step in range(num_steps_visual):
    t_sim = step * timeStep
    phase = (t_sim % dataset.total_time) / dataset.total_time
    phase_tensor = torch.tensor([[phase]], dtype=torch.float32)
    pred = model(phase_tensor)
    pred_np = pred.detach().numpy().flatten()
    leg_targets = pred_np[0:12]
    for idx, joint_index in enumerate(actuated_joint_indices):
        if idx < len(leg_targets):
            p.setJointMotorControl2(robotId, joint_index, p.POSITION_CONTROL,
                                    targetPosition=leg_targets[idx], force=50)
    p.stepSimulation()
    time.sleep(timeStep)

print("Visualization complete.")
p.disconnect()
