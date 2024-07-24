import math
import numpy as np
import cv2
import torch
import csv

class SDAE(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(SDAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_sdae(model, data, num_epochs=100, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)
    
    previous_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data)
        loss.backward()
        
        optimizer.step()
        
        current_loss = loss.item()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {current_loss:.4f}')
        
        if abs(previous_loss - current_loss) < 1e-4:
            print(f'Converged at epoch {epoch+1} with loss: {current_loss:.4f}')
            break
        
        previous_loss = current_loss
    
    return model

def estimate_measurement_noise_covariance(sdae_model, sensor_data):
    denoised_data = sdae_model(sensor_data).detach().numpy()
    S_nf = denoised_data
    S_n = sensor_data.detach().numpy()

    delta_x = S_nf[:, 0] - S_n[:, 0]
    delta_y = S_nf[:, 1] - S_n[:, 1]
    delta_v = S_nf[:, 2] - S_n[:, 2]

    R = np.array([
        [np.mean(delta_x ** 2), 0, 0],
        [0, np.mean(delta_y ** 2), 0],
        [0, 0, np.mean(delta_v ** 2)]
    ])
    return R

def sdae_based_ekf_prediction(sdae_model, ekf, sensor_data, control_input, measurement):
    denoised_data = sdae_model(sensor_data).detach().numpy()
    R = estimate_measurement_noise_covariance(sdae_model, sensor_data)
    ekf.set_measurement_noise_covariance(R)
    ekf.predict(control_input)
    ekf.update(measurement)
    estimated_state = ekf.x
    return estimated_state
