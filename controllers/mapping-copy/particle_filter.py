import numpy as np
from robot_controller import RobotController
from map_update import update_map

class Particle:
    def __init__(self, x, y, theta, weight):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight

def initialize_particles(num_particles, initial_position):
    particles = []
    for _ in range(num_particles):
        x, y, theta = initial_position
        particles.append(Particle(x + np.random.randn() * 0.1,
                                y + np.random.randn() * 0.1,
                                theta + np.random.randn() * 0.1,
                                1.0))  # Initialize weight to 1.0
    return particles

def resample(particles):
    weights = np.array([p.weight for p in particles])
    weights /= np.sum(weights)
    indices = np.random.choice(len(particles), len(particles), p=weights)
    new_particles = [particles[i] for i in indices]
    return new_particles
def motion_model(particle, robot, timestep):
    """
    Updates particle pose based on robot's odometry data retrieved from Webots.

    Args:
        particle: The particle to update.
        robot: A Webots robot instance.
        timestep: The simulation timestep in milliseconds.

    Returns:
        The updated particle with a new pose.
    """

    # Get left and right wheel motor devices (replace with actual names)
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")

    # Get previous encoder values (initialize on first call)
    if not hasattr(motion_model, 'prev_left_encoder'):
        motion_model.prev_left_encoder = left_motor.getTargetPosition()
        motion_model.prev_right_encoder = right_motor.getTargetPosition()

    # Update only if enough timesteps have passed since last update
    if timestep % (params["updateFreq"] * robot.getBasicTimeStep()) == 0:
        # Get current encoder values
        current_left_encoder = left_motor.getTargetPosition()
        current_right_encoder = right_motor.getTargetPosition()

        # Calculate wheel distances based on encoder change and wheel radius
        left_wheel_distance = (current_left_encoder - motion_model.prev_left_encoder) * (2 * np.pi * params["wheelRadius"]) / params["encoder_ticks_per_revolution"]  # Replace with actual encoder resolution
        right_wheel_distance = (current_right_encoder - motion_model.prev_right_encoder) * (2 * np.pi * params["wheelRadius"]) / params["encoder_ticks_per_revolution"]  # Replace with actual encoder resolution

        # Update previous encoder values for next call
        motion_model.prev_left_encoder = current_left_encoder
        motion_model.prev_right_encoder = current_right_encoder

        # Differential drive kinematics
        distance = (left_wheel_distance + right_wheel_distance) / 2.0
        theta = (right_wheel_distance - left_wheel_distance) / params["axleLength"]

        # Update particle pose based on odometry
        particle.x += distance * np.cos(particle.theta)
        particle.y += distance * np.sin(particle.theta)
        particle.theta += theta

    # Consider adding a simple odometry noise model (optional)

    return particle

def sensor_model(particle, sensor_data, map, lidar_fov, lidar_resolution, grid_size, grid_resolution):
    weight = 1.0
    num_measurements = len(sensor_data)
    for i in range(num_measurements):
        distance = sensor_data[i]
        angle = i * (lidar_fov / num_measurements) - (lidar_fov / 2)
        x, y = calculate_grid_coordinates(distance, angle, (particle.x, particle.y, particle.theta), grid_size, grid_resolution)
        if x is not None and y is not None and is_within_grid(x, y, map) and map[x][y] != -1:
            expected_distance = np.linalg.norm([x - particle.x, y - particle.y])
            # Add a sensor noise model (e.g., Gaussian noise)
            sensor_noise = np.random.normal(0, 0.05)  # Example Gaussian noise with mean 0 and stddev 0.05
            weight *= np.exp(-((distance - expected_distance + sensor_noise) ** 2) / (2 * 0.05**2))  # Adjust variance based on sensor noise

    particle.weight = weight
    return particle

def resample(particles):
    weights = np.array([p.weight for p in particles])
    weights /= np.sum(weights)
    indices = np.random.choice(len(particles), len(particles), p=weights)
    new_particles = [particles[i] for i in indices]
    return new_particles


