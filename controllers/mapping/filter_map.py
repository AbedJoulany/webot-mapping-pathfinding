import numpy as np
import pandas as pd

def load_map(file_path):
    """Load the map from a CSV file."""
    return pd.read_csv(file_path, header=None).values

def filter_similar_points(points, distance_threshold=0.001):
    """
    Exclude points that are within a certain distance threshold of each other.
    
    :param points: Array of points to filter.
    :param distance_threshold: Minimum distance between points.
    :return: Filtered array of points.
    """
    if len(points) == 0:
        return points

    filtered_points = []
    for point in points:
        if all(np.linalg.norm(point - fp) > distance_threshold for fp in filtered_points):
            filtered_points.append(point)

    return np.array(filtered_points)

def save_filtered_map(filtered_points, output_file_path):
    """Save the filtered points to a CSV file."""
    pd.DataFrame(filtered_points).to_csv(output_file_path, header=None, index=False)

def main():
    input_file_path = 'C:/Users/abeda/webot-mapping-pathfinding/controllers/mapping/map.csv'
    output_file_path = 'filtered_map.csv'

    points = load_map(input_file_path)
    filtered_points = filter_similar_points(points, distance_threshold=0.001)
    save_filtered_map(filtered_points, output_file_path)
    print(f"Filtered map saved to {output_file_path}")

if __name__ == "__main__":
    main()

