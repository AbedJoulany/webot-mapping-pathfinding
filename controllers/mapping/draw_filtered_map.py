import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_map(file_path):
    """Load the map from a CSV file."""
    return pd.read_csv(file_path, header=None).values

def draw_map(points):
    """Draw the map points using matplotlib."""
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], s=10, c='b', label='Filtered Map Points')
    plt.title('Filtered Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def main():
    input_file_path = 'filtered_map.csv'
    points = load_map(input_file_path)
    draw_map(points)

if __name__ == "__main__":
    main()
