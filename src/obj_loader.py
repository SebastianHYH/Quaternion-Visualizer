import numpy as np

def load_obj(filename):
    vertices = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('v '):  # Vertex line
                    parts = line.strip().split()
                    vertex = list(map(float, parts[1:4]))  # Get x, y, z
                    vertices.append(vertex)
    except Exception as e:
        print(f"Error loading OBJ file: {e}")
        return np.array([])
    
    if not vertices:
        print("Tidak ada vertex ditemukan dalam file OBJ.")
        
    return np.array(vertices)
