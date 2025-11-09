# 🗺️ Paris Pathfinder

An interactive educational tool for exploring graph algorithms on Paris's road network. Compare Breadth-First Search (BFS), Depth-First Search (DFS), Dijkstra's algorithm, and A* pathfinding in real-time with visualizations.

## 📋 Overview

This project demonstrates fundamental graph traversal and shortest path algorithms using real-world data from Paris's road network. It consists of three main parts:

1. **Theory Part** (`theory.ipynb`): Educational notebook covering the theoretical foundations of graph algorithms
2. **Exercises** (`exercices.ipynb`): Hands-on exercises to practice implementing and understanding graph algorithms
3. **Streamlit Web App** (`app.py`): Interactive web application for exploring algorithms in real-time with visualizations

### Key Features

- **Four Pathfinding Algorithms**: BFS, DFS, Dijkstra, and A* with side-by-side comparison
- **Real-World Data**: Uses OSMnx to fetch actual Paris road network from OpenStreetMap
- **Interactive Visualization**: Plotly maps showing paths with color-coded algorithms
- **Performance Metrics**: Compare execution time, path length, and node count across algorithms
- **Famous Landmarks**: Pre-configured locations including Tour Eiffel, Louvre, major train stations, and more

## 🚀 Quick Start

### Prerequisites

- Python ≥3.11
- Internet connection (to download road network data)

### Installation

1. Clone or navigate to the project directory:
```bash
cd graph_trip
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using `uv` (if available):
```bash
uv pip install -r requirements.txt
```

### Running the Web App

Launch the Streamlit application:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Jupyter Notebooks

1. **Theory Part**: Start with the theory notebook to understand the algorithms:
```bash
jupyter lab theory.ipynb
```

2. **Exercises**: Practice with hands-on exercises:
```bash
jupyter lab exercices.ipynb
```

## 🎯 How It Works

1. **Select Points**: Choose departure and arrival points from famous Paris landmarks
2. **Calculate Paths**: The app automatically computes paths using all four algorithms
3. **Compare Results**: View side-by-side comparison of:
   - Path distances (in kilometers)
   - Execution time (in milliseconds)
   - Number of nodes traversed
   - Estimated travel time
4. **Visualize**: See all paths overlaid on an interactive map with distinct colors

## 📊 Algorithm Comparison

| Algorithm | Optimality | Best For | Color on Map |
|-----------|------------|----------|--------------|
| **Dijkstra** | ✅ Optimal (distance) | Weighted graphs, guaranteed shortest path | 🔵 Blue |
| **A*** | ✅ Optimal (with admissible heuristic) | Fast pathfinding with geographic data | 🟢 Green |
| **BFS** | ✅ Optimal (unweighted, fewest nodes) | Unweighted graphs, fewest hops | 🟠 Orange |
| **DFS** | ⚠️ Not optimal | Exploration, tree traversal | 🟣 Purple |

## 🏗️ Project Structure

```
graph_trip/
├── app.py                           # Streamlit web application
├── theory.ipynb      # Theory notebook
├── exercices.ipynb                  # Exercises notebook
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project configuration
└── cache/                           # Cached road network data
```

## 💡 Educational Insights

- **BFS vs DFS**: Understand why BFS guarantees the shortest path in unweighted graphs while DFS explores deeply but not optimally
- **Dijkstra's Algorithm**: Learn how it finds the shortest path in weighted graphs
- **A* Heuristic**: Discover how heuristics can speed up pathfinding while maintaining optimality
- **Real-World Application**: See how graph algorithms power navigation systems

## 🔧 Technologies

- **OSMnx**: Road network data from OpenStreetMap
- **NetworkX**: Graph algorithms and data structures
- **Streamlit**: Interactive web interface
- **Plotly**: Interactive map visualizations
- **NumPy**: Numerical computations

## 📝 Notes

- The first run downloads Paris road network data (cached for subsequent runs)
- Large graphs are automatically sampled for visualization performance
- All paths are computed on the full network, only visualization is optimized

---

*Educational project for understanding graph traversal and shortest path algorithms*

