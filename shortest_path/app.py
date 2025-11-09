import streamlit as st
import osmnx as ox
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import time
from collections import deque

# Configuration de la page
st.set_page_config(
    page_title="Paris Pathfinder",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache pour le graphe (évite de le recharger à chaque fois)
@st.cache_data
def load_graph():
    """Charge le graphe du réseau routier de Paris"""
    with st.spinner("Chargement du réseau routier de Paris... (cela peut prendre quelques secondes)"):
        G = ox.graph_from_place("Paris, France", network_type="drive")
        G = G.to_undirected()
        
        # S'assurer que le graphe est connexe
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        
        return G

def get_weight_attribute(_G):
    """Détermine l'attribut de poids à utiliser"""
    if len(_G.edges()) > 0:
        sample_edge = list(_G.edges(data=True))[0]
        if 'length' in sample_edge[2]:
            return 'length'
    return None

def find_closest_node(G, lon, lat):
    """Trouve le nœud le plus proche d'une coordonnée géographique"""
    min_dist = float('inf')
    closest_node = None
    
    for node in G.nodes():
        node_lon = G.nodes[node].get('x', G.nodes[node].get('lon', None))
        node_lat = G.nodes[node].get('y', G.nodes[node].get('lat', None))
        
        if node_lon is not None and node_lat is not None and \
           isinstance(node_lon, (int, float)) and isinstance(node_lat, (int, float)):
            dist = np.sqrt((node_lon - lon)**2 + (node_lat - lat)**2)
            if dist < min_dist:
                min_dist = dist
                closest_node = node
                closest_coords = (node_lon, node_lat)
    
    return closest_node, min_dist * 111000  # Convertir en mètres approximatifs

def heuristic(u, v, G):
    """Heuristique pour A* : distance euclidienne entre deux nœuds"""
    try:
        if 'x' in G.nodes[u] and 'y' in G.nodes[u]:
            x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
            x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
        elif 'lon' in G.nodes[u] and 'lat' in G.nodes[u]:
            x1, y1 = G.nodes[u]['lon'], G.nodes[u]['lat']
            x2, y2 = G.nodes[v]['lon'], G.nodes[v]['lat']
        else:
            return 0
        
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    except:
        return 0

def calculate_path_bfs(G, start, end):
    """Calcule le chemin avec BFS"""
    visited = set()
    queue = deque([start])
    visited.add(start)
    parent = {start: None}
    
    while queue:
        node = queue.popleft()
        
        if node == end:
            path = []
            current = end
            while current is not None:
                path.append(current)
                current = parent[current]
            return list(reversed(path))
        
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                parent[neighbor] = node
    
    return None

def calculate_path_dfs(G, start, end):
    """Calcule le chemin avec DFS"""
    visited = set()
    stack = [start]
    parent = {start: None}
    
    while stack:
        node = stack.pop()
        
        if node not in visited:
            visited.add(node)
            
            if node == end:
                path = []
                current = end
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return list(reversed(path))
            
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    stack.append(neighbor)
                    if neighbor not in parent:
                        parent[neighbor] = node
    
    return None

def calculate_path(G, start, end, algorithm, weight):
    """Calcule le chemin selon l'algorithme choisi"""
    if algorithm == "BFS":
        return calculate_path_bfs(G, start, end)
    elif algorithm == "DFS":
        return calculate_path_dfs(G, start, end)
    elif algorithm == "Dijkstra":
        if weight:
            return nx.shortest_path(G, source=start, target=end, weight=weight, method='dijkstra')
        else:
            return nx.shortest_path(G, source=start, target=end)
    elif algorithm == "A*":
        if weight:
            h = lambda u, v: heuristic(u, v, G)
            return nx.astar_path(G, source=start, target=end, weight=weight, heuristic=h)
        else:
            h = lambda u, v: heuristic(u, v, G)
            return nx.astar_path(G, source=start, target=end, heuristic=h)
    return None

def get_path_length(G, path, weight):
    """Calcule la longueur totale du chemin"""
    if not path or len(path) < 2:
        return 0
    
    total = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if weight:
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                # Pour MultiGraph, prendre la première clé
                first_key = list(edge_data.keys())[0]
                total += edge_data[first_key].get(weight, 1.0)
        else:
            total += 1
    
    return total

def plot_all_paths(G, paths_dict, start_node, end_node, start_coords, end_coords, title):
    """Crée une visualisation Plotly avec tous les chemins de différents algorithmes"""
    # Collecter tous les nœuds de tous les chemins
    all_path_nodes = set()
    for path in paths_dict.values():
        if path:
            all_path_nodes.update(path)
            # Ajouter quelques voisins pour le contexte
            for node in path[::5]:  # Prendre 1 nœud sur 5
                for neighbor in list(G.neighbors(node))[:3]:
                    all_path_nodes.add(neighbor)
    
    if not all_path_nodes:
        return None
    
    G_path = G.subgraph(list(all_path_nodes)[:1500]).copy()
    
    # Récupérer les coordonnées
    pos = {}
    for node in G_path.nodes():
        if 'x' in G_path.nodes[node] and 'y' in G_path.nodes[node]:
            pos[node] = (G_path.nodes[node]['x'], G_path.nodes[node]['y'])
        elif 'lon' in G_path.nodes[node] and 'lat' in G_path.nodes[node]:
            pos[node] = (G_path.nodes[node]['lon'], G_path.nodes[node]['lat'])
    
    if not pos:
        return None
    
    # Calculer le centre basé sur les points de départ et d'arrivée
    center_lon = (start_coords[0] + end_coords[0]) / 2
    center_lat = (start_coords[1] + end_coords[1]) / 2
    
    # Ajuster le zoom pour mieux voir Paris (calculer la distance entre les points)
    distance = np.sqrt((end_coords[0] - start_coords[0])**2 + (end_coords[1] - start_coords[1])**2)
    # Ajuster le zoom basé sur la distance
    if distance > 0.05:  # Grande distance
        zoom_level = 11
    elif distance > 0.02:  # Distance moyenne
        zoom_level = 12
    else:  # Petite distance
        zoom_level = 13
    
    fig = go.Figure()
    
    # Arêtes du réseau (légères)
    edge_lons, edge_lats = [], []
    for edge in G_path.edges():
        if edge[0] in pos and edge[1] in pos:
            lon0, lat0 = pos[edge[0]]
            lon1, lat1 = pos[edge[1]]
            if (isinstance(lon0, (int, float)) and isinstance(lat0, (int, float)) and
                isinstance(lon1, (int, float)) and isinstance(lat1, (int, float))):
                edge_lons.extend([lon0, lon1, None])
                edge_lats.extend([lat0, lat1, None])
    
    if edge_lons:
        fig.add_trace(go.Scattermap(
            mode='lines',
            lon=edge_lons,
            lat=edge_lats,
            line=dict(width=0.3, color='lightgray'),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Couleurs pour chaque algorithme
    algorithm_colors = {
        'Dijkstra': 'blue',
        'A*': 'green',
        'BFS': 'orange',
        'DFS': 'purple'
    }
    
    # Ajouter chaque chemin avec une couleur différente
    for algo_name, path in paths_dict.items():
        if path:
            path_lons = [pos[node][0] for node in path if node in pos and isinstance(pos[node][0], (int, float))]
            path_lats = [pos[node][1] for node in path if node in pos and isinstance(pos[node][1], (int, float))]
            
            if path_lons and path_lats:
                color = algorithm_colors.get(algo_name, 'red')
                fig.add_trace(go.Scattermap(
                    mode='lines+markers',
                    lon=path_lons,
                    lat=path_lats,
                    line=dict(width=4, color=color),
                    marker=dict(size=6, color=color, opacity=0.7),
                    name=algo_name,
                    showlegend=True,
                    hovertemplate=f'<b>{algo_name}</b><extra></extra>'
                ))
    
    # Points de départ et d'arrivée
    fig.add_trace(go.Scattermap(
        mode='markers',
        lon=[start_coords[0]],
        lat=[start_coords[1]],
        marker=dict(size=22, color='green', symbol='circle'),
        name='Départ',
        showlegend=True,
        hovertemplate=f'<b>Départ: {title.split(" à ")[0] if " à " in title else "Départ"}</b><br>Lat: {start_coords[1]:.4f}<br>Lon: {start_coords[0]:.4f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scattermap(
        mode='markers',
        lon=[end_coords[0]],
        lat=[end_coords[1]],
        marker=dict(size=22, color='red', symbol='circle'),
        name='Arrivée',
        showlegend=True,
        hovertemplate=f'<b>Arrivée: {title.split(" à ")[-1] if " à " in title else "Arrivée"}</b><br>Lat: {end_coords[1]:.4f}<br>Lon: {end_coords[0]:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        mapbox=dict(
            style="open-street-map",
            center=dict(lon=center_lon, lat=center_lat),
            zoom=zoom_level,
            bearing=0,
            pitch=0
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=700,
        showlegend=True
    )
    
    return fig

# Points connus à Paris
KNOWN_PLACES = {
    'Tour Eiffel': (2.2945, 48.8584),
    'Arc de Triomphe': (2.2950, 48.8738),
    'Notre-Dame de Paris': (2.3499, 48.8530),
    'Louvre': (2.3364, 48.8606),
    'Sacré-Cœur': (2.3431, 48.8867),
    'Gare du Nord': (2.3553, 48.8809),
    'Gare de Lyon': (2.3733, 48.8447),
    'Gare Montparnasse': (2.3217, 48.8412),
    'Gare Saint-Lazare': (2.3260, 48.8767),
    'Place de la Bastille': (2.3697, 48.8532),
    'Place de la République': (2.3630, 48.8676),
    'Panthéon': (2.3463, 48.8462),
    'Rosa Parks (Station)': (2.3683, 48.8967),  # Station de métro Rosa Parks
    'Place de la Concorde': (2.3212, 48.8656),
    'Champs-Élysées': (2.3050, 48.8698),
}

# Interface Streamlit
st.title("🗺️ Paris Pathfinder")
st.markdown("### Calcul du plus court chemin sur le réseau routier de Paris")

# Sidebar pour les paramètres
with st.sidebar:
    st.header("⚙️ Paramètres")
    
    # Sélection du point de départ
    st.subheader("📍 Point de départ")
    start_place = st.selectbox(
        "Choisir un lieu",
        options=list(KNOWN_PLACES.keys()),
        key="start"
    )
    start_coords = KNOWN_PLACES[start_place]
    
    # Sélection du point d'arrivée
    st.subheader("🎯 Point d'arrivée")
    end_place = st.selectbox(
        "Choisir un lieu",
        options=list(KNOWN_PLACES.keys()),
        key="end",
        index=1 if len(KNOWN_PLACES) > 1 else 0
    )
    end_coords = KNOWN_PLACES[end_place]
    
    st.markdown("---")
    
    if st.button("🚀 Calculer les chemins", type="primary"):
        st.session_state.calculate = True

# Charger le graphe
G = load_graph()
weight = get_weight_attribute(G)

# Si on doit calculer
if st.session_state.get('calculate', False):
    with st.spinner("Recherche des nœuds les plus proches..."):
        start_node, start_dist = find_closest_node(G, start_coords[0], start_coords[1])
        end_node, end_dist = find_closest_node(G, end_coords[0], end_coords[1])
    
    if start_node is None or end_node is None:
        st.error("Impossible de trouver des nœuds proches des coordonnées sélectionnées.")
    else:
        st.success(f"✅ Nœud de départ trouvé à {start_dist:.0f}m du lieu sélectionné")
        st.success(f"✅ Nœud d'arrivée trouvé à {end_dist:.0f}m du lieu sélectionné")
        
        # Calcul de tous les chemins avec tous les algorithmes
        algorithms = ['Dijkstra', 'A*', 'BFS', 'DFS']
        paths_dict = {}
        results_dict = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, algo in enumerate(algorithms):
            status_text.text(f"Calcul avec {algo}...")
            try:
                start_time = time.time()
                path = calculate_path(G, start_node, end_node, algo, weight)
                elapsed_time = time.time() - start_time
                
                if path:
                    path_length = get_path_length(G, path, weight)
                    num_nodes = len(path)
                    
                    paths_dict[algo] = path
                    results_dict[algo] = {
                        'path': path,
                        'distance': path_length,
                        'num_nodes': num_nodes,
                        'time': elapsed_time,
                        'success': True
                    }
                    
                    if weight:
                        estimated_time = path_length / (50/3.6)  # Temps estimé à 50 km/h
                        results_dict[algo]['estimated_time'] = estimated_time
                else:
                    results_dict[algo] = {'success': False, 'error': 'Aucun chemin trouvé'}
            except Exception as e:
                results_dict[algo] = {'success': False, 'error': str(e)}
            
            progress_bar.progress((i + 1) / len(algorithms))
        
        status_text.text("Calcul terminé!")
        progress_bar.empty()
        status_text.empty()
        
        # Affichage des résultats
        st.markdown("---")
        st.header("📊 Comparaison des Algorithmes")
        
        # Tableau de comparaison
        comparison_data = []
        for algo in algorithms:
            if algo in results_dict and results_dict[algo].get('success'):
                result = results_dict[algo]
                comparison_data.append({
                    'Algorithme': algo,
                    'Distance (km)': f"{result['distance']/1000:.2f}" if weight else "N/A",
                    'Nœuds': result['num_nodes'],
                    'Temps (ms)': f"{result['time']*1000:.2f}",
                    'Temps estimé (min)': f"{result.get('estimated_time', 0)/60:.1f}" if weight and 'estimated_time' in result else "N/A"
                })
            else:
                error_msg = results_dict.get(algo, {}).get('error', 'Erreur inconnue')
                comparison_data.append({
                    'Algorithme': algo,
                    'Distance (km)': "Erreur",
                    'Nœuds': "N/A",
                    'Temps (ms)': "N/A",
                    'Temps estimé (min)': "N/A"
                })
        
        st.dataframe(comparison_data, use_container_width=True, hide_index=True)
        
        # Visualisation avec tous les chemins
        st.subheader("🗺️ Visualisation des chemins")
        
        fig = plot_all_paths(
            G,
            paths_dict,
            start_node,
            end_node,
            start_coords,
            end_coords,
            f"Chemins de {start_place} à {end_place}"
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Insights détaillés
        st.subheader("💡 Insights par Algorithme")
        
        cols = st.columns(2)
        
        for idx, algo in enumerate(algorithms):
            col = cols[idx % 2]
            
            with col:
                if algo in results_dict and results_dict[algo].get('success'):
                    result = results_dict[algo]
                    
                    # Couleurs selon l'algorithme
                    if algo == 'Dijkstra':
                        st.success(f"**{algo}** ✅")
                        st.write(f"- Distance: {result['distance']/1000:.2f} km" if weight else f"- Nœuds: {result['num_nodes']}")
                        st.write(f"- Temps calcul: {result['time']*1000:.2f} ms")
                        st.write("- ✅ **Optimal en distance** (garanti)")
                    elif algo == 'A*':
                        st.success(f"**{algo}** ✅")
                        st.write(f"- Distance: {result['distance']/1000:.2f} km" if weight else f"- Nœuds: {result['num_nodes']}")
                        st.write(f"- Temps calcul: {result['time']*1000:.2f} ms")
                        st.write("- ✅ **Optimal en distance** (heuristique)")
                    elif algo == 'BFS':
                        st.info(f"**{algo}** ⚠️")
                        st.write(f"- Nœuds: {result['num_nodes']}")
                        st.write(f"- Distance: {result['distance']/1000:.2f} km" if weight else "N/A")
                        st.write(f"- Temps calcul: {result['time']*1000:.2f} ms")
                        st.write("- ⚠️ **Optimal en nombre de nœuds** (pas forcément en distance)")
                    elif algo == 'DFS':
                        st.warning(f"**{algo}** ⚠️")
                        st.write(f"- Nœuds: {result['num_nodes']}")
                        st.write(f"- Distance: {result['distance']/1000:.2f} km" if weight else "N/A")
                        st.write(f"- Temps calcul: {result['time']*1000:.2f} ms")
                        st.write("- ⚠️ **Non optimal** (exploration en profondeur)")
                else:
                    error = results_dict.get(algo, {}).get('error', 'Erreur inconnue')
                    st.error(f"**{algo}** ❌")
                    st.write(f"Erreur: {error}")
        
        # Analyse comparative
        if weight:
            successful_results = {k: v for k, v in results_dict.items() if v.get('success')}
            if len(successful_results) > 1:
                st.subheader("📈 Analyse Comparative")
                
                best_distance = min((r['distance'], algo) for algo, r in successful_results.items())
                fastest = min((r['time'], algo) for algo, r in successful_results.items())
                
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    st.metric("🏆 Plus court chemin", f"{best_distance[1]} ({best_distance[0]/1000:.2f} km)")
                
                with comp_col2:
                    st.metric("⚡ Plus rapide", f"{fastest[1]} ({fastest[0]*1000:.2f} ms)")
        
        # Détails des chemins
        with st.expander("🔍 Détails des chemins"):
            for algo in algorithms:
                if algo in paths_dict:
                    path = paths_dict[algo]
                    result = results_dict[algo]
                    st.write(f"**{algo}:**")
                    st.write(f"- Distance: {result['distance']/1000:.2f} km" if weight else f"- Nœuds: {result['num_nodes']}")
                    st.write(f"- Nombre de nœuds: {result['num_nodes']}")
                    st.write(f"- Temps de calcul: {result['time']*1000:.2f} ms")
                    if st.checkbox(f"Afficher les nœuds ({algo})", key=f"show_nodes_{algo}"):
                        st.code(", ".join(map(str, path[:50])))
                        if len(path) > 50:
                            st.caption(f"... et {len(path) - 50} autres nœuds")
                    st.markdown("---")
    
    st.session_state.calculate = False

else:
    # Écran d'accueil
    st.info("👈 Utilisez la barre latérale pour choisir vos points de départ et d'arrivée, puis cliquez sur 'Calculer les chemins'.")
    
    st.markdown("""
    ### 🔍 Fonctionnement
    
    L'application calcule automatiquement le chemin avec **tous les algorithmes** disponibles et les compare :
    
    - **Dijkstra** 🔵 (Bleu): Trouve le plus court chemin en distance sur un graphe pondéré. Optimal et garanti.
    - **A*** 🟢 (Vert): Amélioration de Dijkstra avec heuristique. Plus rapide tout en restant optimal.
    - **BFS** 🟠 (Orange): Trouve le plus court chemin en nombre de nœuds (pour graphes non pondérés).
    - **DFS** 🟣 (Violet): Parcours en profondeur. Ne garantit pas l'optimalité mais utile pour l'exploration.
    
    Tous les chemins sont affichés sur la carte avec des couleurs différentes pour faciliter la comparaison visuelle.
    
    ### 🎯 Points d'intérêt disponibles
    
    L'application inclut plusieurs monuments et lieux emblématiques de Paris, dont :
    - Tour Eiffel
    - Arc de Triomphe
    - Notre-Dame
    - Louvre
    - Gares principales
    - Stations de métro (ex: Rosa Parks)
    """)

