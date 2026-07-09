import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

def generate_graph_arrays(levels=3):
    nodes_data = []
    edges_raw = []
    node_id = 0

    spine_length = 6

    # 1. Geração da Estrutura Base
    spine_nodes = []
    for i in range(spine_length):
        nodes_data.append({'x': i * 4.0, 'y': 0.0})
        spine_nodes.append(node_id)
        if i > 0: edges_raw.append((node_id - 1, node_id))
        node_id += 1

    def add_fractal_branches(parent_id, px, py, angle, length, depth):
        nonlocal node_id
        if depth == 0: return
        angles = [angle + np.pi/6, angle - np.pi/6]
        branch_len = length * 0.75
        for a in angles:
            nx = px + branch_len * np.cos(a)
            ny = py + branch_len * np.sin(a)
            curr_id = node_id
            nodes_data.append({'x': nx, 'y': ny})
            edges_raw.append((parent_id, curr_id))
            node_id += 1
            add_fractal_branches(curr_id, nx, ny, a, branch_len, depth - 1)

    for s_id in spine_nodes[1:-1]:
        add_fractal_branches(s_id, nodes_data[s_id]['x'], nodes_data[s_id]['y'], np.pi/2, 3.0, levels)
        add_fractal_branches(s_id, nodes_data[s_id]['x'], nodes_data[s_id]['y'], -np.pi/2, 3.0, levels)

    # 2. Adição das Coletoras (Manifolds)
    df_temp = pd.DataFrame(nodes_data)
    y_max, y_min = df_temp['y'].max() + 1.0, df_temp['y'].min() - 1.0

    all_indices = [e[0] for e in edges_raw] + [e[1] for e in edges_raw]
    counts = pd.Series(all_indices).value_counts()
    leaf_ids = counts[counts == 1].index.tolist()
    leaf_ids = [idx for idx in leaf_ids if idx not in [spine_nodes[0], spine_nodes[-1]]]

    for l_id in leaf_ids:
        target_y = y_max if nodes_data[l_id]['y'] > 0 else y_min
        new_id = node_id
        nodes_data.append({'x': nodes_data[l_id]['x'], 'y': target_y})
        edges_raw.append((l_id, new_id))
        node_id += 1

    # 3. Processamento Geométrico de Interseções
    lines = [LineString([(nodes_data[e[0]]['x'], nodes_data[e[0]]['y']),
                         (nodes_data[e[1]]['x'], nodes_data[e[1]]['y'])]) for e in edges_raw]

    # Adicionar linhas das coletoras para o merge
    df_nodes_final = pd.DataFrame(nodes_data)
    for y_lim in [y_max, y_min]:
        pts = df_nodes_final[df_nodes_final['y'] == y_lim].sort_values('x')
        if len(pts) > 1:
            lines.append(LineString(pts[['x', 'y']].values))

    # Quebra todas as linhas nas interseções
    merged_graph = unary_union(lines)

    # 4. Conversão para Arrays NumPy (Mapeamento de IDs)
    final_nodes_map = {}
    final_nodes_list = []
    final_edges_list = []

    def get_node_id(pt):
        # Arredondamento para evitar erros de precisão de ponto flutuante
        coords = (round(pt[0], 6), round(pt[1], 6))
        if coords not in final_nodes_map:
            final_nodes_map[coords] = len(final_nodes_list)
            final_nodes_list.append([pt[0], pt[1]])
        return final_nodes_map[coords]

    # Itera sobre cada segmento gerado pelo unary_union
    segments = merged_graph.geoms if hasattr(merged_graph, 'geoms') else [merged_graph]
    for seg in segments:
        id_start = get_node_id(seg.coords[0])
        id_end = get_node_id(seg.coords[-1])
        final_edges_list.append([id_start, id_end])

    nodes_np = np.array(final_nodes_list)
    edges_np = np.array(final_edges_list)
    mask = edges_np[:, 0] != edges_np[:, 1]
    edges_np = edges_np[mask]

    return nodes_np, edges_np
