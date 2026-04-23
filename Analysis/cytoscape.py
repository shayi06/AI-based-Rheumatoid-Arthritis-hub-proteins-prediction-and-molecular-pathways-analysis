import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from collections import defaultdict
import community.community_louvain as community_louvain
import os, requests, time

# ================== FILE PATHS ==================
ML_FILE    = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\final_prediction\prediction\predicted_ra_proteins_rf_with_genes.xlsx"
SEED_FILE  = r"D:\abi\abi\pythonProject\Abi 2025\Research\network\seed_retrieving\processed_protein.xlsx"
LINKS_FILE = r"D:\abi\abi\pythonProject\Abi 2025\Research\network\9606.protein.links.v12.0.txt"
INFO_FILE  = r"D:\abi\abi\pythonProject\Abi 2025\Research\network\9606.protein.info.v12.0.txt"
module_path = r"D:\abi\abi\pythonProject\Abi 2025\Research\GNN_RA\DEG analysis\modules"
os.makedirs(module_path, exist_ok=True)

# ================== LOAD + BUILD GRAPH ==================
print("Loading data...")
ml_df     = pd.read_excel(ML_FILE)
seed_data = pd.read_excel(SEED_FILE)
ml_genes  = set(ml_df["Gene_Symbol"].dropna())
seeds     = set(seed_data["preferredName"].dropna())
all_prot  = (ml_genes - seeds) | seeds

links   = pd.read_table(LINKS_FILE, sep=" ")
info    = pd.read_table(INFO_FILE)
id2gene = dict(zip(info["#string_protein_id"], info["preferred_name"]))
links   = links[links["combined_score"] >= 700]

G = nx.Graph()
for _, row in links.iterrows():
    g1, g2 = id2gene.get(row["protein1"]), id2gene.get(row["protein2"])
    if g1 and g2 and g1 in all_prot and g2 in all_prot:
        G.add_edge(g1, g2, weight=row["combined_score"] / 1000)
if not nx.is_connected(G):
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
print(f"  Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")

# ================== LOUVAIN ==================
print("Louvain clustering...")
partitions = community_louvain.best_partition(G, random_state=42)
nx.set_node_attributes(G, partitions, "module")
modules = defaultdict(list)
for node, mod in partitions.items():
    modules[mod].append(node)
sorted_mods = sorted(modules.keys(), key=lambda m: len(modules[m]), reverse=True)
print(f"  {len(modules)} modules")

# ================== DEGREE + HUBS ==================
degree_dict = dict(G.degree())
z_scores, intra_deg = {}, {}
for mod_id, members in modules.items():
    vals = []
    for node in members:
        k = sum(1 for nb in G.neighbors(node) if partitions[nb] == mod_id)
        intra_deg[node] = k
        vals.append(k)
    mu, sig = np.mean(vals), np.std(vals, ddof=1)
    for node in members:
        z_scores[node] = 0 if sig == 0 else (intra_deg[node] - mu) / sig

hubs = []
for node, z in z_scores.items():
    if z >= 1.5:
        nbrs = list(G.neighbors(node))
        deg  = len(nbrs)
        mc   = defaultdict(int)
        for nb in nbrs: mc[partitions[nb]] += 1
        pc = 1 - sum((c/deg)**2 for c in mc.values()) if deg > 0 else 0
        hubs.append({"Gene": node, "Module": partitions[node],
                     "Z-score": round(z,3), "PC": round(pc,3),
                     "Type": "Connector Hub" if pc > 0.5 else "Module Hub"})
hub_df = pd.DataFrame(hubs)
print(f"  {len(hub_df)} hub genes")

# ================== GO ENRICHMENT ==================
print("GO enrichment (STRING API)...")
def get_go_term(gene_list):
    if len(gene_list) < 3:
        return "Small module"
    try:
        r = requests.post("https://string-db.org/api/json/enrichment",
                          data={"identifiers": "\r".join(gene_list),
                                "species": 9606,
                                "caller_identity": "ppi_module_analysis"},
                          timeout=30)
        data = r.json()
        bp = [d for d in data if d.get("category") == "Process"] or data
        if not bp: return "Unknown process"
        bp.sort(key=lambda x: float(x.get("fdr", 1.0)))
        desc = bp[0].get("description", "Unknown process")
        return (desc[:44]+"…") if len(desc) > 44 else desc
    except:
        return "Unknown process"

module_go = {}
for i, mod in enumerate(sorted_mods):
    lbl = get_go_term(modules[mod])
    module_go[mod] = lbl
    print(f"  Module {i+1:2d} ({len(modules[mod]):3d}): {lbl}")
    time.sleep(1.5)

# ================== SAVE TABLES ==================
hub_df.to_excel(os.path.join(module_path, "hub_genes.xlsx"), index=False)
pd.DataFrame([{"Gene": n, "Module": m} for n, m in partitions.items()]
             ).to_excel(os.path.join(module_path, "cytoscape_modules.xlsx"), index=False)
pd.DataFrame([{"Num": i+1, "Module_ID": mod, "Size": len(modules[mod]),
               "GO_BP": module_go[mod]}
              for i, mod in enumerate(sorted_mods)]
             ).to_excel(os.path.join(module_path, "module_go_enrichment.xlsx"), index=False)
nx.write_gml(G, os.path.join(module_path, "ppi_network.gml"))

# ==================== INTRA-MODULE LAYOUT ====================
NODE_SEP = 2.5  # minimum distance between node centres inside a module

def sparse_layout(members, seed=42):
    n   = len(members)
    rng = np.random.default_rng(seed)
    if n == 1:
        return {members[0]: np.zeros(2)}
    if n == 2:
        return {members[0]: np.array([-NODE_SEP/2, 0.]),
                members[1]: np.array([ NODE_SEP/2, 0.])}

    # Grid seed
    cols    = max(1, int(np.ceil(np.sqrt(n))))
    pos_arr = np.zeros((n, 2))
    for idx in range(n):
        r, c = divmod(idx, cols)
        pos_arr[idx] = [c*NODE_SEP + rng.uniform(-0.1,0.1),
                        r*NODE_SEP + rng.uniform(-0.1,0.1)]

    # Light spring (topology hint)
    sub      = G.subgraph(members)
    node_idx = {nd: k for k, nd in enumerate(members)}
    for _ in range(60):
        delta = np.zeros_like(pos_arr)
        for u, v in sub.edges():
            i, j  = node_idx[u], node_idx[v]
            diff  = pos_arr[j] - pos_arr[i]
            dist  = np.linalg.norm(diff) + 1e-9
            force = 0.025 * max(0, dist - NODE_SEP*0.9) * (diff/dist)
            delta[i] += force
            delta[j] -= force
        pos_arr += delta

    # Hard repulsion
    for _ in range(500):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                diff = pos_arr[i] - pos_arr[j]
                dist = np.linalg.norm(diff)
                if dist < NODE_SEP:
                    if dist < 1e-9:
                        diff = rng.standard_normal(2); dist = np.linalg.norm(diff)
                    push = (NODE_SEP - dist)/2 + 1e-6
                    pos_arr[i] += diff/dist*push
                    pos_arr[j] -= diff/dist*push
                    moved = True
        if not moved: break

    pos_arr -= pos_arr.mean(0)
    return {nd: pos_arr[k] for k, nd in enumerate(members)}

print("Computing intra-module layouts...")
local_pos  = {}
mod_radius = {}
for mod, members in modules.items():
    lp = sparse_layout(members)
    local_pos[mod] = lp
    pts = np.array(list(lp.values()))
    mod_radius[mod] = np.linalg.norm(pts - pts.mean(0), axis=1).max() + NODE_SEP * 2.0

# ==================== MODULE CENTRE PLACEMENT ====================
# KEY CHANGE: Use attraction + repulsion forces instead of pure repulsion.
# Pure repulsion sent small/isolated modules flying to infinity.
# Now: inter-connected modules attract each other (spring),
#      all modules repel each other if they'd overlap (hard floor).
# Result: all modules stay close together in a compact cluster,
#         small modules pulled toward the main body, none fly away.

MODULE_GAP = NODE_SEP * 1.8   # ← REDUCED gap (was 4.0) so modules sit closer

# Build meta-graph with edge weights = inter-module connections
meta = nx.Graph()
for mod in modules:
    meta.add_node(mod)
for u, v in G.edges():
    mu, mv = partitions[u], partitions[v]
    if mu != mv:
        w = (meta[mu][mv]["weight"] + 1) if meta.has_edge(mu,mv) else 1
        meta.add_edge(mu, mv, weight=w)

# ---- Seed: compact spring layout, small initial scale ----
total_area = sum(np.pi * r**2 for r in mod_radius.values())
R_init     = np.sqrt(total_area / np.pi) * 1.6   # ← TIGHTER initial spread (was 3.5)

raw   = nx.spring_layout(meta, seed=7, k=1.2, weight="weight", iterations=600)
raw_a = np.array(list(raw.values()))
raw_a -= raw_a.mean(0)
span  = raw_a.std() + 1e-6
centres = {m: (np.array(raw[m]) - raw_a.mean(0)) / span * R_init for m in modules}

mod_list = list(modules.keys())
rng2     = np.random.default_rng(77)

print("Placing module centres (attraction + repulsion)...")
for it in range(3000):
    moved = False

    # -- Attractive spring toward connected neighbours --
    for a, b, data in meta.edges(data=True):
        diff  = centres[b] - centres[a]
        dist  = np.linalg.norm(diff)
        ideal = mod_radius[a] + mod_radius[b] + MODULE_GAP
        if dist > ideal:
            # pull toward ideal separation
            pull   = 0.01 * (dist - ideal)
            centres[a] += diff/dist * pull
            centres[b] -= diff/dist * pull
            moved = True

    # -- Hard repulsion: prevent overlap --
    for i in range(len(mod_list)):
        for j in range(i+1, len(mod_list)):
            a, b  = mod_list[i], mod_list[j]
            diff  = centres[a] - centres[b]
            dist  = np.linalg.norm(diff)
            need  = mod_radius[a] + mod_radius[b] + MODULE_GAP
            if dist < need:
                if dist < 1e-9:
                    diff = rng2.standard_normal(2); dist = np.linalg.norm(diff)
                push = (need - dist)/2 + 1e-6
                centres[a] += diff/dist * push
                centres[b] -= diff/dist * push
                moved = True

    if not moved:
        print(f"  Converged at iteration {it+1}")
        break

# ==================== GLOBAL POSITIONS ====================
pos = {}
for mod, members in modules.items():
    c = centres[mod]
    for node, lp in local_pos[mod].items():
        pos[node] = lp + c

# ==================== COLOURS ====================
PALETTE = [
    "#8B2500","#CC2222","#007B7B","#1A44CC",
    "#555555","#AAAAAA","#C9A800","#7B2D8B",
    "#9966CC","#55AADD","#EE7711","#CC44AA",
    "#FF7700","#228B22","#FF6B9D","#6495ED",
]
mod_color = {mod: PALETTE[i % len(PALETTE)] for i, mod in enumerate(sorted_mods)}

# ==================== PCA ELLIPSE ====================
def draw_ellipse(ax, pts_list, pad, lw=1.5):
    pts = np.array(pts_list)
    cx, cy = pts.mean(0)
    if len(pts) < 3:
        w = max(pts[:,0].ptp()+pad*2, pad*2)
        h = max(pts[:,1].ptp()+pad*2, pad*2)
        ang = 0.
    else:
        cen = pts - [cx, cy]
        ev, evec = np.linalg.eigh(np.cov(cen.T))
        order = np.argsort(ev)[::-1]
        evec  = evec[:, order]
        ang   = np.degrees(np.arctan2(evec[1,0], evec[0,0]))
        proj  = cen @ evec
        w = proj[:,0].ptp() + pad*2
        h = proj[:,1].ptp() + pad*1.8
        w, h = max(w, pad*2), max(h, pad*1.5)
    ax.add_patch(Ellipse((cx,cy), width=w, height=h, angle=ang,
                         fill=False, edgecolor="black", linewidth=lw,
                         linestyle="-", alpha=0.55, zorder=4))
    return cx, cy, max(w,h)/2

# ==================== RENDER AT 600 DPI ====================
print("Rendering figure at 600 DPI...")

fig, ax = plt.subplots(figsize=(10, 10), dpi=600)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Edges
nx.draw_networkx_edges(G, pos, ax=ax,
                       alpha=0.25, edge_color="#888888", width=0.5)

# Nodes
# ==================== NODE SIZES BY DEGREE ====================
degrees = dict(G.degree())
min_deg, max_deg = min(degrees.values()), max(degrees.values())
def scale_node(d, min_s=20, max_s=180):
    if max_deg == min_deg:
        return (min_s + max_s) / 2
    return min_s + (d - min_deg) / (max_deg - min_deg) * (max_s - min_s)

node_sizes = {n: scale_node(degrees[n]) for n in G.nodes()}

# Nodes — sized by degree
for mod, members in modules.items():
    sizes = [node_sizes[n] for n in members]
    nx.draw_networkx_nodes(G.subgraph(members), pos, ax=ax,
                           nodelist=members,
                           node_color=mod_color[mod],
                           node_size=sizes,          # <-- list of sizes
                           alpha=0.93,
                           linewidths=0.25,
                           edgecolors="white")


# Ellipses + badges
EPAD = NODE_SEP * 1.4
for i, mod in enumerate(sorted_mods):
    pts = [pos[n] for n in modules[mod]]
    cx, cy, hm = draw_ellipse(ax, pts, pad=EPAD)
    bx, by = cx, cy + hm + 0.5
    num    = i + 1
    ax.add_patch(plt.Circle((bx, by), radius=2.0,
                             color="black", zorder=15, clip_on=False))
    ax.text(bx, by, str(num),
            fontsize=14, fontweight="bold", color="white",
            ha="center", va="center", zorder=16)

# Hub labels — ALL z-score hubs + top 3 per module
label_nodes = set(h["Gene"] for h in hubs)

bbox_kw = dict(boxstyle="round,pad=0.15", facecolor="white",
               edgecolor="#CCCCCC", alpha=0.90, linewidth=0.35)

from adjustText import adjust_text
# Hub type → box colour
HUB_BOX = {
    "Module Hub":    "#FFF3CD",   # warm amber tint
    "Connector Hub": "#D6EAF8",   # cool blue tint
}
HUB_EDGE = {
    "Module Hub":    "#E6A817",
    "Connector Hub": "#2E86C1",
}

# Build lookup: gene → hub type
hub_type_map = {row["Gene"]: row["Type"] for _, row in hub_df.iterrows()}

texts = []
for node in label_nodes:
    if node not in pos:
        continue
    x, y = pos[node]
    htype = hub_type_map.get(node, "Module Hub")

    txt = ax.text(x, y,
                  node,
                  fontsize=10,
                  fontweight="bold",
                  color="#111111",
                  ha="center", va="center",
                  bbox=dict(boxstyle="round,pad=0.2",
                            facecolor=HUB_BOX[htype],
                            edgecolor=HUB_EDGE[htype],
                            alpha=0.92,
                            linewidth=0.9),
                  zorder=16)
    texts.append(txt)


# 🔥 MAGIC: auto-adjust positions
adjust_text(texts,
            expand_points=(1.2, 1.4),
            expand_text=(1.2, 1.4),
            force_text=0.8,
            force_points=0.5,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.4))

# ==================== LEGEND ====================
# Module colour legend — upper right
module_handles = [
    mpatches.Patch(
        color=mod_color[mod],
        label=f"{i+1}. {module_go.get(mod, 'Module '+str(i+1))}  (n={len(modules[mod])})"
    )
    for i, mod in enumerate(sorted_mods)
]

leg1 = ax.legend(
    handles=module_handles,
    loc="lower left",
    frameon=True,
    framealpha=0.85,
    edgecolor="#CCCCCC",
    fontsize=9,
    title="Modules (GO:BP)",
    title_fontsize=9.5,
)
leg1.get_frame().set_linewidth(0.6)
ax.add_artist(leg1)   # ← keeps leg1 alive when leg2 is added

# Hub type legend — lower left
hub_handles = [
    mpatches.Patch(facecolor="#FFF3CD", edgecolor="#E6A817",
                   linewidth=1.2, label="Module Hub"),
    mpatches.Patch(facecolor="#D6EAF8", edgecolor="#2E86C1",
                   linewidth=1.2, label="Connector Hub"),
]

leg2 = ax.legend(
    handles=hub_handles,
    loc="upper right",
    frameon=True,
    framealpha=0.85,
    edgecolor="#CCCCCC",
    fontsize=9,
    title="Hub Types",
    title_fontsize=9.5,
)
leg2.get_frame().set_linewidth(0.6)

ax.set_title("Protein–Protein Interaction Network — Louvain Community Modules",
             fontsize=15, fontweight="bold", pad=18, color="#111111")

# Save
out = os.path.join(module_path, "ppi_publication_v6.png")
fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white", format="png")
print(f"\n✔  Saved → {out}")

# Confirm pixel size
try:
    from PIL import Image
    img = Image.open(out)
    print(f"  Output size: {img.size[0]} × {img.size[1]} pixels  (600 DPI confirmed)")
except:
    pass
print("✔  ALL DONE")