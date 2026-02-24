#!/usr/bin/env python3
"""
Generate a GMSH 2.2 ASCII format structured hex mesh for the
lid-driven cavity (100 x 100 x 1 cells, unit square extruded 0.01 in z).

Boundary physical groups
------------------------
  tag 1  "top_lid"  — y = 1  (moving lid)
  tag 2  "no_slip"  — x = 0, x = 1, y = 0  (fixed walls)
  tag 3  "symmetry" — z = 0, z = dz  (front/back planes)

Volume physical group
---------------------
  tag 4  "fluid"
"""

import os

# ── parameters ──────────────────────────────────────────────────────────────
N  = 100       # cells per side (x and y)
dz = 0.01      # z-thickness of the pseudo-2D slab

TAG_TOP_LID  = 1
TAG_NO_SLIP  = 2
TAG_SYMMETRY = 3
TAG_FLUID    = 4

# ── helpers ──────────────────────────────────────────────────────────────────
dx = 1.0 / N
dy = 1.0 / N

def nid(i, j, k):
    """1-based node index for grid position (i, j, k)."""
    return k * (N + 1) * (N + 1) + j * (N + 1) + i + 1


# ── nodes ────────────────────────────────────────────────────────────────────
nodes = []
for k in range(2):          # z = 0 (k=0) and z = dz (k=1)
    z = 0.0 if k == 0 else dz
    for j in range(N + 1):
        y = j * dy
        for i in range(N + 1):
            x = i * dx
            nodes.append((nid(i, j, k), x, y, z))

n_nodes = len(nodes)

# ── boundary quads ────────────────────────────────────────────────────────────
top_quads    = []   # y = 1
noslip_quads = []   # x = 0, x = 1, y = 0
sym_quads    = []   # z = 0, z = dz

# Top lid (y = 1, j = N)
for i in range(N):
    top_quads.append((nid(i,   N, 0), nid(i+1, N, 0),
                      nid(i+1, N, 1), nid(i,   N, 1)))

# Bottom wall (y = 0, j = 0)
for i in range(N):
    noslip_quads.append((nid(i,   0, 0), nid(i+1, 0, 0),
                         nid(i+1, 0, 1), nid(i,   0, 1)))

# Left wall (x = 0, i = 0)
for j in range(N):
    noslip_quads.append((nid(0, j,   0), nid(0, j+1, 0),
                         nid(0, j+1, 1), nid(0, j,   1)))

# Right wall (x = 1, i = N)
for j in range(N):
    noslip_quads.append((nid(N, j,   0), nid(N, j+1, 0),
                         nid(N, j+1, 1), nid(N, j,   1)))

# Front symmetry face (z = 0, k = 0)
for j in range(N):
    for i in range(N):
        sym_quads.append((nid(i,   j,   0), nid(i+1, j,   0),
                          nid(i+1, j+1, 0), nid(i,   j+1, 0)))

# Back symmetry face (z = dz, k = 1)
for j in range(N):
    for i in range(N):
        sym_quads.append((nid(i,   j,   1), nid(i+1, j,   1),
                          nid(i+1, j+1, 1), nid(i,   j+1, 1)))

# ── hex elements ──────────────────────────────────────────────────────────────
# GMSH hex type 5 node order:
#   4 bottom nodes (z=z0) then 4 top nodes (z=z1)
#   1:(x0,y0,z0), 2:(x1,y0,z0), 3:(x1,y1,z0), 4:(x0,y1,z0)
#   5:(x0,y0,z1), 6:(x1,y0,z1), 7:(x1,y1,z1), 8:(x0,y1,z1)
hex_elems = []
for j in range(N):
    for i in range(N):
        hex_elems.append((
            nid(i,   j,   0), nid(i+1, j,   0),
            nid(i+1, j+1, 0), nid(i,   j+1, 0),
            nid(i,   j,   1), nid(i+1, j,   1),
            nid(i+1, j+1, 1), nid(i,   j+1, 1),
        ))

n_elems = (len(top_quads) + len(noslip_quads) +
           len(sym_quads) + len(hex_elems))

# ── write GMSH 2.2 ────────────────────────────────────────────────────────────
out_dir  = os.path.join(os.path.dirname(__file__), "cavity", "DATA")
out_path = os.path.join(out_dir, "cavity.msh")
os.makedirs(out_dir, exist_ok=True)

with open(out_path, "w") as f:
    f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

    # Physical group names
    f.write("$PhysicalNames\n4\n")
    f.write(f'2 {TAG_TOP_LID}  "top_lid"\n')
    f.write(f'2 {TAG_NO_SLIP}  "no_slip"\n')
    f.write(f'2 {TAG_SYMMETRY} "symmetry"\n')
    f.write(f'3 {TAG_FLUID}    "fluid"\n')
    f.write("$EndPhysicalNames\n")

    # Nodes
    f.write(f"$Nodes\n{n_nodes}\n")
    for nd in nodes:
        f.write(f"{nd[0]} {nd[1]:.8f} {nd[2]:.8f} {nd[3]:.8f}\n")
    f.write("$EndNodes\n")

    # Elements — quad type 3, hex type 5; 2 tags: (physical, elementary)
    f.write(f"$Elements\n{n_elems}\n")
    eid = 1

    for q in top_quads:
        f.write(f"{eid} 3 2 {TAG_TOP_LID} {TAG_TOP_LID} "
                f"{q[0]} {q[1]} {q[2]} {q[3]}\n")
        eid += 1

    for q in noslip_quads:
        f.write(f"{eid} 3 2 {TAG_NO_SLIP} {TAG_NO_SLIP} "
                f"{q[0]} {q[1]} {q[2]} {q[3]}\n")
        eid += 1

    for q in sym_quads:
        f.write(f"{eid} 3 2 {TAG_SYMMETRY} {TAG_SYMMETRY} "
                f"{q[0]} {q[1]} {q[2]} {q[3]}\n")
        eid += 1

    for h in hex_elems:
        f.write(f"{eid} 5 2 {TAG_FLUID} {TAG_FLUID} "
                f"{h[0]} {h[1]} {h[2]} {h[3]} "
                f"{h[4]} {h[5]} {h[6]} {h[7]}\n")
        eid += 1

    f.write("$EndElements\n")

print(f"Mesh written to: {out_path}")
print(f"  Nodes         : {n_nodes}")
print(f"  Hex cells     : {len(hex_elems)}")
print(f"  Top-lid quads : {len(top_quads)}")
print(f"  No-slip quads : {len(noslip_quads)}")
print(f"  Symmetry quads: {len(sym_quads)}")
print(f"  Total elements: {n_elems}")
