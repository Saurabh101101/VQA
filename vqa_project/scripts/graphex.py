
import json
import re
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import networkx as nx

# ======== File Paths (Change these as needed) ========
image_path = "D:/WORK/PG/Project/vqa_project/data/visual_genome/VG_100K/2.jpg"
attribute_file = "D:/WORK/PG/Project/Documentation/ex_attribute.txt"
relationship_file = "D:/WORK/PG/Project/Documentation/ex_relationship.txt"

# ======== Step 1: Load Image ========
img = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(img)

# ======== Step 2: Parse Attributes (Bounding Boxes) ========
with open(attribute_file, "r") as f:
    attr_raw = f.read()

attribute_blocks = re.findall(
    r'{\s*"synsets":\s*\[.*?\],\s*"h":\s*(\d+),\s*"object_id":\s*(\d+),\s*"names":\s*\["(.*?)"\],\s*"w":\s*(\d+),\s*(?:"attributes":\s*\[.*?\],)?\s*"y":\s*(\d+),\s*"x":\s*(\d+)',
    attr_raw, re.S
)

objects = []
for block in attribute_blocks:
    h, object_id, name, w, y, x = block
    objects.append({
        'h': int(h),
        'object_id': int(object_id),
        'name': name,
        'w': int(w),
        'y': int(y),
        'x': int(x)
    })

# ======== Step 3: Draw Bounding Boxes with Backgrounded Labels ========
font = ImageFont.load_default()

for obj in objects:
    x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
    name = obj['name']
    
    # Draw thicker rectangle
    draw.rectangle([(x, y), (x + w, y + h)], outline="lime", width=5)
    
    # Draw filled rectangle for label
    text_size = draw.textsize(name, font=font)
    draw.rectangle([(x, y - text_size[1] - 4), (x + text_size[0] + 4, y)], fill="white")
    
    # Draw label text
    draw.text((x + 2, y - text_size[1] - 2), name, fill="red", font=font)

# ======== Step 4: Parse Relationships (Scene Graph) ========
with open(relationship_file, "r") as f:
    rel_raw = f.read()

relationship_blocks = re.findall(
    r'"predicate":\s*"(.*?)",\s*"object":\s*{.*?"name":\s*"(.*?)".*?},\s*"relationship_id":.*?,\s*"synsets":\s*\[.*?\],\s*"subject":\s*{.*?"name":\s*"(.*?)"',
    rel_raw, re.S
)

relationships = []
for pred, obj, subj in relationship_blocks:
    relationships.append({
        'predicate': pred,
        'object': obj,
        'subject': subj
    })

# ======== Step 5: Create Scene Graph ========
G = nx.DiGraph()
for rel in relationships:
    subj = rel['subject']
    obj_ = rel['object']
    pred = rel['predicate']
    G.add_node(subj)
    G.add_node(obj_)
    G.add_edge(subj, obj_, label=pred)

# ======== Step 6: Plotting ========
plt.figure(figsize=(18, 9))

# Left: Image with bounding boxes
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.axis('off')
plt.title('Image with Object Bounding Boxes', fontsize=14)

# Right: Scene graph
plt.subplot(1, 2, 2)
pos = nx.spring_layout(G, k=0.7)
nx.draw(G, pos, with_labels=True, node_size=1800, node_color="skyblue", font_size=8, font_weight="bold", arrows=True)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
plt.title('Scene Graph Representation', fontsize=14)

plt.tight_layout()
plt.show()

# ======== Step 1: Load Image ========
img = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(img)

# ======== Step 2: Parse Attributes (Bounding Boxes) ========
with open(attribute_file, "r") as f:
    attr_raw = f.read()

# Extract attributes manually
attribute_blocks = re.findall(
    r'{\s*"synsets":\s*\[.*?\],\s*"h":\s*(\d+),\s*"object_id":\s*(\d+),\s*"names":\s*\["(.*?)"\],\s*"w":\s*(\d+),\s*(?:"attributes":\s*\[.*?\],)?\s*"y":\s*(\d+),\s*"x":\s*(\d+)',
    attr_raw, re.S
)
objects = []
for block in attribute_blocks:
    h, object_id, name, w, y, x = block
    objects.append({
        'h': int(h),
        'object_id': int(object_id),
        'name': name,
        'w': int(w),
        'y': int(y),
        'x': int(x)
    })

# ======== Step 3: Draw Bounding Boxes ========
for obj in objects:
    x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
    name = obj['name']
    draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)
    draw.text((x, max(0, y - 10)), name, fill="red")

# ======== Step 4: Parse Relationships (Scene Graph) ========
with open(relationship_file, "r") as f:
    rel_raw = f.read()

# Extract relationships manually
relationship_blocks = re.findall(
    r'"predicate":\s*"(.*?)",\s*"object":\s*{.*?"name":\s*"(.*?)".*?},\s*"relationship_id":.*?,\s*"synsets":\s*\[.*?\],\s*"subject":\s*{.*?"name":\s*"(.*?)"',
    rel_raw, re.S
)

relationships = []
for pred, obj, subj in relationship_blocks:
    relationships.append({
        'predicate': pred,
        'object': obj,
        'subject': subj
    })

# ======== Step 5: Create Scene Graph ========
G = nx.DiGraph()
for rel in relationships:
    subj = rel['subject']
    obj_ = rel['object']
    pred = rel['predicate']
    G.add_node(subj)
    G.add_node(obj_)
    G.add_edge(subj, obj_, label=pred)

# ======== Step 6: Plotting ========
plt.figure(figsize=(18, 9))

# Left: Image with bounding boxes
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.axis('off')
plt.title('Image with Object Bounding Boxes', fontsize=14)

# Right: Scene graph
plt.subplot(1, 2, 2)
pos = nx.spring_layout(G, k=0.7)
nx.draw(G, pos, with_labels=True, node_size=1800, node_color="skyblue", font_size=8, font_weight="bold", arrows=True)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
plt.title('Scene Graph Representation', fontsize=14)

plt.tight_layout()
plt.show()
