import json

f = open("outputs/Template_2L-mem93-resnet18-layer_LayerNode_3_complete.json")

data = json.load(f)
spatial_map = data["inputs"]["spatial_mapping"]["spatial_mapping"]
temporal_map = data["inputs"]["temporal_mapping"]["temporal_mapping"]

loops = []
N = len(temporal_map['O'])

for i in range(N):

	loops += [[*spatial_map['O'][i], 'S']]
	loops += [[*temporal_map['O'][i], 'T']]

print(loops)

print(spatial_map, temporal_map)