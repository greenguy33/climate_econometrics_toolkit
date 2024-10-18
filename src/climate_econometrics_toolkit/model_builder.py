import xml.etree.ElementTree as ET
import networkx as nx
import pandas as pd

import climate_econometrics_toolkit.ClimateEconometricsModel as cem
import climate_econometrics_toolkit.climate_econometrics_utils as utils

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def build_model_from_graph(graph, dataset):
	model = cem.ClimateEconometricsModel()
	target_var = [node for node in graph.nodes() if len(list(graph.successors(node))) == 0][0]
	input_nodes = list(graph.predecessors(target_var))

	covars = [node for node in input_nodes if not any(node[0:2] == val for val in utils.supported_effects)]
	fixed_effects = [node.split("(")[1].split(")")[0] for node in input_nodes if node[0:2] == "fe"]
	time_trends = [node.split("(")[1].split(")")[0] + " " + node[2] for node in input_nodes if node[0:2] == "ie"]
	model.covariates = covars
	model.target_var = target_var
	model.model_vars = covars + [target_var]
	model.fixed_effects = fixed_effects
	model.time_trends = time_trends
	model.dataset = dataset.split("/")[-1]

	time_column = None
	if len(model.time_trends) > 0:
		time_column = list(graph.predecessors([node for node in input_nodes if node[0:2] == "ie"][0]))[0]
	model.time_column = time_column

	unused_nodes = [node for node in graph.nodes() if node != model.target_var and node not in input_nodes and node != model.time_column]
	return model, unused_nodes


def parse_model_input(model, dataset):
	from_indices,to_indices = model[0],model[1]
	graph = nx.DiGraph()
	for index in range(len(from_indices)):
		graph.add_edge(from_indices[index], to_indices[index])

	assert nx.is_directed_acyclic_graph(graph), "Graph is cyclical - please remove cycles"
	len_target_vars = len([node for node in graph.nodes() if len(list(graph.successors(node))) == 0])
	assert len_target_vars == 1, f"There must be exactly one target variable: found {len_target_vars}"

	return build_model_from_graph(graph, dataset)


def parse_cxl(filepath):

	file = ET.parse(filepath)
	root = file.getroot()

	fromIds, toIds = [], []
	nodeMap, edgeMap, finalMap = {}, {}, {}

	for child in root.iter("{http://cmap.ihmc.us/xml/cmap/}map"):
		for concept in child.iter("{http://cmap.ihmc.us/xml/cmap/}concept"):
			nodeMap[concept.get("id")] = concept.get("label").strip()
		for link in child.iter("{http://cmap.ihmc.us/xml/cmap/}connection"):
			if link.get("from-id") not in edgeMap:
				edgeMap[link.get("from-id")] = []
			edgeMap[link.get("from-id")].append(link.get("to-id"))
		
	for node in nodeMap:
		if node in edgeMap:
			target_nodes = []
			for edge in edgeMap[node]:
				target_nodes.extend(edgeMap[edge])
			finalMap[nodeMap[node]] = [nodeMap[node] for node in target_nodes]

	graph = nx.DiGraph()
	for source, targets in finalMap.items():
		for target in targets:
			graph.add_edge(source, target)

	assert nx.is_directed_acyclic_graph(graph), "Graph is cyclical - please remove cycles"
	len_target_vars = len([node for node in graph.nodes() if len(list(graph.successors(node))) == 0])
	assert len_target_vars == 1, f"There must be exactly one target variable: found {len_target_vars}"

	return build_model_from_graph(graph, filepath)
