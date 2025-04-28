import xml.etree.ElementTree as ET
import networkx as nx
import pandas as pd

import climate_econometrics_toolkit.ClimateEconometricsModel as cem
import climate_econometrics_toolkit.utils as utils

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# TODO: it would be better to refactor the model to have a "transformations" dict and not have the transformations represented in the name of the covariates/target var
def build_model_from_graph(graph, data_file, panel_column, time_column):
	model = cem.ClimateEconometricsModel()
	target_var = [node for node in graph.nodes() if len(list(graph.successors(node))) == 0][0]
	input_nodes = list(graph.predecessors(target_var))

	function_split = [node.split("(") for node in input_nodes]
	covars = [node for index, node in enumerate(input_nodes) if not any(function_split[index][0] == val for val in utils.supported_effects)]
	fixed_effects = [node.split("(")[1].split(")")[0] for node in input_nodes if node[0:3] == "fe("]
	time_trends = [node.split("(")[1].split(")")[0] + " " + node[2] for node in input_nodes if node[0:2] == "tt" and node[3] == "("]
	random_effects = [node.split("(")[1].split(")")[0] for node in input_nodes if node[0:2] == "re" and node[2] == "("]
	model.covariates = covars
	model.target_var = target_var
	model.model_vars = covars + [target_var]
	model.fixed_effects = fixed_effects
	if len(random_effects) > 0:
		model.random_effects = [random_effects[0],panel_column]
	model.time_trends = time_trends
	model.data_file = data_file.split("/")[-1]
	model.full_data_path = data_file
	model.time_column = time_column
	model.panel_column = panel_column

	unused_nodes = [node for node in graph.nodes() if node != model.target_var and node not in input_nodes]
	return model, unused_nodes


def parse_model_input(model, data_file, panel_column, time_column):
	from_indices,to_indices = model[0],model[1]
	graph = nx.DiGraph()
	for index in range(len(from_indices)):
		graph.add_edge(from_indices[index], to_indices[index])

	utils.assert_with_log(nx.is_directed_acyclic_graph(graph), "Cyclical graphs are not permitted.")
	len_target_vars = len([node for node in graph.nodes() if len(list(graph.successors(node))) == 0])
	utils.assert_with_log(len_target_vars == 1, f"There must be exactly one target variable: found {len_target_vars}")

	return build_model_from_graph(graph, data_file, panel_column, time_column)
