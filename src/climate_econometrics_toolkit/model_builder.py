import xml.etree.ElementTree as ET
import networkx as nx
import pandas as pd
import copy
import numpy as np
from statsmodels.tsa.statespace.tools import diff
import statsmodels.api as sm
import pymc as pm
from pytensor import tensor as pt 
import pickle as pkl

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


supported_functions = ["fd","sq","ln"]
supported_effects = ["fe", "ie"]


def parse_cxl(file):

	file = ET.parse(file)
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

	return graph


def add_transformation_to_data(data, function):
	if function[0:2] == "sq":
		data[function] = np.square(data[function[3:-1]])
	elif function[0:2] == "fd":
		data[function] = diff(data[function[3:-1]])
	elif function[0:2] == "ln":
		data[function] = np.log(data[function[3:-1]])
	return data


def add_effect_to_data(effect, node, data):
	if effect == "fe":
		for element in sorted(list(set(data[node])))[1:]:
			data[f"fe_{element}_{node}"] = np.where(data[node] == element, 1, 0)
	elif effect == "ie":
		ie_level = 1
		node_split = node.split(" ")
		if len(node_split) > 1:
			ie_level = int(node_split[1].strip())
		for element in sorted(list(set(data[node_split[0]]))):
			data[f"ie_{element}_{node_split[0]}_1"] = np.where(data[node_split[0]] == element, 1, 0)
			data[f"ie_{element}_{node_split[0]}_1"] = np.where(data[node_split[0]] == element, data[f"ie_{element}_{node_split[0]}_1"].cumsum(), 0)
			for i in range(1, ie_level+1):
				data[f"ie_{element}_{node_split[0]}_{i}"] = np.power(data[f"ie_{element}_{node_split[0]}_1"], i)
	return data


def transform_data(data, graph):
	transformations = []
	for node in graph.nodes():
		if node[0:2] not in supported_functions and node[0:2] not in supported_effects:
			assert node in data, f"Element {node} not found in data"
		elif node[0:2] in supported_functions:
			function_split = node.split("(")
			data_node = function_split[-1].replace(")","")
			assert data_node in data, f"Element {data_node} not found in data"
			for function in reversed(function_split[:-1]):
				assert function in supported_functions, f"Invalid function call {function}"
				transformations.append(function + f"({data_node})")
				data = add_transformation_to_data(data, transformations[-1])
				data_node = transformations[-1]
		else:
			effect_split = [val.strip() for val in node.split(":")]
			assert effect_split[1].split(" ")[0].strip() in data, "Element " + effect_split[1].split(" ")[0].strip() + " not found in data"
			data = add_effect_to_data(effect_split[0], effect_split[1], data)
	return data


def get_covar_list(data, target_var, graph):
	input_nodes = list(graph.predecessors(target_var))
	covars = []
	for node in input_nodes:
		if node in data:
			covars.append(node)
	effect_cols = [col for col in data.columns if any(col.startswith(effect) for effect in supported_effects)]
	covars.extend(effect_cols)
	return covars


def run_standard_regression(data, graph):
	transformed_data = transform_data(data, graph)
	target_var = [node for node in graph.nodes() if len(list(graph.successors(node))) == 0][0]
	covars = get_covar_list(transformed_data, target_var, graph)
	regression_data = transformed_data[covars]
	if len([col for col in covars if col.startswith("fe")]) == 0:
		regression_data = sm.add_constant(regression_data)
	model = sm.OLS(transformed_data[target_var],regression_data,missing="drop")
	regression = model.fit()
	return regression


def run_bayesian_regression(data, graph):
	transformed_data = transform_data(data, graph).dropna(axis=0).reset_index(drop=True)
	target_var = [node for node in graph.nodes() if len(list(graph.successors(node))) == 0][0]
	covars = get_covar_list(transformed_data, target_var, graph)
	with pm.Model() as model:
		covar_coefs = pm.Normal("covar_coefs", 0, 10, shape=(len(covars)))
		regressors = pm.Deterministic("regressors", pt.sum(covar_coefs * transformed_data[covars], axis=1))
		target_scale = pm.HalfNormal("target_scale", 10)
		target_std = pm.HalfNormal("target_std", sigma=target_scale)
		target_posterior = pm.Normal('target_posterior', mu=regressors, sigma=target_std, observed=transformed_data[target_var])

		prior = pm.sample_prior_predictive()
		trace = pm.sample(target_accept=.99, cores=4, tune=1000, draws=1000)
		posterior = pm.sample_posterior_predictive(trace, extend_inferencedata=True)

	with open ('bayes_model.pkl', 'wb') as buff:
		pkl.dump({
			"prior":prior,
			"trace":trace,
			"posterior":posterior,
			"var_list":list(covars),
			"target_var":target_var
		},buff)