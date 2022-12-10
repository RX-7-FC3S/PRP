# coding: utf-8
import pandas as pd
import numpy as np
import pulp as lp
import networkx as nx
import matplotlib.pyplot as plt
import re
import math
from copy import deepcopy
import time


# 把直角坐标转换成序号
def tto(cols, x, y):
    # switch two dimension index to one
    return x * cols + y


# 把序号转换成直角坐标
def ott(cols, n):
    # switch one dimension index to two
    return int(n / cols), int(n % cols)


def calculate(df):
    # print('-------------------  模型求解  ------------------')
    d = np.array(df)
    n = len(d)

    model = lp.LpProblem('routing_of_packing', sense=lp.LpMinimize)
    routes = []
    for i in range(n):
        for j in range(n):
            routes.append((i, j))

    x = lp.LpVariable.dicts('x', routes, 0, 1, lp.LpBinary)
    u = {i: lp.LpVariable('u_{}'.format(i), cat=lp.LpInteger, lowBound=0, upBound=n - 1) for i in range(n)}
    for i in range(n):
        model += lp.lpSum(x[i, j] for j in range(n)) == 1
    for j in range(n):
        model += lp.lpSum(x[i, j] for i in range(n)) == 1
    for i in range(n):
        for j in range(n):
            if i != 0 and j != 0:
                model += (u[i] - u[j] + n * x[i, j] <= n - 1)
    model += lp.lpSum(d[i, j] * x[i, j] for (i, j) in routes)
    model.solve()

    res = {
        'variables': {
            'x': list(x.name for x in model.variables() if x.name[0] == 'x' and x.value() == 1),
            'u': list({u.name: u.value()} for u in model.variables() if u.name[0] == 'u')
        },
        'objective': lp.value(model.objective)
    }

    # print('模型计算结果', res)
    # print('------------------------------------------------')
    return res


def branch(graph, start, end):
    # graph = {0:{1:1, 13:1}}
    costs = {}  # 记录start到其他所有点的距离
    trace = {start: [start]}  # 记录start到其他所有点的路径

    # 初始化costs
    for key in graph.keys():
        costs[key] = math.inf
    costs[start] = 0

    queue = [start]  # 初始化queue

    while len(queue) != 0:
        head = queue[0]  # 起始节点
        for key in graph[head].keys():  # 遍历起始节点的子节点
            dis = graph[head][key] + costs[head]
            if costs[key] > dis:
                costs[key] = dis
                temp = deepcopy(trace[head])  # 深拷贝
                temp.append(key)
                trace[key] = temp  # key节点的最优路径为起始节点最优路径+key
                queue.append(key)
            else:
                pass

        queue.pop(0)  # 删除原来的起始节点

    return trace.get(end)


def create_nodes(rows, cols, shf_size):
    # print('-------------------  生成节点  ------------------')
    # 创建仓库节点集合
    pos = {}
    for i in range(rows):
        for j in range(cols):
            pos[tto(cols, i, j)] = (i, j)

    # 创建过道和货架节点集合
    psg = []
    bins = []

    for i in pos.keys():
        coord = pos[i]
        if coord[0] not in [0, rows - 1] and coord[1] not in [0, shf_size + 1, cols - 1]:
            bins.append(tto(cols, coord[0], coord[1]))
        else:
            psg.append(tto(cols, coord[0], coord[1]))

    for i in range(66, 71):
        bins.remove(i)
        psg.append(i)

    # print('仓库节点集合', pos)
    # print('过道节点集合', psg)
    # print('储位节点集合', bins)
    # print('------------------------------------------------')

    return pos, psg, bins


def create_destinations(cols, shf_size, bins, num, start):
    # print('------------------  生成目标点  ------------------')
    # 随机生成目标点集合
    des = list(np.random.choice(bins, num, replace=False))
    des.append(start)
    des.sort()

    # 构建目标点邻接矩阵
    df = pd.DataFrame(0, des, des)
    for i in df.index:
        for j in df.columns:
            c0 = ott(cols, i)
            c1 = ott(cols, j)
            x0, y0 = c0[0], c0[1]
            x1, y1 = c1[0], c1[1]
            if x0 == x1 or y0 < shf_size + 1 < y1 or y1 < shf_size + 1 < y0:
                dis = abs(x0 - x1) + abs(y0 - y1)
            else:
                dis = abs(x0 - x1) + abs(y0 - y1) + 2 * min(
                    abs(y - c) for y in [y0, y1] for c in [0, shf_size + 1, cols - 1])
            df[i][j] = dis

    # print('目标点集合', des)
    # print('邻接矩阵\n', df)
    # print('------------------------------------------------')
    return des, df


def create_graph_for_branch(edges):
    nodes = edges.copy()
    for i in edges:
        item = list(i)
        item.reverse()
        nodes.append(tuple(item))

    graph = {}
    for i in nodes:
        graph[i[0]] = {j[1]: 1 for j in nodes if j[0] == i[0]}

    return graph


def create_graph_for_networkx(cols, rows, shf_size, pos, psg, bins, des):
    # print('--------------------  生成图  -------------------')
    # 节点
    nodes = []
    for i in psg + bins:
        if i in des:
            if i == des[0]:
                nodes.append((i, {'color': 'green'}))
            else:
                nodes.append((i, {'color': '#F08300'}))
        elif i in psg:
            nodes.append((i, {'color': '#A1A1A1'}))
        else:
            nodes.append((i, {'color': '#2E75B6'}))
    nodes.sort()

    # 边
    edges = []
    for (i, j) in pos.values():
        if j < cols - 1:
            edges.append((tto(cols, i, j), tto(cols, i, j + 1)))
        if i < rows - 1 and j % (shf_size + 1) == 0:
            edges.append((tto(cols, i, j), tto(cols, i + 1, j)))
    # print('图节点集合', nodes)
    # print('边集合', edges)
    # print('------------------------------------------------')

    return nodes, edges


def draw_base_graph(pos, nodes, edges):
    plt.figure(figsize=(10.8, 9.6), dpi=100, frameon=False)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    node_color = [i for i in nx.get_node_attributes(G, 'color').values()]
    nx.draw(G, pos, with_labels=True, node_size=700, node_color=node_color, width=10, edge_color='#A1A1A1')

    return G


def by_model(des, df, graph_branch):
    relative_sequence = []  # 储存为 nx.edge 的格式： [(d0, d1, {'weight': dis, 'nodes_in_route': [0, 1, 2, ...]}, {...}]
    res = calculate(df)

    x = []
    for i in res['variables']['x']:
        c = re.search(r'\d+,_\d+', i)
        c = tuple(map(int, c.group().replace('_', '').split(',')))
        x.append(c)

    x = order(x)

    for c in x:
        d0 = des[c[0]]
        d1 = des[c[1]]
        # 利用 "分支界定法" branch() 得到实际路径 (路径包含的节点)
        relative_sequence.append((d0, d1, {'weight': int(df[d0][d1]), 'nodes_in_route': branch(graph_branch, d0, d1)}))

    return relative_sequence, res['objective']


def by_sequence(des, df, graph_branch):
    relative_sequence = []  # 储存为 nx.edge 的格式： [(d0, d1, {'weight': dis, 'nodes_in_route': [0, 1, 2, ...]}, {...}]

    x = []
    for i in range(len(des)):
        if i != len(des) - 1:
            x.append((i, i + 1))
        else:
            x.append((i, 0))

    objective = 0
    for c in x:
        d0 = des[c[0]]
        d1 = des[c[1]]
        # 利用 "分支界定法" branch() 得到实际路径 (路径包含的节点)
        relative_sequence.append((d0, d1, {'weight': int(df[d0][d1]), 'nodes_in_route': branch(graph_branch, d0, d1)}))
        objective += int(df[d0][d1])

    return relative_sequence, objective


def by_nearest_node(des, df, graph_branch):

    relative_sequence = []  # 储存为 nx.edge 的格式： [(d0, d1, {'weight': dis, 'nodes_in_route': [0, 1, 2, ...]}, {...}]

    nearestPoints = []
    des_Copy = des.copy()
    df_Copy = df.copy()
    minIndex = 0
    minPoint = des[minIndex]

    while len(des_Copy) != 1:
        des_Copy.remove(minPoint)
        # 获取一列
        currentList = df_Copy[minPoint].values
        indexP = np.where(currentList == 0)
        currentList[indexP] = 20000

        minIndex = currentList.argmin()
        indexP = np.where(currentList != 20000)
        currentList[indexP] = 20000

        # 获取一行
        currentList = df_Copy.loc[minPoint].values
        indexP = np.where(currentList != 20000)
        currentList[indexP] = 20000

        minPoint = des[minIndex]
        nearestPoints.append(minPoint)
    nearestPoints.insert(0, des[0])

    des = nearestPoints

    x = []

    for i in range(len(des)):
        if i != len(des) - 1:
            x.append((i, i + 1))
        else:
            x.append((i, 0))

    objective = 0
    for c in x:
        d0 = des[c[0]]
        d1 = des[c[1]]
        # 利用 "分支界定法" branch() 得到实际路径 (路径包含的节点)
        relative_sequence.append((d0, d1, {'weight': int(df[d0][d1]), 'nodes_in_route': branch(graph_branch, d0, d1)}))
        objective += int(df[d0][d1])

    return relative_sequence, objective


def order(original):
    ordered = []
    tail = int
    while original:
        for i in original:
            head = i[0]
            if head == 0:
                ordered.append(i)
                tail = i[-1]
                original.remove(i)
            if head == tail:
                ordered.append(i)
                tail = i[-1]
                original.remove(i)
    return ordered


def clean_dir(path):
    import os
    ls = os.listdir(path)
    for i in ls:
        os.remove(os.path.join(path, i))


def main(rows, cols, shf_size, start, num, opponent):

    nodes = create_nodes(rows, cols, shf_size)
    pos = nodes[0]
    psg = nodes[1]
    bins = nodes[2]

    destinations = create_destinations(cols, shf_size, bins, num, start)
    des = destinations[0]
    df = destinations[1]

    graph_networkx = create_graph_for_networkx(cols, rows, shf_size, pos, psg, bins, des)
    nodes = graph_networkx[0]
    edges = graph_networkx[1]

    graph_branch = create_graph_for_branch(edges)

    by_model_res = by_model(des, df, graph_branch)
    # relative_sequence = by_model_res[0]
    # G = draw_base_graph(pos, nodes, edges)
    # nx.draw_networkx_edges(G, pos, relative_sequence, 10, 'red')
    # plt.savefig('p/model{}.png'.format(num))

    if opponent == 0:
        by_sequence_res = by_sequence(des, df, graph_branch)
        # relative_sequence = sequence_res[0]
        # G = draw_base_graph(pos, nodes, edges)
        # nx.draw_networkx_edges(G, pos, relative_sequence, 10, 'red')
        # plt.savefig('p/sequence{}.png'.format(num))
        s = by_sequence_res[1]
    else:
        by_nearest_node_res = by_nearest_node(des, df, graph_branch)
        # print('by_nearest_node_objective', by_nearest_node_res[1])
        # relative_sequence = by_nearest_node_res[0]
        # G = draw_base_graph(pos, nodes, edges)
        # nx.draw_networkx_edges(G, pos, relative_sequence, 10, 'red')
        # plt.savefig('p/nearest_node{}.png'.format(num))
        s = by_nearest_node_res[1]

    m = by_model_res[1]
    rate = (m - s) / s * 100

    return rate