import networkx
import numpy
import matplotlib


def initialize_graph(dimension):
    matrix = numpy.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(i + 1, dimension):
            if numpy.random.randint(0, 2) == 1:
                matrix[i][j] = matrix[j][i] = numpy.random.randint(0, 100)
    print(matrix)
    return matrix


def dijkstra(matrix, node):
    distances = numpy.ones(matrix.shape[0]) * -1
    edges = {node: [node]}
    distances[node] = 0
    for _index in range(matrix.shape[0]):
        for i in range(matrix.shape[0]):
            if distances[i] == -1:
                continue
            for j in range(matrix.shape[0]):
                if matrix[i, j] == 0:
                    continue
                new_distance = distances[i] + matrix[i, j]
                if distances[j] == -1 or new_distance < distances[j]:
                    distances[j] = new_distance
                    path = edges[i].copy()
                    path.append(j)
                    edges[j] = path
                    continue
    return edges


def tree(graph):
    matrix = networkx.to_numpy_matrix(graph)
    i = -1
    j = -1
    minimum = -1
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            if matrix[x, y] > 0 and (minimum == -1 or matrix[i, j] < minimum):
                minimum = matrix[i, j]
                i = x
                j = y
    paths = dijkstra(matrix, i)
    edges = set()
    for _key, value in paths.items():
        for i in range(len(value) - 1):
            edges.add((value[i], value[i + 1]))
            edges.add((value[i + 1], value[i]))
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[0]):
            if (x, y) not in edges:
                matrix[x, y] = 0
                matrix[y, x] = 0
    return networkx.from_numpy_matrix(matrix)


def remove_maximum(graph):
    matrix = networkx.to_numpy_matrix(graph)
    i = -1
    j = -1
    max = -1
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            if matrix[x, y] > 0 and (max == -1 or matrix[i, j] > max):
                max = matrix[i, j]
                i = x
                j = y
    matrix[i, j] = 0
    matrix[j, i] = 0
    return networkx.from_numpy_matrix(matrix)


if __name__ == '__main__':
    n = 10

    matrix = initialize_graph(n)
    gr = networkx.from_numpy_matrix(matrix)
    pos = networkx.spring_layout(gr, seed=7)
    networkx.draw_networkx(gr)
    matplotlib.pyplot.show()

    gr = tree(gr)
    pos = networkx.spring_layout(gr, seed=2)
    networkx.draw_networkx_nodes(gr, pos)
    networkx.draw_networkx_edges(gr, pos)
    edge_labels = networkx.get_edge_attributes(gr, "weight")
    networkx.draw_networkx_edge_labels(gr, pos, edge_labels)
    matplotlib.pyplot.show()

    gr = remove_maximum(gr)
    pos = networkx.spring_layout(gr, seed=2)
    networkx.draw_networkx_nodes(gr, pos)
    networkx.draw_networkx_edges(gr, pos)
    edge_labels = networkx.get_edge_attributes(gr, "weight")
    networkx.draw_networkx_edge_labels(gr, pos, edge_labels)
    matplotlib.pyplot.show()
