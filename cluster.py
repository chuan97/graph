import numpy as np


def recursive_cluster(A, checked=None, clusters=None):
    """Returns a list of clusters from adjacency matrix

    Args:
        A (ndarray): The adjacency matrix, an NxN ndarray whose non-zero elements indicate links.
        checked (set, optional): The set of nodes that have been covered so far. Defaults to set().
        clusters (list[set], optional): The current list of clusters. Defaults to [{0}].

    Returns:
        list[set]: The updated list of clusters.
    """
    if checked is None:
        checked = set()

    if clusters is None:
        clusters = [{0}]

    connected_nodes = (
        set()
    )  # empty set, to contain new connections of the current cluster
    for idx in clusters[-1]:  # run over current members of current cluster
        if idx not in checked:  # if the node hasn't been checked before
            connected_nodes.update(
                np.nonzero(A[idx, :])[0]
            )  # add its connections to the list of new connections
            checked.add(idx)  # mark node as checked

    remaining = set(range(A.shape[0])) - checked  # build set of unchecked nodes
    if (
        not remaining
    ):  # if all nodes have been checked, nothing more to do, end of recursive depth
        return clusters  # return current clusters
    if connected_nodes.issubset(
        clusters[-1]
    ):  # if the new connections where already in the cluster, i.e if the cluster is complete
        clusters.append({remaining.pop()})  # start new cluster with an unchecked node
        return recursive_cluster(
            A, checked=checked, clusters=clusters
        )  # fill that new cluster recursively
    # if there are new connections
    clusters[-1].update(connected_nodes)  # add them to the current cluster
    return recursive_cluster(
        A, checked=checked, clusters=clusters
    )  # continue filling the current cluster recursively


if __name__ == "__main__":
    # build the matrix
    A = np.zeros((10, 10))
    connections = [
        (0, 2),
        (2, 5),
        (1, 7),
        (1, 8),
        (1, 9),
        (7, 8),
        (3, 4),
        (3, 6),
        (4, 6),
    ]  # only upper diagonal
    for idx in connections:
        A[idx] = 1
    A = A + A.T  # symetrize

    # find clusters
    clusters = recursive_cluster(A)
    print(clusters)
