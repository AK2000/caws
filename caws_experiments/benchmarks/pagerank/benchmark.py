from caws.task import caws_task
from caws.features import ArgFeature

def graph_pagerank(size):
    import datetime
    import igraph

    graph_generating_begin = datetime.datetime.now()
    graph = igraph.Graph.Barabasi(size, 10)
    graph_generating_end = datetime.datetime.now()

    process_begin = datetime.datetime.now()
    result = graph.pagerank()
    process_end = datetime.datetime.now()

    graph_generating_time = (graph_generating_end - graph_generating_begin) / datetime.timedelta(microseconds=1)
    process_time = (process_end - process_begin) / datetime.timedelta(microseconds=1)

    return {
            'result': result[0],
            'measurement': {
                'graph_generating_time': graph_generating_time,
                'compute_time': process_time
            }
    }

graph_pagerank = caws_task(graph_pagerank, features=[ArgFeature()])