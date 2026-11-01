"""python -m rgbd_hssg SNAPSHOT.json --output GRAPH.json [--previous GRAPH.json]"""
import argparse

from . import build_scene_graph, graph_delta, load_graph, load_snapshot, save_graph, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot')
    parser.add_argument('--output', required=True)
    parser.add_argument('--previous')
    parser.add_argument('--delta-output')
    args = parser.parse_args()
    objects, metadata = load_snapshot(args.snapshot)
    graph = build_scene_graph(objects, metadata)
    previous = load_graph(args.previous) if args.previous else None
    delta = graph_delta(previous, graph)
    save_graph(graph, args.output)
    if args.delta_output:
        save_json(delta, args.delta_output)
    print(f"Saved {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges to {args.output}")


if __name__ == '__main__':
    main()
