import os
from pathlib import Path

import arviz as az
import pydot


# ==================================================================================================
chain_directory = Path("results")
dotfile_directory = Path("results") / Path("mltree")
visualize_tree = True


# ==================================================================================================
def postprocess_chains(chain_directory):
    pass

def render_dot_files(dotfile_directory):
    dot_files = [dotfile_directory/ Path(file) for file in os.listdir(dotfile_directory)
             if (os.path.isfile(dotfile_directory / Path(file)) and file.endswith("dot"))]
    
    for file in dot_files:
        graph = pydot.graph_from_dot_file(file)[0]
        graph.write_png(file.with_suffix(".png"))

def main():
    postprocess_chains()

    if visualize_tree:
        render_dot_files(dotfile_directory)


if __name__ == "__main__":
    main()