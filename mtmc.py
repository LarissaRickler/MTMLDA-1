from anytree import Node, NodeMixin, RenderTree, LevelOrderGroupIter, PreOrderIter #, Walker
import numpy as np
import umbridge
from concurrent.futures import ThreadPoolExecutor, as_completed
import model

from anytree.exporter import DotExporter
import os

class MTNodeBase(object):
  probability_reached = None
  state = None
  logposterior = None
  computing = False

class MTNode(MTNodeBase, NodeMixin):
  def __init__(self, name, parent=None, children=None):
    super(MTNodeBase, self).__init__()

    # Setup as required by anytree
    self.name = name
    self.parent = parent
    if children:
        self.children = children

def print_mtree(root):
  for pre, _, node in RenderTree(root):
    treestr = u"%s%s" % (pre, node.name)
    print(treestr.ljust(8), node.probability_reached, node.state, node.logposterior)

def update_probability_reached(root, acceptance_rate_estimate):

  for level_children in LevelOrderGroupIter(root):
    for node in level_children:
      if node.parent == None:
        node.probability_reached = 1.0
      elif node.name == 'a':
        node.probability_reached = acceptance_rate_estimate * node.parent.probability_reached
      elif node.name == 'r':
        node.probability_reached = (1 - acceptance_rate_estimate) * node.parent.probability_reached
      else:
        raise ValueError(f'Invalid node name: {node.name}')

def metropolis_hastings_proposal(current_state):
  return np.random.multivariate_normal(current_state, .01*np.eye(2))

def add_layer_to_tree(root):
  *_, last_level = LevelOrderGroupIter(root)

  for last_level_node in last_level:
    accepted = MTNode('a', parent=last_level_node)
    rejected = MTNode('r', parent=last_level_node)
    accepted.state = metropolis_hastings_proposal(last_level_node.state)
    rejected.state = last_level_node.state

def propagate_log_posterior_to_reject_children(root):
  for level_children in LevelOrderGroupIter(root):
    for child in level_children:
      if child.name == 'r' and child.parent is not None:
        child.logposterior = child.parent.logposterior

def max_probability_todo_node(root):
  max_probability = .0
  max_node = None
  for node in PreOrderIter(root):
    if node.name == 'a': # Only 'accept' nodes need model evaluations, since 'reject' nodes are just copies of their parents
      if node.probability_reached > max_probability and node.logposterior is None and not node.computing:
        max_probability = node.probability_reached
        max_node = node
  return max_node

counter_print = 0
def print_graph(root):
  global counter_print

  # Print graph to file
  def name_from_parents(node):
    if node.parent is None:
      return node.name
    else:
      return name_from_parents(node.parent) + node.name
  def nodeattrfunc(node):
    if node.logposterior is not None:
      return 'style=filled,fillcolor=green'
    if node.computing == True:
      return 'style=filled,fillcolor=yellow'
    return 'style=filled,fillcolor=white'
  DotExporter(root, nodenamefunc=name_from_parents, nodeattrfunc=nodeattrfunc).to_dotfile(f"mtmc{str(counter_print).rjust(5, '0')}.dot")
  counter_print += 1


model = model.Banana() #umbridge.HTTPModel('http://localhost:4243', 'posterior')

root = MTNode('a')
root.state = np.array([4.0,4.0])
chain = [root.state]

for fill_iter in range(0,6):
  add_layer_to_tree(root)

acceptance_rate_estimate = .8

counter_computed_models = 0

print_mtree(root)

with ThreadPoolExecutor(max_workers=8) as executor:
  futures = []
  futuremap = {}

  def submit_next_job(root, acceptance_rate_estimate):

    # Update (estimated) probability of reaching each node
    update_probability_reached(root, acceptance_rate_estimate)
    # Pick most likely (and not yet computed) node to evaluate
    todo_node = max_probability_todo_node(root)

    if todo_node is None:
      raise ValueError('No more nodes to compute, need to increase tree size')

    # Submit node for model evaluation
    todo_node.computing = True
    future = executor.submit(model, [todo_node.state.tolist()])
    futures.append(future)
    futuremap[future] = todo_node

  # Initialize by submitting as many jobs as there are workers
  for iter in range(0,8):
    submit_next_job(root, acceptance_rate_estimate)

  while True:

    # Wait for model evaluation to finish
    future = next(as_completed(futures))
    computed_node = futuremap.pop(future)
    futures.remove(future)
    counter_computed_models += 1

    # Set logposterior of newly computed node, and update any reject children (same of which may have been added recently)
    computed_node.logposterior = future.result()[0][0]
    propagate_log_posterior_to_reject_children(root)

    print_graph(root)


    while True:
      # Retrieve root's children
      accept_sample = [node for node in root.children if node.name == 'a'][0]
      reject_sample = [node for node in root.children if node.name == 'r'][0]

      # See if we can compute an accept/reject step (root and accept child logposterior must be computed)
      if root.logposterior is None or accept_sample.logposterior is None:
        break
      else:

        accept_sample.parent = None
        reject_sample.parent = None

        # Depending on accept/reject, change root to current root's accept or reject child
        # (and orphan the other child, game of thrones style)
        if accept_sample.logposterior > root.logposterior:
          root = accept_sample
        else:
          if np.random.uniform() < np.exp(accept_sample.logposterior - root.logposterior):
            root = accept_sample
            acceptance_rate_estimate = acceptance_rate_estimate * .99 + .01
          else:
            root = reject_sample
            acceptance_rate_estimate = acceptance_rate_estimate * .99

        print_graph(root)


        # Add new state to chain and add a new layer to the tree to compensate for removed old root and sibling subtree
        chain.append(root.state)
        add_layer_to_tree(root)

        print_graph(root)


    if len(chain) >= 50:
      break
    submit_next_job(root, acceptance_rate_estimate)

    print_graph(root)


print_mtree(root)

print(f"MCMC chain length: {len(chain)}")
print(f"Model evaluations computed: {counter_computed_models}")
print(f"Acceptance rate: {acceptance_rate_estimate}")
