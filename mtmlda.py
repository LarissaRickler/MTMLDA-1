from anytree import Node, NodeMixin, LevelOrderGroupIter, PreOrderIter, util #, Walker
import numpy as np
import umbridge
from concurrent.futures import ThreadPoolExecutor, as_completed

from anytree.exporter import DotExporter
import os


models = [umbridge.HTTPModel('http://localhost:4243', 'posterior'), umbridge.HTTPModel('http://localhost:4243', 'posterior_coarse')]
subsampling_rates = [5, -1]

class MTNodeBase(object):
  probability_reached = None
  state = None
  logposterior = None
  logposterior_coarse = None
  computing = False
  random_draw = None
  level = -1
  subchain_index = 0

class MTNode(MTNodeBase, NodeMixin):
  def __init__(self, name, parent=None, children=None):
    super(MTNodeBase, self).__init__()

    # Setup as required by anytree
    self.name = name
    self.parent = parent
    if children:
        self.children = children

# Basic strategy for estimating probability of needing a node: Simply go by estimated acceptance rate
def update_probability_reached2(root, acceptance_rate_estimate):

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

# Advanced strategy using a coarse model to estimate acceptance rate
def update_probability_reached3(root, acceptance_rate_estimate):

  model_coarse = umbridge.HTTPModel('http://localhost:4243', 'posterior_coarse')

  for level_children in LevelOrderGroupIter(root):
    for node in level_children:

      if node.name == 'r' and node.parent is not None:
          node.logposterior_coarse = node.parent.logposterior_coarse
      if node.logposterior_coarse is None:
        node.logposterior_coarse = model_coarse([node.state.tolist()])[0][0]

      if node.parent == None:
        node.probability_reached = 1.0
      elif node.name == 'a':
        accept_estimate = min(1, np.exp(node.logposterior_coarse - node.parent.logposterior_coarse))
        node.probability_reached = accept_estimate * node.parent.probability_reached
      elif node.name == 'r':
        sibling = util.leftsibling(node)
        accept_estimate = min(1, np.exp(sibling.logposterior_coarse - node.parent.logposterior_coarse))
        node.probability_reached = (1 - accept_estimate) * node.parent.probability_reached
      else:
        raise ValueError(f'Invalid node name: {node.name}')

def update_probability_reached(root, acceptance_rate_estimate):

  model_coarse = umbridge.HTTPModel('http://localhost:4243', 'posterior_coarse')

  for level_children in LevelOrderGroupIter(root):
    for node in level_children:

      if node.name == 'r' and node.parent is not None:
          node.logposterior_coarse = node.parent.logposterior_coarse
      if node.logposterior_coarse is None:
        node.logposterior_coarse = model_coarse([node.state.tolist()])[0][0]

      if node.parent == None:
        node.probability_reached = 1.0
      elif len(node.parent.children) == 1:
        node.probability_reached = node.parent.probability_reached
      #elif node.name == 'a' and node.logposterior is not None and node.parent.logposterior is not None:
      #  accept_probability = min(1, np.exp(node.logposterior - node.parent.logposterior))
      #  node.probability_reached = (1 if node.parent.random_draw < accept_probability else 0) * node.parent.probability_reached
      #elif node.name == 'r' and util.leftsibling(node).logposterior is not None and node.parent.logposterior is not None:
      #  accept_probability = min(1, np.exp(util.leftsibling(node).logposterior - node.parent.logposterior))
      #  node.probability_reached = (0 if node.parent.random_draw < accept_probability else 1) * node.parent.probability_reached
      elif node.name == 'a':
        accept_estimate = min(1, np.exp(node.logposterior_coarse - node.parent.logposterior_coarse))
        node.probability_reached = accept_estimate * node.parent.probability_reached
      elif node.name == 'r':
        sibling = util.leftsibling(node)
        accept_estimate = min(1, np.exp(sibling.logposterior_coarse - node.parent.logposterior_coarse))
        node.probability_reached = (1 - accept_estimate) * node.parent.probability_reached
      else:
        raise ValueError(f'Invalid node name: {node.name}')


def metropolis_hastings_proposal(current_state):
  return np.random.multivariate_normal(current_state, .01*np.eye(2))

def pcn_proposal(current_state):
  beta = 0.4
  return np.random.multivariate_normal(np.sqrt(1 - beta**2) * current_state, beta**2 * np.eye(2))


def propagate_log_posterior_to_reject_children(root):
  for level_children in LevelOrderGroupIter(root):
    for child in level_children:
      if child.name == 'r' and child.parent is not None:
        child.computing = child.parent.computing
        child.logposterior = child.parent.logposterior

def max_probability_todo_node(root):
  max_probability = .0
  max_node = None
  for node in PreOrderIter(root):
    if node.name == 'a': # Only 'accept' nodes need model evaluations, since 'reject' nodes are just copies of their parents
      parent_probability_reached = 1.0
      if node.parent is not None:
        parent_probability_reached = node.parent.probability_reached
      if parent_probability_reached > max_probability and node.logposterior is None and not node.computing:
        max_probability = parent_probability_reached
        max_node = node
  return max_node

counter_print = 0
def print_graph(root):
  #return
  global counter_print

  # Print graph to file
  def name_from_parents(node):
    if node.parent is None:
      return node.name
    else:
      return name_from_parents(node.parent) + node.name
  def node_name(node):
    return name_from_parents(node) + "\n" + str(node.probability_reached) + "\n" + str(node.level) + " (" + str(node.subchain_index) + ")" + "\n" + str(node.state[0])
  def nodeattrfunc(node):
    if node.logposterior is not None:
      return 'style=filled,fillcolor=green'
    if node.computing == True:
      return 'style=filled,fillcolor=yellow'
    return 'style=filled,fillcolor=white'
  DotExporter(root, nodenamefunc=node_name, nodeattrfunc=nodeattrfunc).to_dotfile(f"mtmc{str(counter_print).rjust(5, '0')}.dot")
  counter_print += 1

def get_same_level_parent(node):
  same_level_parent = node.parent
  while same_level_parent.level != node.level:
    same_level_parent = same_level_parent.parent
  return same_level_parent


def add_new_children_to_node(node):
  accepted = MTNode('a', parent=node)
  rejected = MTNode('r', parent=node)

  for new_node in [accepted, rejected]:
    new_node.random_draw = np.random.uniform()

  if node.subchain_index == subsampling_rates[node.level] - 1: # Reached end of subchain, go to next finer level
    for new_node in [accepted, rejected]:
      new_node.level = node.level + 1
      new_node.subchain_index = get_same_level_parent(new_node).subchain_index + 1
    accepted.state = node.state
    rejected.state = get_same_level_parent(rejected).state
  else: # Within subchain
    if node.level == 0: # Extend subchain on coarsest level
      for new_node in [accepted, rejected]:
        new_node.subchain_index = node.subchain_index + 1
        new_node.level = node.level
      accepted.state = pcn_proposal(node.state)
      rejected.state = node.state
    else: # Spawn new subchain on next coarser level
      for new_node in [accepted, rejected]:
        new_node.subchain_index = 0
        new_node.level = node.level - 1
      accepted.state = node.state
      rejected.state = node.state
      #rejected.parent = None # Don't need to add reject node here: When passing to coarse levels there is no accept/reject decision to be made


def expand_tree(root):
  # Iterate over tree, add new nodes to computed accept leaves
  for node in root.leaves:
    if node.name == 'a' and (node.logposterior is not None or node.computing):
      add_new_children_to_node(node)

  for node in root.leaves:
    if node.name == 'r' and (util.leftsibling(node).logposterior is not None or util.leftsibling(node).computing):
      add_new_children_to_node(node)



root = MTNode('a')
root.state = np.array([4.0,4.0])
root.random_draw = np.random.uniform()
chain = [root.state]
root.level = len(models)-1
root.subchain_index = 0

num_workers = 8

acceptance_rate_estimate = .8

counter_computed_models = 0

#print_mtree(root)

with ThreadPoolExecutor(max_workers=num_workers) as executor:
  futures = []
  futuremap = {}

  def submit_next_job(root, acceptance_rate_estimate):

    expand_tree(root)
    # Update (estimated) probability of reaching each node
    update_probability_reached(root, acceptance_rate_estimate)
    # Pick most likely (and not yet computed) node to evaluate
    todo_node = max_probability_todo_node(root)

    if todo_node is None:
      raise ValueError('No more nodes to compute, most likely tree expansion failed!')

    # Submit node for model evaluation
    todo_node.computing = True
    future = executor.submit(models[todo_node.level], [todo_node.state.tolist()])
    futures.append(future)
    futuremap[future] = todo_node

  # Initialize by submitting as many jobs as there are workers
  print_graph(root)
  for iter in range(0,num_workers):
    submit_next_job(root, acceptance_rate_estimate)
    print_graph(root)

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

    resolved_a_decision = True
    while resolved_a_decision:
      resolved_a_decision = False
      for level_children in LevelOrderGroupIter(root):
        for node in level_children:

          if node.name == 'a' and node.parent is not None and len(node.parent.children) > 1 and node.level == node.parent.level == 0 and node.logposterior is not None and node.parent.logposterior is not None:
            accept_probability = min(1, np.exp(node.logposterior - node.parent.logposterior))
            if node.parent.random_draw < accept_probability:
              util.rightsibling(node).parent = None
            else:
              node.parent = None
            print("resolved decision")
            resolved_a_decision = True
            print_graph(root)
            break # TODO: possibly better to break both fors?

          # TODO!!!
          if node.name == 'a' and node.parent is not None and len(node.parent.children) > 1 and node.level - 1 == node.parent.level and node.logposterior is not None and get_same_level_parent(node).logposterior is not None:
            accept_probability = min(1, np.exp(node.logposterior - get_same_level_parent(node).logposterior))
            if node.parent.random_draw < accept_probability:
              util.rightsibling(node).parent = None
            else:
              node.parent = None
            print("resolved decision")
            resolved_a_decision = True
            print_graph(root)
            break



    while False:


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
          if root.random_draw < np.exp(accept_sample.logposterior - root.logposterior):
            root = accept_sample
            acceptance_rate_estimate = acceptance_rate_estimate * .99 + .01
          else:
            root = reject_sample
            acceptance_rate_estimate = acceptance_rate_estimate * .99

        # If on finest level, add new state to chain
        if (root.level == len(models)-1):
          chain.append(root.state)

        print_graph(root)


    if len(chain) >= 100:
      break
    submit_next_job(root, acceptance_rate_estimate)

    print_graph(root)


print(f"MCMC chain length: {len(chain)}")
print(f"Model evaluations computed: {counter_computed_models}")
print(f"Acceptance rate: {acceptance_rate_estimate}")
