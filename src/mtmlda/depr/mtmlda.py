from anytree import Node, NodeMixin, LevelOrderGroupIter, PreOrderIter, util
import numpy as np
import umbridge
from concurrent.futures import ThreadPoolExecutor, as_completed


root = MTNode("a")
root.state = np.array([4.0, 4.0])
root.random_draw = np.random.uniform()
chain = [root.state]
root.level = len(models) - 1
root.subchain_index = 0

num_workers = 8

acceptance_rate_estimate = [0.5, 0.7, 0.8]

counter_computed_models = 0

# print_mtree(root)

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = []
    futuremap = {}

    def submit_next_job(root, acceptance_rate_estimate):
        expand_tree(root)
        # Update (estimated) probability of reaching each node
        update_probability_reached2(root, acceptance_rate_estimate)
        # Pick most likely (and not yet computed) node to evaluate
        todo_node = max_probability_todo_node(root)

        if todo_node is None:
            raise ValueError("No more nodes to compute, most likely tree expansion failed!")

        # Submit node for model evaluation
        todo_node.computing = True
        future = executor.submit(models[todo_node.level], [todo_node.state.tolist()])
        futures.append(future)
        futuremap[future] = todo_node

    def some_job_is_done():
        for future in futures:
            if future.done():
                return True
        return False

    # Initialize by submitting as many jobs as there are workers
    print_graph(root)
    for iter in range(0, num_workers):
        submit_next_job(root, acceptance_rate_estimate)
        print_graph(root)

    while True:
        # Wait for model evaluation to finish
        # while some_job_is_done():
        for future in as_completed(futures):
            computed_node = futuremap.pop(future)
            futures.remove(future)
            print("job done, futures: " + str(len(futures)))
            counter_computed_models += 1
            # Set logposterior of newly computed node, and update any reject children (same of which may have been added recently)
            computed_node.logposterior = future.result()[0][0]
            propagate_log_posterior_to_reject_children(root)
            if not some_job_is_done():
                break

        print_graph(root)

        resolved_a_decision = True
        while resolved_a_decision:
            resolved_a_decision = False
            for level_children in LevelOrderGroupIter(root):
                for node in level_children:
                    if (
                        node.name == "a"
                        and node.parent is not None
                        and len(node.parent.children) > 1
                        and node.level == node.parent.level == 0
                        and node.logposterior is not None
                        and node.parent.logposterior is not None
                    ):
                        accept_probability = min(
                            1, np.exp(node.logposterior - node.parent.logposterior)
                        )
                        if node.parent.random_draw < accept_probability:
                            util.rightsibling(node).parent = None
                            acceptance_rate_estimate[node.level] = (
                                acceptance_rate_estimate[node.level] * 0.99 + 0.01
                            )
                        else:
                            node.parent = None
                            acceptance_rate_estimate[node.level] = (
                                acceptance_rate_estimate[node.level] * 0.99
                            )
                        print("resolved level 0 decision")
                        resolved_a_decision = True
                        print_graph(root)
                    elif (
                        node.name == "a"
                        and node.parent is not None
                        and len(node.parent.children) > 1
                        and node.level - 1 == node.parent.level
                        and node.logposterior is not None
                        and node.parent.logposterior is not None
                        and get_same_level_parent(node).logposterior is not None
                        and get_same_level_parent_child_on_path(node).logposterior is not None
                    ):
                        same_level_parent = get_same_level_parent(node)
                        same_level_parent_child = same_level_parent.children[0]
                        accept_probability = min(
                            1,
                            np.exp(
                                node.logposterior
                                - get_same_level_parent(node).logposterior
                                - same_level_parent.logposterior
                                + same_level_parent_child.logposterior
                            ),
                        )
                        if node.parent.random_draw < accept_probability:
                            util.rightsibling(node).parent = None
                            acceptance_rate_estimate[node.level] = (
                                acceptance_rate_estimate[node.level] * 0.99 + 0.01
                            )
                        else:
                            node.parent = None
                            acceptance_rate_estimate[node.level] = (
                                acceptance_rate_estimate[node.level] * 0.99
                            )
                        print("resolved ML decision")
                        resolved_a_decision = True
                        print_graph(root)

                    if resolved_a_decision:
                        break
                if resolved_a_decision:
                    break

        unique_child = get_unique_fine_level_child(root)
        if unique_child is not None:
            chain.append(root.state)
            unique_child.parent = None
            root = unique_child
            print("extended chain")

        print(acceptance_rate_estimate)

        if len(chain) >= 100:
            break
        while len(futures) < num_workers:
            submit_next_job(root, acceptance_rate_estimate)
            print("job submitted, futures: " + str(len(futures)))

        print_graph(root)


print(f"MCMC chain length: {len(chain)}")
print(f"Model evaluations computed: {counter_computed_models}")
print(f"Acceptance rate: {acceptance_rate_estimate}")
