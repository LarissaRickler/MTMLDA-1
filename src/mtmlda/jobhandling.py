"""Thread-parallel job handling for posterior evaluations.

This module provides the interface for thread-parallel and asynchronous submission and retrieval of
compute jobs for posterior evaluations. The functionality is based on Python's `ThreadPoolExecutor`.

Classes:
    JobHandler: Thread-parallel job handling for model evaluations in the MLDA algorithm
"""

import concurrent.futures as concurrent
from collections.abc import Callable, Sequence

import anytree as at


# ==================================================================================================
class JobHandler:
    """Handler for thread parallel job requests and retrieval.

    Model evaluations correspond to the hierarchy of posteriors in the MLDA algorithm. The main
    components of the class are a Python ThreadPoolExecutor and a list of callables that resemble
    the model hierarchy. All jobs are associated with nodes in a Markov tree. The activity of these
    nodes is modified accordingly, along with the posterior of the node from the result of the model
    evaluation. The execution is non-blocking, but the precise nature of the callable dictates if
    the execution is actually parallel.
        
    Methods:
        submit_job: Submit a job to the threadpool executor
        get_finished_jobs: Retrieve all jobs that have been fully executed
        some_job_is_done: Check if any of the submitted jobs is done
        workers_available: Check if free workers are available in the thread pool
        num_evaluations: Returns number of jobs that have been submitted for each model
    """

    def __init__(
        self, executor: concurrent.ThreadPoolExecutor, models: Sequence[Callable], num_threads: int
    ) -> None:
        """Constructor of the JobHandler.

        The constructor only initializes data structures and does not perform any actions.

        Args:
            executor (concurrent.ThreadPoolExecutor): Python Threadpool object to use
            models (Sequence[Callable]): List of callables that represent models to evaluate
            num_threads (int): Number of threads in the pool
        """
        self._futures = []
        self._futuremap = {}
        self._executor = executor
        self._models = models
        self._num_threads = num_threads
        self._num_evals = [
            0,
        ] * len(models)

    # ----------------------------------------------------------------------------------------------
    def submit_job(self, node: at.AnyNode) -> None:
        """Submit a job to the threadpool executor.

        Jobs are associated with a node in the Markov tree. Upon submission, the node is marked as
        computing. Note that the computation is non-blocking, but the precise nature of the callable
        dictates if the execution is actually parallel.

        Args:
            node (at.AnyNode): Node for which to evaluate the model
        """
        assert node.level < len(self._models), "Node level exceeds number of models"
        assert node.state is not None, "Node state is None"

        node.computing = True
        future = self._executor.submit(self._models[node.level], [node.state.tolist()])
        self._futures.append(future)
        self._futuremap[future] = node
        self._num_evals[node.level] += 1

    # ----------------------------------------------------------------------------------------------
    def get_finished_jobs(self) -> tuple[list[float], list[at.AnyNode]]:
        """Retrieve all jobs that have been fully executed.

        Submitted jobs in a `ThreadPoolExecutor` object are automatically checked for completion
        with the `as_completed` method. Nodes of completed jobs are marked as not computing. The
        routine assumes that the result of the submitted callable is a list of lists to match the
        UM-Bridge interface. Only the entry `[0][0]` is treated as the result of the computation.

        Returns:
            tuple[list[float], list[at.AnyNode]]: Lists of job results (model evaluation) and
                corresponding nodes 
        """
        results = []
        nodes = []

        for future in concurrent.as_completed(self._futures):
            self._futures.remove(future)
            result = future.result()[0][0]
            results.append(result)
            node = self._futuremap.pop(future)
            node.computing = False
            nodes.append(node)

            # Leave the routine when no further jobs are currently finished
            if not self._some_job_is_done():
                break

        return results, nodes

    # ----------------------------------------------------------------------------------------------
    def _some_job_is_done(self) -> bool:
        """Check if any of the submitted jobs is done."""
        for future in self._futures:
            if future.done():
                return True
        return False

    # ----------------------------------------------------------------------------------------------
    @property
    def workers_available(self) -> bool:
        """Check if free workers are available in the thread pool."""
        workers_available = len(self._futures) < self._num_threads
        return workers_available

    # ----------------------------------------------------------------------------------------------
    @property
    def num_evaluations(self) -> list[int]:
        """Return number of jobs that have been submitted for each model."""
        return self._num_evals
