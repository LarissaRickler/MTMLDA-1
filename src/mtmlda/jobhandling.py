from collections.abc import Callable, Sequence
import concurrent.futures as concurrent

import anytree as at


# ==================================================================================================
class JobHandler:
    def __init__(
        self, executor: concurrent.ThreadPoolExecutor, models: Sequence[Callable], num_threads: int
    ) -> None:
        self._futures = []
        self._futuremap = {}
        self._executor = executor
        self._models = models
        self._num_threads = num_threads

    # ----------------------------------------------------------------------------------------------
    def submit_job(self, node: at.AnyNode) -> None:
        node.computing = True
        future = self._executor.submit(self._models[node.level], [node.state.tolist()])
        self._futures.append(future)
        self._futuremap[future] = node

    # ----------------------------------------------------------------------------------------------
    def get_finished_jobs(self) -> tuple[list[float], list[at.AnyNode]]:
        results = []
        nodes = []

        for future in concurrent.as_completed(self._futures):
            result = future.result()[0][0]
            node = self._futuremap.pop(future)
            node.computing = False
            self._futures.remove(future)
            results.append(result)
            nodes.append(node)
            if not self._some_job_is_done():
                break

        return results, nodes

    # ----------------------------------------------------------------------------------------------
    def _some_job_is_done(self) -> bool:
        for future in self._futures:
            if future.done():
                return True
        return False

    # ----------------------------------------------------------------------------------------------
    @property
    def workers_available(self) -> bool:
        workers_available = len(self._futures) < self._num_threads
        return workers_available
