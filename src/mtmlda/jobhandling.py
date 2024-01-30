from concurrent.futures import as_completed


# ==================================================================================================
class JobHandler:
    def __init__(self, executor, models):
        self._futures = []
        self._futuremap = {}
        self._executor = executor
        self._models = models

    # ----------------------------------------------------------------------------------------------
    def submit_job(self, node):
        node.computing = True
        future = self._executor.submit(self._models[node.level], [node.state.tolist()])
        self._futures.append(future)
        self._futuremap[future] = node

    # ----------------------------------------------------------------------------------------------
    def get_finished_jobs(self):
        results = []
        nodes = []

        for future in as_completed(self._futures):
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
    def _some_job_is_done(self):
        for future in self._futures:
            if future.done():
                return True
        return False
    
    # ----------------------------------------------------------------------------------------------
    @property
    def num_busy_workers(self):
        return len(self._futures)