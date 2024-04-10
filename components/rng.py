def distribute_seeds_to_processes(seeds, process_id):
    if isinstance(seeds, list):
        return seeds[process_id]
    else:
        return int(seeds * process_id)
