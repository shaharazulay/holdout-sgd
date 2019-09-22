import time
import multiprocessing as mp
import progressbar


def _create_pool():
    #return mp.Pool(mp.cpu_count())
    
    ###
    import sys
    # For python 2/3 compatibility, define pool context manager
    # to support the 'with' statement in Python 2
    if sys.version_info[0] == 2:
        from contextlib import contextmanager
        @contextmanager
        def multiprocessing_context(*args, **kwargs):
            pool = mp.Pool(*args, **kwargs)
            yield pool
            pool.terminate()
    else:
        multiprocessing_context = mp.Pool
        
    return multiprocessing_context
    ###
    
def run_in_parallel(func, args_list):    
    pool = _create_pool()
    results = pool.starmap(func, args_list)
    pool.close()
    return results
    
def async_run_in_parallel(func, args_list):
    pbar = progressbar.ProgressBar(
        maxval=len(args_list),
        widgets=[progressbar.Bar('=', '[', ']'), ' ', 
        progressbar.Percentage()]).start()
    
    results = []
    
    #pool = _create_pool()
    with _create_pool()(mp.cpu_count()) as pool:
        results_obj = [pool.apply_async(func, args, callback=results.append) for args in args_list]
        
        while len(results) != len(args_list):
            pbar.update(len(results))
            time.sleep(0.5)
    
    pbar.finish()
    #pool.close()
    return results
    