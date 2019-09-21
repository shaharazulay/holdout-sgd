import time
import multiprocessing as mp
import progressbar


def _create_pool():
    return mp.Pool(10)#mp.cpu_count())
    
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
    
    pool = _create_pool()
    results_obj = [pool.apply_async(func, args, callback=results.append) for args in args_list]
    
    while len(results) != len(args_list):
        print("WAITING")
        pbar.update(len(results))
        time.sleep(0.5)
    
    print("DONE")
    pbar.finish()
    pool.close()
    return results