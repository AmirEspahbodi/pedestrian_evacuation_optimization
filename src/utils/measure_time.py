import time


def timing_decorator(original_function):
    def wrapper_function(*args, **kwargs):
        start_time = time.time()
        result = original_function(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        # print(
        #     f"Function '{original_function.__name__}' executed in: {execution_time:.4f} seconds"
        # )
        return result

    # Return the wrapper function
    return wrapper_function
