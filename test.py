import ray

@ray.remote(resources={"TPU": 1})
def io_bound_task():
    import time

    time.sleep(1)
    return 2


io_bound_task.remote()

@ray.remote(resources={"TPU": 1})
class IOActor:
    def ping(self):
        import os

        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        print("DONE")


# Two actors can share the same GPU.
io_actor1 = IOActor.remote()
io_actor2 = IOActor.remote()

print("Running ray get")
ray.get(io_actor1.ping.remote())
ray.get(io_actor2.ping.remote())
