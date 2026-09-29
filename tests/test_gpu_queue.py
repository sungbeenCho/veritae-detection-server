import threading
import time

import pytest

from app.services.gpu_queue import GpuQueue, GpuQueueFullError


def test_acquire_runs_one_at_a_time():
    queue = GpuQueue(max_concurrent=1, max_queue_depth=10)
    order = []

    def worker(n):
        with queue.acquire(f"job{n}"):
            order.append(("start", n))
            time.sleep(0.05)
            order.append(("end", n))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # max_concurrent=1이므로 어떤 job도 다른 job이 끝나기 전에 시작하면 안 된다.
    for i in range(0, len(order), 2):
        assert order[i][0] == "start"
        assert order[i + 1] == ("end", order[i][1])


def test_acquire_allows_up_to_max_concurrent():
    queue = GpuQueue(max_concurrent=2, max_queue_depth=10)
    concurrent = []
    max_seen = []
    lock = threading.Lock()

    def worker():
        with queue.acquire("job"):
            with lock:
                concurrent.append(1)
                max_seen.append(len(concurrent))
            time.sleep(0.05)
            with lock:
                concurrent.pop()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max(max_seen) <= 2


def test_acquire_rejects_when_queue_full():
    queue = GpuQueue(max_concurrent=1, max_queue_depth=1)
    holder_entered = threading.Event()
    release_holder = threading.Event()

    def holder():
        with queue.acquire("holder"):
            holder_entered.set()
            release_holder.wait(timeout=2)

    def waiter():
        with queue.acquire("waiter"):
            pass

    t_holder = threading.Thread(target=holder)
    t_holder.start()
    holder_entered.wait(timeout=2)

    t_waiter = threading.Thread(target=waiter)
    t_waiter.start()
    time.sleep(0.05)  # waiter가 대기열에 들어갈 시간을 준다

    with pytest.raises(GpuQueueFullError):
        with queue.acquire("overflow"):
            pass

    release_holder.set()
    t_holder.join()
    t_waiter.join()
