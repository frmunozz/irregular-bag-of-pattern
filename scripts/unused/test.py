def worker(x, y, out_q):
    out_q.put((x, x*x))