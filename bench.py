import json, os, platform, subprocess, time, urllib.request
import numpy as np

def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()

r = {"cpus": os.cpu_count(), "arch": platform.machine(),
     "lscpu": sh("lscpu | egrep 'Model name|Flags' | head -2 | cut -c1-600"),
     "mem_gb": round(int(sh("awk '/MemTotal/{print $2}' /proc/meminfo")) / 1e6, 1),
     "disk_free_gb": round(os.statvfs("/").f_bavail * os.statvfs("/").f_frsize / 1e9, 1)}

# memory bandwidth: streaming sum over 1 GiB
a = np.ones(1 << 27, dtype=np.float64)
t = time.perf_counter()
for _ in range(5): a.sum()
r["mem_bw_GBps"] = round(5 * a.nbytes / (time.perf_counter() - t) / 1e9, 1)

# fp32 matmul throughput
n = 2048
x = np.random.rand(n, n).astype(np.float32)
t = time.perf_counter()
for _ in range(5): x @ x
r["matmul_fp32_GFLOPS"] = round(5 * 2 * n**3 / (time.perf_counter() - t) / 1e9, 1)

# network: latency + bandwidth to Hugging Face (capped 256 MB, discarded)
t = time.perf_counter()
urllib.request.urlopen("https://huggingface.co", timeout=15).read(1)
r["hf_first_byte_s"] = round(time.perf_counter() - t, 3)
try:
    req = urllib.request.Request(
        "https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/config.json",
        headers={"Range": "bytes=0-1000000"})
    urllib.request.urlopen(req, timeout=30).read()
except Exception as e:
    r["hf_err"] = str(e)
print(json.dumps(r, indent=1))
