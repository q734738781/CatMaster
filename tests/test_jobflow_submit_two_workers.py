#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小连通性测试：
分别向 cpu-worker 与 gpu-worker 提交 add→add 两步 Flow
轮询 JobStore 直到取得第二步作业输出
"""
from __future__ import annotations
import time
from jobflow_remote.utils.examples import add
from jobflow_remote import submit_flow, get_jobstore
from jobflow import Flow

CPU_WORKER = "cpu-worker"
GPU_WORKER = "gpu-worker"
POLL_INTERVAL = 2.0
TIMEOUT_SEC = 600

def wait_output(js, uuid: str, timeout: int = TIMEOUT_SEC, interval: float = POLL_INTERVAL):
    start = time.time()
    while True:
        try:
            out = js.get_output(uuid)
            if out is not None:
                return out
        except Exception:
            # 作业尚未完成或暂不可读时继续轮询
            pass
        if time.time() - start > timeout:
            raise TimeoutError(f"等待作业 {uuid} 超过 {timeout} 秒")
        time.sleep(interval)

def submit_simple_flow(worker: str, a: int, b: int, c: int):
    j1 = add(a, b)
    j2 = add(j1.output, c)
    flow = Flow([j1, j2])
    dbid = submit_flow(flow, worker=worker)
    return dbid, j2.uuid

def main():
    js = get_jobstore()
    js.connect()

    print("提交 CPU 流程……")
    cpu_dbid, cpu_uuid = submit_simple_flow(CPU_WORKER, 1, 2, 3)

    print("提交 GPU 流程……")
    gpu_dbid, gpu_uuid = submit_simple_flow(GPU_WORKER, 10, 20, 30)

    cpu_result = wait_output(js, cpu_uuid)
    gpu_result = wait_output(js, gpu_uuid)

    print(f"CPU DB id: {cpu_dbid}，结果: {cpu_result}，期望: 6")
    print(f"GPU DB id: {gpu_dbid}，结果: {gpu_result}，期望: 60")
    print("完成。")

if __name__ == "__main__":
    main()
