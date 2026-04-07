"""
Vast.ai PyWorker proxy for HybrIK-X.

This file is only needed when deploying on vast.ai Serverless.
Enable by uncommenting the vastai-sdk install + COPY in the Dockerfile,
and changing CMD to run this instead of start.sh.
"""
import base64
import os

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

# Use a sample image for benchmarking (if available)
SAMPLE_IMG = "/app/HybrIK/examples/img1.jpg"
TEST_IMAGE_B64 = ""
if os.path.exists(SAMPLE_IMG):
    with open(SAMPLE_IMG, "rb") as f:
        TEST_IMAGE_B64 = base64.b64encode(f.read()).decode()

worker_config = WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=8000,
    model_log_file="/var/log/portal/model.log",
    model_healthcheck_url="/health",
    handlers=[
        HandlerConfig(
            route="/predict",
            allow_parallel_requests=False,
            max_queue_time=120.0,
            workload_calculator=lambda data: 100.0,
            benchmark_config=BenchmarkConfig(
                generator=lambda: {"image_base64": TEST_IMAGE_B64},
                runs=2,
                concurrency=1,
            ),
        )
    ],
    log_action_config=LogActionConfig(
        on_load=["Application startup complete."],
        on_error=["Traceback (most recent call last):", "RuntimeError:"],
        on_info=[],
    ),
)

Worker(worker_config).run()
