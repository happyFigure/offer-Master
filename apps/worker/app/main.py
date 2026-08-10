def build_worker_status() -> dict[str, str]:
    return {"service": "jobpilot-worker", "status": "idle"}


if __name__ == "__main__":
    print(build_worker_status())

