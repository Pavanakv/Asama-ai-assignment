import asyncio, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def run(coro):
    return asyncio.run(coro)
