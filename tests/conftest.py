"""Shared test helpers — ย้ายจาก tests/test_trainer_worker.py (Approach B split)"""


class FakeQueue:
    """จำลอง multiprocessing.Queue — เก็บ message ที่ put ไว้ให้ assert ทีหลัง"""

    def __init__(self):
        self.messages: list[dict] = []

    def put(self, msg: dict) -> None:
        self.messages.append(msg)
