from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CurrentUser:
    name: str
    emp_id: str
