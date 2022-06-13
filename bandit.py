from __future__ import annotations

from typing import Optional

import numpy as np


class Bandit():
    def __init__(self,
                 T: int,
                 d: int,
                 K: int,
    ) -> None:
        self.T = T
        self.d = d
        self. K = K

    def get_action(self,
                   context: np.array,
    ) -> int:

        pass

    def update(self,
               action: int
    ) -> None:

        pass

    def reward(self,
               watch_list: np.array,
               action: int
    )-> int:

        pass