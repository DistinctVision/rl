from pathlib import Path
from daft_quick_nick.utils.state_predictor_debug_wrapper import StatePredictorDebugWrapper
from Nexto.bot import Nexto


class NextoDebug0(StatePredictorDebugWrapper[Nexto]):
    
    def __init__(self, name, team, index):
        super().__init__(name=name, team=team, index=index)