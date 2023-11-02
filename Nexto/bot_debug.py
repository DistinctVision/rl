from pathlib import Path
from ppocket_rocket.utils.state_predictor_debug_wrapper import StatePredictorDebugWrapper
from Nexto.bot import Nexto


class NextoDebug0(StatePredictorDebugWrapper[Nexto]):
    
    def __init__(self, name, team, index):
        super().__init__(name=name, team=team, index=index)