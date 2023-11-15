import typing as tp

from pathlib import Path

import random
import logging

import numpy as np

from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from carball.json_parser.game import Game

import subprocess
import json


class RandomReplayStateSetter(StateSetter):
    
    def __init__(self, replays_cfg: tp.Dict[str, tp.Union[str, float, int]]):
        self.replays_cfg = replays_cfg
        self.replay_data_paths = self._get_replay_data_paths()
        self.games: tp.List[Game] = self._read_random_games()
        self._call_index = 0
        self._read_random_games()
        
    def _get_replay_data_paths(self) -> tp.List[Path]:
        folder_path = Path(str(self.replays_cfg['data_folder']))
        out_paths = []
        for path in folder_path.iterdir():
            if path.is_dir():
                continue
            if path.suffix != '.replay':
                continue
            out_paths.append(path)
        return out_paths
    
    def _read_random_games(self) -> tp.List[Game]:
        max_readed_games = int(self.replays_cfg['max_readed_games'])
        np.random.shuffle(self.replay_data_paths)
        games = []
        for path in self.replay_data_paths[:max_readed_games]:
            proc_out = subprocess.run(['tools/rrrocket.exe', '-n', str(path)], capture_output=True)
            json_data = json.loads(proc_out.stdout.decode('cp1251', errors='ignore'))
            game = Game()
            game.initialize(loaded_json=json_data)
            games.append(game)
        return games
            
    def _get_random_game(self) -> Game:
        reread_games_every_n_step = int(self.replays_cfg['reread_games_every_n_step'])
        self._call_index += 1
        if self._call_index % reread_games_every_n_step == 0:
            print(f'Replays reading...')
            self._read_random_games()
        return np.random.choice(self.games)
    
    def _check_null_vals(self, dict_values: tp.Dict[str, tp.Any], check_keys: tp.List[str]):
        for value_name in check_keys:
            if dict_values[value_name] is None:
                raise ValueError(f'Value "{value_name}" is None')
        

    def reset(self, state_wrapper: StateWrapper):
        
        for try_idx in range(10):
            try:
                game = self._get_random_game()
            
                time_size = min(game.ball.shape[0], min([game.players[player_idx].data.shape[0]
                                                        for player_idx in range(len(game.players))]))
                frame_idx = random.randint(0, time_size - 1)
                
                ball_info = game.ball.loc[frame_idx]
                # self._check_null_vals(ball_info, ['pos_x', 'pos_y', 'pos_z',
                #                                   'vel_x', 'vel_y', 'vel_z',
                #                                   'ang_vel_x', 'ang_vel_y', 'ang_vel_z'])
                
                players_info = []
                for player_idx in range(len(state_wrapper.cars)):
                    player_info = game.players[player_idx].data.loc[frame_idx]
                    # self._check_null_vals(player_info, ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z',
                    #                                     'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
                    #                                     'rot_x', 'rot_y', 'rot_z', 'boost'])
                    players_info.append(player_info)
                
                state_wrapper.ball.set_pos(float(ball_info['pos_x']),
                                           float(ball_info['pos_y']),
                                           float(ball_info['pos_z']))
                state_wrapper.ball.set_lin_vel(float(ball_info['vel_x']),
                                               float(ball_info['vel_y']),
                                               float(ball_info['vel_z']))
                state_wrapper.ball.set_ang_vel(float(ball_info['ang_vel_x']),
                                               float(ball_info['ang_vel_y']),
                                               float(ball_info['ang_vel_z']))
                
                for car, player_info in zip(state_wrapper.cars, players_info):
                    car.set_pos(float(player_info['pos_x']),
                                float(player_info['pos_y']),
                                float(player_info['pos_z']))
                    car.set_lin_vel(float(player_info['vel_x']),
                                    float(player_info['vel_y']),
                                    float(player_info['vel_z']))
                    car.set_ang_vel(float(player_info['ang_vel_x']),
                                    float(player_info['ang_vel_y']),
                                    float(player_info['ang_vel_z']))
                    car.set_rot(float(player_info['rot_x']),
                                float(player_info['rot_y']),
                                float(player_info['rot_z']))
                    car.boost = int(player_info['boost'])
                break
            except Exception as ex:
                logging.warning(f'RandomReplayStateSetter[{try_idx}]: {ex}')
                print(f'RandomReplayStateSetter[{try_idx}]: {ex}')
                continue
