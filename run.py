import subprocess
import sys
import os
import platform
from rlbot.setup_manager import SetupManager
from rlbot.utils.python_version_check import check_python_version
if platform.system() == 'Windows':
    import msvcrt


DEFAULT_LOGGER = 'rlbot'


def set_up_env():
    try:
        from rlbot.utils import public_utils, logging_utils

        logger = logging_utils.get_logger(DEFAULT_LOGGER)
        if not public_utils.have_internet():
            logger.log(logging_utils.logging_level,
                       'Skipping upgrade check for now since it looks like you have no internet')
        elif public_utils.is_safe_to_upgrade():
            subprocess.call([sys.executable, "-m", "pip", "install", '-r', 'requirements.txt'])
            subprocess.call([sys.executable, "-m", "pip", "install", 'rlbot', '--upgrade'])

            # https://stackoverflow.com/a/44401013
            rlbots = [module for module in sys.modules if module.startswith('rlbot')]
            for rlbot_module in rlbots:
                sys.modules.pop(rlbot_module)

    except ImportError:
        subprocess.call([sys.executable, "-m", "pip", "install", '-r', 'requirements.txt', '--upgrade', '--upgrade-strategy=eager'])


def game_inifinite_loop(manager: SetupManager):
    instructions = "Press 'r' to reload all agents, or 'q' to exit"
    manager.logger.info(instructions)
    while not manager.quit_event.is_set():
        # Handle commands
        # TODO windows only library
        if platform.system() == 'Windows':
            if msvcrt.kbhit():
                command = msvcrt.getwch()
                if command.lower() == 'r':  # r: reload
                    manager.reload_all_agents()
                elif command.lower() == 'q' or command == '\u001b':  # q or ESC: quit
                    manager.shut_down()
                    break
                # Print instructions again if a alphabet character was pressed but no command was found
                elif command.isalpha():
                    manager.logger.info(instructions)
        else:
            try:
                # https://python-forum.io/Thread-msvcrt-getkey-for-linux
                import termios, sys
                TERMIOS = termios

                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                new = termios.tcgetattr(fd)
                new[3] = new[3] & ~TERMIOS.ICANON & ~TERMIOS.ECHO
                new[6][TERMIOS.VMIN] = 1
                new[6][TERMIOS.VTIME] = 0
                termios.tcsetattr(fd, TERMIOS.TCSANOW, new)
                command = None
                try:
                    command = os.read(fd, 1)
                finally:
                    termios.tcsetattr(fd, TERMIOS.TCSAFLUSH, old)
                command = command.decode("utf-8")
                if command.lower() == 'r':  # r: reload
                    manager.reload_all_agents()
                elif command.lower() == 'q' or command == '\u001b':  # q or ESC: quit
                    manager.shut_down()
                    break
                # Print instructions again if a alphabet character was pressed but no command was found
                elif command.isalpha():
                    manager.logger.info(instructions)
            except:
                pass

        manager.try_recieve_agent_metadata()


def run_game():
    try:
        check_python_version()
        manager = SetupManager()
        manager.load_config()
        manager.connect_to_game()
        manager.launch_early_start_bot_processes()
        manager.start_match()
        manager.launch_bot_processes()
        manager.infinite_loop()
    except Exception as e:
        print("Encountered exception: ", e)
        print("Press enter to close...")
        input()
        


if __name__ == '__main__':
    run_game()
