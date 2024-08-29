import time
import signal

class TimeoutExpired(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutExpired

def prompt_user_to_overwrite(message, timeout=5, default=False):
    """ 
    Prompts the user to overwrite with a timeout.

    Parameters:
    - message: The message to display to the user.
    - timeout: The number of seconds to wait for user input before timing out.
    - default: The default return value if the timeout expires. Should be True or False.

    Returns:
    - True if the user inputs 'y', False if the user inputs 'n', or the default value if the timeout expires.
    """
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        user_input = input(message).strip().lower()
        signal.alarm(0)  # Cancel the alarm
        if user_input in ['y', 'n']:
            return user_input == 'y' 
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
            return default
    except TimeoutExpired:
        print(f"\nTimeout expired. Proceeding with default choice: {'Yes' if default else 'No'}.")
        return default

