import pyautogui as pag
import time

file_path = 'code.txt'

try:
    with open(file_path, 'r') as file:
        pag.moveTo(x=520, y=75, duration=2)  # Move to the location of the text box
        pag.click()
        time.sleep(3)

        content = file.read()
        lines = content.splitlines()
        for line in lines:
            if line.strip():
                pag.write(line + '\n', interval=0.1)  # Type each line with a small delay
except FileNotFoundError:
    print(f"The file at {file_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")