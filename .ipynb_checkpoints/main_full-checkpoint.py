import subprocess
import datetime

command = "python3 main.py --dataset animalkingdom --model timesformerclipinitvideoguide --gpu 0"

process = subprocess.run(command, shell=True, capture_output=True, text=True)

stdout = process.stdout
stderr = process.stderr

current_date = datetime.datetime.now()
date_string = current_date.strftime("%m-%d %H:%M")

output_file_path = f"Output/full_{date_string}.txt"

with open(output_file_path, "w") as file:
    file.write("Standard Output:\n")
    file.write(stdout)
    file.write("\nStandard Error:\n")
    file.write(stderr)

print(f"Output saved to: {output_file_path}")