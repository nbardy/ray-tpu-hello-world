import subprocess

def run_command(cmd):
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with open('debug_log.txt', 'w') as log_file:
        while True:
            output = process.stdout.readline()
            if output == b'' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                log_file.write(output.decode())
        rc = process.poll()
    return rc

run_command("gcloud config set compute/zone us-west4-a")

