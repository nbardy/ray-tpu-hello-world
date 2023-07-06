import logging
import shutil
from rich.console import Group
import re
from rich.table import Table
from rich.live import Live
import os
import time
import subprocess
import json
from googleapiclient import discovery
from googleapiclient.errors import HttpError

os.environ['PROJECT_ID'] = 'facet-250518'
os.environ['NUM_CHIPS'] = '16'
os.environ['ACCELERATOR_TYPE'] = 'v5litepod-' + os.environ['NUM_CHIPS']
os.environ['ZONE'] = 'us-west4-a'
os.environ['RUNTIME_VERSION'] = 'v2-alpha-tpuv5-lite'
os.environ['TPU_NAME'] = 'nbardy-tpuv5-lite-' + os.environ['NUM_CHIPS']
os.environ['QUEUED_RESOURCE_ID'] = 'nbady-tpuv5-lite-queued-resource-v2-'
os.environ['ZONE'] + '-' + os.environ['NUM_CHIPS']

DEFAULT_CHILD_NODE_COUNT = 4

compute = discovery.build('compute', 'v1')


def run_command(cmd, retries=5):
    while retries:
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            return stdout
        else:
            print(
                f"Failed to run command: {cmd}, error: {stderr}, retrying...")
            retries -= 1
            time.sleep(5)
    raise Exception(f"Failed to run command: {cmd}, error: {stderr}")


# Create a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command_log(cmd, label=None, machine_id=None, retries=1):
    os.makedirs("machine_ups", exist_ok=True)
    os.makedirs("latest", exist_ok=True)

    log_file_path = f"machine_ups/{machine_id}.log"
    print(f"Running command: {label}")
    print("Logging to: ", log_file_path)
    print("Command: ", cmd)

    while retries:
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        with open(log_file_path, 'a') as log_file:
            while True:
                output = process.stdout.readline()
                if output == b'' and process.poll() is not None:
                    break
                if output:
                    print(output.strip().decode())
                    log_file.write(output.decode())

        if process.poll() == 0:
            # copy log file to latest folder
            shutil.copy2(log_file_path, "latest/")
            return process.stdout
        else:
            print(
                f"Failed to run command: {cmd}, retrying...")
            retries -= 1
            time.sleep(5)

    # copy log file to latest folder
    shutil.copy2(log_file_path, "latest/")
    raise Exception(f"Failed to run command: {cmd}")


# updated launch_node function

def launch_node():
    # Make uuid for machine-id of 6 characters
    machine_id = run_command("uuidgen | cut -c1-6").decode('utf-8').strip()

    queued_resource_id = os.environ['QUEUED_RESOURCE_ID'] + machine_id
    tpu_name = os.environ['TPU_NAME'] + machine_id

    # break into many commands
    set_config = f"gcloud config set compute/zone {os.environ['ZONE']} --quiet"
    run_command_log(set_config, label="Set config", machine_id=machine_id)

    delete_tpu_vm = f"gcloud alpha compute tpus tpu-vm delete {tpu_name} --zone {os.environ['ZONE']} --project {os.environ['PROJECT_ID']} --quiet"
    try:
        run_command_log(delete_tpu_vm, label="Delete TPU VM",
                        machine_id=machine_id)
    except Exception as e:
        print(e)

    delete_queued_resource = f"""
    gcloud alpha compute tpus queued-resources delete {queued_resource_id} \
    --project {os.environ['PROJECT_ID']} \
    --zone {os.environ['ZONE']}
    --quiet \
    """

    # This one can fail if it doesn't exist that's ok
    try:
        run_command_log(delete_queued_resource, label="Delete Queued Resource",
                        machine_id=machine_id)
    except Exception as e:
        print(e)

    create_queued_resource = f"""
    gcloud alpha compute tpus queued-resources create {queued_resource_id} \
    --project {os.environ['PROJECT_ID']} \
    --node-id {tpu_name} \
    --zone {os.environ['ZONE']} \
    --accelerator-type {os.environ['ACCELERATOR_TYPE']} \
    --runtime-version {os.environ['RUNTIME_VERSION']} \
    --quiet \
    """
    run_command_log(create_queued_resource, label="Create Queued Resource",
                    machine_id=machine_id)

    return machine_id


def start_ray_head_node():
    import ray
    print("Starting new Ray Head Node")
    ray_cluster = ray.init()

    print("Debug: ")
    print(ray_cluster)

    return ray_cluster


def run_command_on_gcloud(command, zone=None, instance=None):
    # must pass zone and instance
    # check
    if not zone or not instance:
        raise Exception(
            f'zone and instance must be passed to run_command_on_gcloud. zone: {zone}, instance: {instance}'
        )

    cmd = [
        'gcloud', 'compute', 'ssh', '--zone', zone, instance, '--command', command
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise Exception(f'Error executing command: {result.stderr.decode()}')
    return result.stdout.decode()


def start_ray_on_child(head_node_ip, zone=None, instance=None):
    if not zone or not instance:
        raise Exception(
            f'zone and instance must be passed to start_ray_on_child. zone: {zone}, instance: {instance}'
        )

    cmd = f"""
    ray start - -address = {head_node_ip}: 6379 - -resources = '{"TPU": 4}' - -node-ip-address =$(curl - H Metadata-Flavor: Google http: // metadata/computeMetadata/v1/instance/network-interfaces/0/ip)
    """
    return run_command_on_gcloud(cmd, zone=zone, instance=instance)


def check_node_status():
    import ray
    return ray.nodes()


def all_tpu_status(zone):
    cmd = ["gcloud", "alpha", "compute", "tpus", "tpu-vm",
           "list", "--zone", zone, "--format=json"]

    result = subprocess.run(cmd, text=True, capture_output=True)

    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        return None

    tpu_info = json.loads(result.stdout)

    # return all

    return tpu_info


def get_tpu_status(zone, tpu):
    cmd = ["gcloud", "alpha", "compute", "tpus", "tpu-vm",
           "describe", tpu, "--zone", zone, "--format=json"]

    result = subprocess.run(cmd, text=True, capture_output=True)

    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        return None

    tpu_info = json.loads(result.stdout)

    # The status of the TPU VM is stored in the 'state' field
    state = tpu_info.get('state')
    if state:
        print(f"TPU VM state: {state}")
        return state
    else:
        print("State not found in TPU VM info.")
        return None


def create_UI_tables():
    machine_summary_table = Table(
        show_header=True, header_style="bold magenta")
    machine_summary_table.add_column("Expected Count")
    machine_summary_table.add_column("Error Count")
    machine_summary_table.add_column("Pending Count")
    machine_summary_table.add_column("Ready Count")
    machine_summary_table.add_column("Connected (Ray) Count")

    machine_status_table = Table(show_header=True, header_style="bold cyan")
    machine_status_table.add_column("Machine ID")
    machine_status_table.add_column("Status")

    return machine_summary_table, machine_status_table


def update_ui(live, machine_ids, ray_count):
    machine_summary_table, machine_status_table = create_UI_tables()

    # Get machine specific statuses and update the status table
    for machine_id in machine_ids:
        machine_status = get_tpu_status(os.environ['ZONE'], machine_id)
        machine_status_table.add_row(machine_id, machine_status)

    # update the summary table
    machine_summary_table.add_row(
        str(DEFAULT_CHILD_NODE_COUNT),
        str(sum(1 for _ in machine_ids if get_tpu_status(
            os.environ['ZONE'], _) == 'ERROR')),
        str(sum(1 for _ in machine_ids if get_tpu_status(
            os.environ['ZONE'], _) == 'PENDING')),
        str(sum(1 for _ in machine_ids if get_tpu_status(
            os.environ['ZONE'], _) == 'RUNNING')),
        str(ray_count)
    )

    # Update the Live display
    live.update(Group(machine_summary_table, machine_status_table))

    return


def machine_down(machine_id):
    cmd = f"""
    gcloud alpha compute tpus tpu-vm delete {machine_id} - -zone {os.environ['ZONE']} - -project {os.environ['PROJECT_ID']}
    """
    run_command_on_gcloud(cmd, zone=os.environ['ZONE'], instance=machine_id)


def main():
    ray_context = start_ray_head_node()

    # Get head node's IP
    head_node_ip = ray_context['node_ip_address']

    live = Live(refresh_per_second=4)

    while True:
        ray_status = check_node_status()
        machine_ids = []
        print("status")
        print(ray_status)
        ray_count = len(ray_status)

        statuses = all_tpu_status(os.environ['ZONE'])
        machine_ids = [tpu.get('name') for tpu in statuses]
        tpu_count = len(machine_ids)

        for i, tpu in enumerate(statuses):
            print(tpu)
            machine_id = tpu.get('name')
            machine_status = tpu.get('status')

            print(machine_id)

            if machine_status == 'RUNNING':
                if DEFAULT_CHILD_NODE_COUNT > ray_count:
                    print("Starting ray on child")
                    start_ray_on_child(
                        head_node_ip, os.environ['ZONE'], machine_id)
            # if error destroy
            if machine_status == 'ERROR':
                print("Machine in error state")
                machine_down(machine_id)

        if DEFAULT_CHILD_NODE_COUNT > tpu_count:
            for i in range(DEFAULT_CHILD_NODE_COUNT - tpu_count):
                print("Launching node")
                launch_node()

        update_ui(live, statuses, ray_count)

        time.sleep(5)  # sleep for 5 seconds


if __name__ == '__main__':
    main()
