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


def launch_node():
    # Make uuid for machine-id of 6 characters
    machine_id = run_command("uuidgen | cut -c1-6").decode('utf-8').strip()

    queued_resource_id = os.environ['QUEUED_RESOURCE_ID'] + machine_id
    tpu_name = os.environ['TPU_NAME'] + machine_id

    cmd = """
    gcloud config set compute/zone $ZONE

    gcloud alpha compute tpus tpu-vm delete $TPU_NAME --zone ${ZONE} --project ${PROJECT_ID}

    gcloud alpha compute tpus queued-resources delete ${queued_resource_id} \
    --project ${PROJECT_ID} \
    --zone ${ZONE}

    gcloud alpha compute tpus queued-resources create ${queued_resource_id} \
    --project ${PROJECT_ID} \
    --node-id ${tpu_name} \
    --zone ${ZONE} \
    --accelerator-type ${ACCELERATOR_TYPE} \
    --runtime-version ${RUNTIME_VERSION}
    """

    try:
        run_command(cmd)
    except Exception as err:
        print(f"Failed to launch node: {err}")
        return None

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
    ray start --address={head_node_ip}:6379 --resources='{"TPU": 4}' --node-ip-address=$(curl -H Metadata-Flavor:Google http://metadata/computeMetadata/v1/instance/network-interfaces/0/ip)
    """
    return run_command_on_gcloud(cmd, zone=zone, instance=instance)


def check_node_status():
    cmd = "ray status"
    stdout = run_command(cmd)
    return json.loads(stdout)


def get_machine_state(machine_name):
    # Check the machine status on GCP
    request = compute.instances().get(
        project=os.environ['PROJECT_ID'],
        zone=os.environ['ZONE'],
        instance=machine_name)
    try:
        response = request.execute()
        return response
    except HttpError as err:
        print(f"Failed to get machine state from GCP: {err}")
        return None


def get_node_count_gcp():
    request = compute.instances().list(
        project=os.environ['PROJECT_ID'],
        zone=os.environ['ZONE'])
    try:
        response = request.execute()
        return len(response['items'])
    except HttpError as err:
        print(f"Failed to get machine state from GCP: {err}")
        return 0


def get_instance_status(zone, instance):
    cmd = ["gcloud", "compute", "instances", "describe",
           instance, "--zone", zone, "--format=json"]

    result = subprocess.run(cmd, text=True, capture_output=True)

    # Check if the command was successful
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        return None

    # Parse the output as JSON
    instance_info = json.loads(result.stdout)

    # The status of the instance is stored in the 'status' field
    status = instance_info.get('status')
    if status:
        print(f"Instance status: {status}")
        return status
    else:
        print("Status not found in instance info.")
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


def main():
    ray_context = start_ray_head_node()
    machine_summary_table, machine_status_table = create_UI_tables()

    # Stores the last update time of each machine's status
    machine_status_update_times = {}

    with Live(Group(machine_summary_table, machine_status_table), refresh_per_second=4) as live:
        while True:
            # clear the table for fresh data
            machine_summary_table.rows.clear()
            machine_status_table.rows.clear()

            status = check_node_status()
            head_node_ip = status.get('node_ip_address')
            machine_ids = []

            ray_count = len(status.get('nodes', []))
            gcp_count = get_node_count_gcp()

            # Launch additional nodes if necessary
            if DEFAULT_CHILD_NODE_COUNT > ray_count:
                print(
                    f"Child nodes in Ray cluster {ray_count} is less than expected {DEFAULT_CHILD_NODE_COUNT}. Launching nodes...")
                for _ in range(DEFAULT_CHILD_NODE_COUNT - ray_count):
                    machine_id = launch_node()
                    machine_ids.append(machine_id)

            # Get machine specific statuses and update the status table
            for machine_id in machine_ids:
                now = time.time()
                last_update_time = machine_status_update_times.get(
                    machine_id, 0)
                # Only update the status if it was not updated in the last minute
                if now - last_update_time > 60:
                    machine_status = get_tpu_status(
                        os.environ['ZONE'], machine_id)
                    machine_status_update_times[machine_id] = now
                else:
                    machine_status = "Recently checked..."
                machine_status_table.add_row(machine_id, machine_status)

            # update the summary table
            machine_summary_table.add_row(str(DEFAULT_CHILD_NODE_COUNT),
                                          str(sum(1 for _ in machine_ids if get_tpu_status(
                                              os.environ['ZONE'], _) == 'ERROR')),
                                          str(sum(1 for _ in machine_ids if get_tpu_status(
                                              os.environ['ZONE'], _) == 'PENDING')),
                                          str(sum(1 for _ in machine_ids if get_tpu_status(
                                              os.environ['ZONE'], _) == 'RUNNING')),
                                          str(ray_count))

            # Update the Live display
            live.update(Group(machine_summary_table, machine_status_table))

            time.sleep(5)  # sleep for 5 seconds


if __name__ == '__main__':
    main()
