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

DEFAULT_CHILD_NODE_COUNT = 10  # Expected child nodes count

compute = discovery.build('compute', 'v1')

MACHINE_IDS_DIR = "machine_ids"

def run_command(cmd, retries=5):
    while retries:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            return stdout
        else:
            print(f"Failed to run command: {cmd}, error: {stderr}, retrying...")
            retries -= 1
            time.sleep(5)
    raise Exception(f"Failed to run command: {cmd}, error: {stderr}")

def launch_node():
    cmd = """
    gcloud config set compute/zone $ZONE

    gcloud alpha compute tpus tpu-vm delete $TPU_NAME --zone ${ZONE} --project ${PROJECT_ID}

    gcloud alpha compute tpus queued-resources delete ${QUEUED_RESOURCE_ID} \
    --project ${PROJECT_ID} \
    --zone ${ZONE}

    gcloud alpha compute tpus queued-resources create ${QUEUED_RESOURCE_ID} \
    --project ${PROJECT_ID} \
    --node-id ${TPU_NAME} \
    --zone ${ZONE} \
    --accelerator-type ${ACCELERATOR_TYPE} \
    --runtime-version ${RUNTIME_VERSION}
    """
    return run_command(cmd)

def launch_node():
    unique_id = str(uuid.uuid4())
    os.environ['TPU_NAME'] = 'nbardy-tpuv5-lite-' + os.environ['NUM_CHIPS'] + '-' + unique_id
    os.environ['QUEUED_RESOURCE_ID'] = 'nbady-tpuv5-lite-queued-resource-v2-' + os.environ['ZONE'] + '-' + os.environ['NUM_CHIPS'] + '-' + unique_id

    with open(os.path.join(MACHINE_IDS_DIR, f"{unique_id}.txt"), 'w') as file:
        file.write(os.environ['TPU_NAME'] + "\n" + os.environ['QUEUED_RESOURCE_ID'])


def start_ray_head_node():
    cmd = """
    ray start --head --resources='{"TPU": 4}' --node-ip-address=$(curl -H Metadata-Flavor:Google http://metadata/computeMetadata/v1/instance/network-interfaces/0/ip)
    """
    return run_command(cmd)

def start_ray_child_node(head_node_ip):
    cmd = f"""
    ray start --address={head_node_ip}:6379 --resources='{"TPU": 4}' --node-ip-address=$(curl -H Metadata-Flavor:Google http://metadata/computeMetadata/v1/instance/network-interfaces/0/ip)
    """
    return run_command(cmd)

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



def check_node_status():
    cmd = "ray status"
    stdout = run_command(cmd)
    return json.loads(stdout)


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

def main():
    start_ray_head_node()

    while True:
        status = check_node_status()
        head_node_ip = status.get('node_ip_address')

        ray_count = len(status.get('nodes', []))
        gcp_count = get_node_count_gcp()

        if DEFAULT_CHILD_NODE_COUNT > ray_count:
            print(f"Child nodes in Ray cluster {ray_count} is less than expected {DEFAULT_CHILD_NODE_COUNT}. Launching nodes...")
            for _ in range(DEFAULT_CHILD_NODE_COUNT - ray_count):
                launch_node()
                start_ray_child_node(head_node_ip)
        elif DEFAULT_CHILD_NODE_COUNT > gcp_count:
            print(f"Child nodes on GCP {gcp_count} is less than expected {DEFAULT_CHILD_NODE_COUNT}. Launching nodes...")
            for _ in range(DEFAULT_CHILD_NODE_COUNT - gcp_count):
                launch_node()
                start_ray_child_node(head_node_ip)

        time.sleep(60)  # sleep for 60 seconds

if __name__ == '__main__':
    main()

