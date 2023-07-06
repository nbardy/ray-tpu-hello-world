export PROJECT_ID=facet-250518
export NUM_CHIPS=16
export ACCELERATOR_TYPE=v5litepod-${NUM_CHIPS}
export ZONE=us-west4-a
export RUNTIME_VERSION=v2-alpha-tpuv5-lite
export TPU_NAME=nbardy-tpuv5-lite-${NUM_CHIPS}
#export NODE_COUNT=4
export QUEUED_RESOURCE_ID=nbady-tpuv5-lite-queued-resource-v2-${ZONE}-${NUM_CHIPS}

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
# --reserved
