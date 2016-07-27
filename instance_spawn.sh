#! /bin/bash
instances="instance-1"
while [[ $# -gt 1 ]]
do
key="$1"

case $key in
    -c|--cpu)
    cpu_count="$2"
    shift # past argument
    ;;
    -m|--memory)
    ram_size="$2"
    shift # past argument
    ;;
    -i|--instance_count)
    instance_count="$2"
    instance_count=${instance_count-1}
    shift # past argument
    ;;
    *)
            # unknown option
    ;;
esac
shift # past argument or value
done



for i in `seq 2 $instance_count`; do
      instances="$instances instance-$i"
done



ram_size=$ram_size"GiB"

gcloud compute instances create $instances \
	--image-family ubuntu-1404-lts \
	--image-project ubuntu-os-cloud \
	--custom-cpu $cpu_count \
	--custom-memory $ram_size \
	--zone us-east1-c

