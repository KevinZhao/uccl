#!/bin/bash
# Verify 2 spot instances are colocated on the same EFA leaf/switch.
#
# Usage: verify_topology.sh <region> <inst1> <inst2>
#
# Exits 0 when the deepest NetworkNodes entry matches between the two
# instances (→ same leaf). Exits 1 otherwise; callers are expected to
# terminate both instances and retry in a different AZ.
set -euo pipefail

REGION="${1:?region}"
I1="${2:?instance-id 1}"
I2="${3:?instance-id 2}"

AWS_PAGER="" aws ec2 describe-instance-topology \
  --region "$REGION" --instance-ids "$I1" "$I2" \
  --query 'Instances[*].{id:InstanceId,topo:NetworkNodes}' \
  --output json > /tmp/topo.json

LEAF1=$(python3 -c "
import json
j = json.load(open('/tmp/topo.json'))
for x in j:
    if x['id'] == '$I1':
        print(x['topo'][-1])
        break
")
LEAF2=$(python3 -c "
import json
j = json.load(open('/tmp/topo.json'))
for x in j:
    if x['id'] == '$I2':
        print(x['topo'][-1])
        break
")

echo "[topo] $I1 leaf = $LEAF1"
echo "[topo] $I2 leaf = $LEAF2"

if [ -z "$LEAF1" ] || [ -z "$LEAF2" ]; then
  echo "[topo] ERROR: empty leaf id (instance not yet propagated)"
  exit 2
fi

if [ "$LEAF1" != "$LEAF2" ]; then
  echo "[topo] FAIL: different leaves — terminate and retry in another AZ"
  exit 1
fi

echo "[topo] OK: both instances on leaf $LEAF1"
exit 0
