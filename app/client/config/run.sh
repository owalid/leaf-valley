# get last line logs file
LAST_ACCESS_LOG=$(date -d $(echo -n $(cat server-middleware/log/access.log | tail -1)) +%s)
NOW=$(date +"%s") - 3600

# compare to now if > 1 hour stop instance
if [[ "$NOW" > "$LAST_ACCESS_LOG" ]] ;
then
  RESULT_GET=$(curl -H "X-Auth-Token:$authToken" "https://api.scaleway.com/instance/v1/zones/$zone/servers/action?name=instance-cluster-api-leaf")
  SERVER_ID=$(echo $RESULT_GET | jq'[].server_id')
  STATE=$(echo $RESULT_GET | jq'[].state')

  if [[ "$STATE" != "running" ]] ;
  then
    curl -X POST -d "action=poweroff" -H "X-Auth-Token:$authToken" "https://api.scaleway.com/instance/v1/zones/$zone/servers/$SERVER_ID/action"
  fi
fi