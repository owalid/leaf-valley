# get last line logs file
LAST_ACCESS_LOG=$(date -d $(echo -n $(cat server-middleware/log/access.log | tail -1)) +%s)
NOW=$(date +"%s") - 3600

# compare to now - 1 hour with the last entry in log file.
if [[ "$NOW" > "$LAST_ACCESS_LOG" ]] ;
then
  SERVERS_LIST=$(curl -H "X-Auth-Token:$SCW_AUTH_TOKEN" "https://api.scaleway.com/instance/v1/zones/$ZONE/servers?name=scw-leaf-api-cluster")
  SERVER_ID=$(echo -n $(echo $SERVERS_LIST | jq -r '.[] | .[] | .id'))
  STATE=$(echo -n $(echo $SERVERS_LIST | jq -r '.[] | .[] | .state'))

  if [[ "$STATE" == "running" ]] ;
  then
    curl -X POST -d "action=poweroff" -H "X-Auth-Token:$AUTH_TOKEN" "https://api.scaleway.com/instance/v1/zones/$ZONE/servers/$SERVER_ID/action"
  fi
fi