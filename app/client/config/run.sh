#!/bin/sh
# get last line logs file
LAST_ACCESS_LOG=$(date -d "$(echo -n $(cat /home/app/server-middleware/log/access.log | tail -1))" +%s)
NOW=$(date -d "@$(($(date +%s) - 900))") # now - 1 hour

# compare to now - 1 hour with the last entry in log file.
if [[ "$NOW" > "$LAST_ACCESS_LOG" ]] ;
then
  SERVERS_LIST=$(curl -H "X-Auth-Token:$SCW_AUTH_TOKEN" "https://api.scaleway.com/instance/v1/zones/$SCW_SERVER_ZONE/servers?name=$SCW_NAME_INSTANCE")
  SERVER_ID=$(echo -n $(echo $SERVERS_LIST | jq -r '.[] | .[] | .id'))
  STATE=$(echo -n $(echo $SERVERS_LIST | jq -r '.[] | .[] | .state'))

  if [[ "$STATE" == "running" ]] ;
  then
    curl -X POST -d '{"action":"poweroff"}' -H "X-Auth-Token:$SCW_AUTH_TOKEN" -H "Content-Type: application/json" "https://api.scaleway.com/instance/v1/zones/$SCW_SERVER_ZONE/servers/$SERVER_ID/action"
  fi
fi