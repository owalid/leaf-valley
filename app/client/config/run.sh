# get last line logs file
LAST_ACCESS_LOG=$(echo -n $(cat server-middleware/log/access.log | tail -1))

# compare to now if > 1 hour stop instance

