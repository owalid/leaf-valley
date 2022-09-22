package cluster

import (
	"bufio"
	"net/http"
	"bytes"
	"time"
	"os"
	"encoding/json"
	"econome/utils"
	"econome/structs"
)

type StartClusterResponse struct {
	Status      string    `json:"status"`
}

type ClusterStatusResponse struct {
	State       string `json: "state"`
}


func insertNowInFileLastStarting() {
	now := time.Now()
	file, err := os.Create("last_starting_cluster.txt")
	file.Truncate(0)

    if err != nil {
    }

    defer file.Close()
    _, err = file.WriteString(now.Format(time.RFC3339))

    if err != nil {
    }
}


func getLastRestart() (string, error) {
	file, err := os.Open("last_starting_cluster.txt")

    if err != nil {
        return "", err
    }

    defer file.Close()

    scanner := bufio.NewScanner(file)
	var result = ""
    for scanner.Scan() {
		result = scanner.Text()
    }

	if result == "" {
		insertNowInFileLastStarting()
	}
	return result, nil
}

func getNodeAvailability() (string, error) {
	clusterZone := utils.GoDotEnvVariable("SCW_CLUSTER_ZONE")
	authToken := utils.GoDotEnvVariable("SCW_AUTH_TOKEN")
	clusterId := utils.GoDotEnvVariable("SCW_CLUSTER_ID")
	

	url := "https://api.scaleway.com/k8s/v1/regions/" + clusterZone + "/clusters/" + clusterId + "/nodes"

	client := &http.Client{}
	reqServerAlive, err := http.NewRequest("GET", url, nil)

	if err != nil {
		return "", err
	}

	reqServerAlive.Header.Set("X-Auth-Token", authToken)
	resServerAlive, err := client.Do(reqServerAlive)

	if err != nil {
		return "", err
	}

	var parsedResponse = scwResponses.ScwNodesResponse{}
	json.NewDecoder(resServerAlive.Body).Decode(&parsedResponse)

	if len(parsedResponse.Nodes) == 0 {
		return "", nil
	}
	var finalResult = parsedResponse.Nodes[0]
	
	return finalResult.Status, nil
}

func getServerDetail() (scwResponses.ScwServerResponse, error) {
	zone := utils.GoDotEnvVariable("SCW_SERVER_ZONE")
	authToken := utils.GoDotEnvVariable("SCW_AUTH_TOKEN")
	nameInstance := utils.GoDotEnvVariable("SCW_NAME_INSTANCE")

	var defaultParsedResult = scwResponses.ScwServerResponse{}
	url := "https://api.scaleway.com/instance/v1/zones/" + zone + "/servers?name=" + nameInstance

	client := &http.Client{}
	reqServerAlive, err := http.NewRequest("GET", url, nil)

	if err != nil {
		return defaultParsedResult, err
	}

	reqServerAlive.Header.Set("X-Auth-Token", authToken)
	resServerAlive, err := client.Do(reqServerAlive)

	if err != nil {
		return defaultParsedResult, err
	}

	var parsedResponse = scwResponses.ScwListServerResponse{}
	json.NewDecoder(resServerAlive.Body).Decode(&parsedResponse)

	if len(parsedResponse.Servers) == 0 {
		return defaultParsedResult, nil
	}

	finalResult := scwResponses.ScwServerResponse{Server: parsedResponse.Servers[0]}

	return finalResult, nil
}

func StartCluster() string {
    zone := utils.GoDotEnvVariable("SCW_SERVER_ZONE")
    authToken := utils.GoDotEnvVariable("SCW_AUTH_TOKEN")

	var parsedResponse, err = getServerDetail()

	if err != nil {
		return ""
	}

	var serverId = string(parsedResponse.Server.ID)

    url := "https://api.scaleway.com/instance/v1/zones/" + zone + "/servers/" + serverId + "/action"

    var body = []byte(`{"action":"poweron"}`)
    reqServerAlive, err := http.NewRequest("POST", url, bytes.NewBuffer(body))

    if err != nil {
        return ""
    }

    reqServerAlive.Header.Set("X-Auth-Token", authToken)
    reqServerAlive.Header.Set("Content-Type", "application/json")

    client := &http.Client{}
    _, err = client.Do(reqServerAlive)

    if err != nil {
        return ""
    }

	insertNowInFileLastStarting()

    return "OK"
}

func GetStateCluster() string {
	var parsedResponse, err = getServerDetail()
	if err != nil {
		return ""
	}

	stateCluster := string(parsedResponse.Server.State)

	if stateCluster != "running" {
		return stateCluster
	}

	lastRestartStr, err := getLastRestart()
	if err != nil {
		return ""
	}

	now := time.Now()
	lastRestartDate, err := time.Parse(time.RFC3339, lastRestartStr)
	var diffRestartDate = now.Sub(lastRestartDate).Seconds()

	if err != nil {
		return ""
	}

	if diffRestartDate < 120 {
		return "node_not_ready"
	}

	stateNode, err := getNodeAvailability()

	if err != nil {
		return ""
	}

	if stateNode == "ready" {
		return "running"
	}

	return stateNode
}
