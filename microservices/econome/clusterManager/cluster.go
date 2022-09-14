package cluster

import (
	"net/http"
	"bytes"
	"fmt"
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


func getNodeAvailability() (string, error) {
	clusterZone := utils.GoDotEnvVariable("SCW_CLUSTER_ZONE")
	authToken := utils.GoDotEnvVariable("SCW_AUTH_TOKEN")
	clusterId := utils.GoDotEnvVariable("SCW_CLUSTER_ID")
	

	url := "https://api.scaleway.com/k8s/v1/regions/" + clusterZone + "/clusters/" + clusterId + "/nodes"

	client := &http.Client{}
	reqServerAlive, err := http.NewRequest("GET", url, nil)

	if err != nil {
		fmt.Println("err 2")
		fmt.Println("first err")
		return "", err
	}

	reqServerAlive.Header.Set("X-Auth-Token", authToken)
	resServerAlive, err := client.Do(reqServerAlive)

	if err != nil {
		fmt.Println("err 1")
		return "", err
	}

	var parsedResponse = scwResponses.ScwNodesResponse{}
	json.NewDecoder(resServerAlive.Body).Decode(&parsedResponse)

	fmt.Println("%v\n", parsedResponse)

	if len(parsedResponse.Nodes) == 0 {
		fmt.Println("no Nodes")
		return "", nil
	}
	var finalResult = parsedResponse.Nodes[0]
	
	fmt.Println("Node availability: ", finalResult.Status)
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
		fmt.Println("first err")
		return defaultParsedResult, err
	}

	reqServerAlive.Header.Set("X-Auth-Token", authToken)
	resServerAlive, err := client.Do(reqServerAlive)

	if err != nil {
		fmt.Println("second err")
		return defaultParsedResult, err
	}

	var parsedResponse = scwResponses.ScwListServerResponse{}
	json.NewDecoder(resServerAlive.Body).Decode(&parsedResponse)

	if len(parsedResponse.Servers) == 0 {
		fmt.Println("second err")
		return defaultParsedResult, nil
	}

	finalResult := scwResponses.ScwServerResponse{Server: parsedResponse.Servers[0]}
	fmt.Println("Instance availability: ", finalResult.Server.State)

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

    return "OK"
}

func GetStateCluster() string {
	var parsedResponse, err = getServerDetail()
	if err != nil {
		return ""
	}

    stateCluster := string(parsedResponse.Server.State)
	fmt.Println("stateCluster: ", stateCluster)
	if stateCluster != "running" {
		return stateCluster
	}
	
	stateNode, err := getNodeAvailability()
	fmt.Println("stateNode: ", stateNode)
	fmt.Println(err)
	if err != nil {
		return ""
	}

	if stateNode == "ready" {
		return "running"
	}
	return stateNode
}
