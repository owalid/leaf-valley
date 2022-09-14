package cluster

import (
	"net/http"
	"bytes"
	"fmt"
	"time"
	"strings"
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
	zone := utils.GoDotEnvVariable("SCW_SERVER_ZONE")
	authToken := utils.GoDotEnvVariable("SCW_AUTH_TOKEN")
	clusterId := utils.GoDotEnvVariable("SCW_CLUSTER_ID")
	
	url := "https://api.scaleway.com/k8s/v1/regions/" + zone + "/clusters/" + clusterId + "/nodes"

	client := &http.Client{}
	reqServerAlive, err := http.NewRequest("GET", url, nil)

	if err != nil {
		fmt.Println("first err")
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
		return defaultParsedResult, err
	}

	var parsedResponse = scwResponses.ScwListServerResponse{}
	json.NewDecoder(resServerAlive.Body).Decode(&parsedResponse)

	if len(parsedResponse.Servers) == 0 {
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

	parisTz, err := time.LoadLocation("Europe/Paris")
	if err != nil {
		return ""
	}

	now := time.Now()

	creationDateString := strings.Replace(parsedResponse.Server.CreationDate, "00:00", "02:00" , 1)
	modificationDateString := strings.Replace(parsedResponse.Server.ModificationDate, "00:00", "02:00" , 1)

	creationDate, err := time.ParseInLocation(time.RFC3339, creationDateString, parisTz)
	if err != nil {
		return ""
	}

	modificationDate, err := time.ParseInLocation(time.RFC3339, modificationDateString, parisTz)
	if err != nil {
		return ""
	}

	diffCreation := now.Sub(creationDate)
	diffModification := now.Sub(modificationDate)

	fmt.Println(creationDate)
	fmt.Println(modificationDate)
	fmt.Println(now)
	fmt.Println(diffCreation.Seconds())
	fmt.Println(diffModification.Seconds())

	if (int(diffCreation.Seconds()) < 600 || int(diffModification.Seconds()) < 600) {
		return "starting"
	}

    stateCluster := string(parsedResponse.Server.State)
	if stateCluster != "running" {
		return stateCluster
	}
	
	stateNode, err := getNodeAvailability()
	if err != nil {
		return ""
	}

	return stateNode
}
