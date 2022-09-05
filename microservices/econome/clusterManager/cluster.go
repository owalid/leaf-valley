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
	State          string `json: "state"`
}

func getServerDetail() scwResponses.ScwServerResponse {
	zone := utils.GoDotEnvVariable("SCW_SERVER_ZONE")
	authToken := utils.GoDotEnvVariable("SCW_AUTH_TOKEN")
	var nameInstance = "instance-cluster-api-leaf"

	url := "https://api.scaleway.com/instance/v1/zones/" + zone + "/servers/?name=" + nameInstance

	client := &http.Client{}
	reqServerAlive, err := http.NewRequest("GET", url, nil)

	if err != nil {
		fmt.Println("first err")
		return nil
	}

	reqServerAlive.Header.Set("X-Auth-Token", authToken)
	resServerAlive, err := client.Do(reqServerAlive)

	parsedResponse := scwResponses.ScwListServerResponse{}
	json.NewDecoder(resServerAlive.Body).Decode(&parsedResponse)

	return parsedResponse[0]
}

func StartCluster() string {
    zone := utils.GoDotEnvVariable("SCW_SERVER_ZONE")
    authToken := utils.GoDotEnvVariable("SCW_AUTH_TOKEN")

	parsedResponse := scwResponses.ScwServerResponse{Server: getServerDetail()}
	serverId = string(parsedResponse.Server.id)

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
    serverId := utils.GoDotEnvVariable("SCW_SERVER_ID")
    zone := utils.GoDotEnvVariable("SCW_SERVER_ZONE")
    authToken := utils.GoDotEnvVariable("SCW_AUTH_TOKEN")

	parsedResponse := scwResponses.ScwServerResponse{Server: getServerDetail()}
	if parsedResponse == nil {
		return ""
	}

    state := string(parsedResponse.Server.State)
    return state
}
