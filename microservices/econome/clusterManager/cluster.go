package cluster

import (
	"net/http"
	"bytes"
	"fmt"
	"time"
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

	fmt.Println("parsedResponse: %v\n", parsedResponse)
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

    return "OK"
}

func GetStateCluster() string {
	var parsedResponse, err = getServerDetail()
	if err != nil {
		return ""
	}

	var now = time.Now().Unix()

	creationDate, err := time.Parse(time.RFC3339, parsedResponse.Server.Image.CreationDate)
	modificationDate, err := time.Parse(time.RFC3339, parsedResponse.Server.Image.ModificationDate)

	fmt.Println(now)
	fmt.Println(creationDate.Unix())
	fmt.Println(modificationDate.Unix())

	if (now - creationDate.Unix() < 600 || now - modificationDate.Unix() < 600) {
		return "starting"
	}

    state := string(parsedResponse.Server.State)
    return state
}
