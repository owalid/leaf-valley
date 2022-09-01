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

func StartCluster() string {
    serverId := utils.GoDotEnvVariable("SCW_SERVER_ID")
    zone := utils.GoDotEnvVariable("SCW_SERVER_ZONE")
    authToken := utils.GoDotEnvVariable("SCW_AUTH_TOKEN")

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
    
    url := "https://api.scaleway.com/instance/v1/zones/" + zone + "/servers/" + serverId

    client := &http.Client{}
    reqServerAlive, err := http.NewRequest("GET", url, nil)
    
    if err != nil {
        fmt.Println("first err")
        return ""
    }
    
    reqServerAlive.Header.Set("X-Auth-Token", authToken)
    resServerAlive, err := client.Do(reqServerAlive)

    if err != nil {
        fmt.Println("second err")
        return ""
    }

    // GET STATE FROM RESPONSE
    parsedResponse := scwResponses.ScwServerResponse{}
    json.NewDecoder(resServerAlive.Body).Decode(&parsedResponse)
    state := string(parsedResponse.Server.State)

    return state
}
