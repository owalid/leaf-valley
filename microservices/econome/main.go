package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    // "io"
    "os"
    "econome/structs"
    "github.com/gorilla/mux"
    "github.com/joho/godotenv"
)

func goDotEnvVariable(key string) string {
    err := godotenv.Load()
    
    if err != nil {
      log.Fatalf("Error loading .env file")
      return ""
    }
  
    return os.Getenv(key)
}

type isAliveResponse struct {
	IsAlive          bool `json: "isAlive"`
}

func respondWithJSON(w http.ResponseWriter, code int, payload interface{}) {
    response, _ := json.Marshal(payload)
    fmt.Println(string(response))
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(code)
    w.Write(response)
}

func clusterIsRunning() bool {
    serverId := goDotEnvVariable("SCW_SERVER_ID")
    zone := goDotEnvVariable("SCW_SERVER_ZONE")
    authToken := goDotEnvVariable("SCW_AUTH_TOKEN")
    
    url := "https://api.scaleway.com/instance/v1/zones/" + zone + "/servers/" + serverId

    client := &http.Client{}
    reqServerAlive, err := http.NewRequest("GET", url, nil)
    
    if err != nil {
        fmt.Println("first err")
        return false
    }
    
    reqServerAlive.Header.Set("X-Auth-Token", authToken)
    resServerAlive, err := client.Do(reqServerAlive)

    if err != nil {
        fmt.Println("second err")
        return false
    }

    // GET STATE FROM RESPONSE
    parsedResponse := scwResponses.ScwServerResponse{}
    json.NewDecoder(resServerAlive.Body).Decode(&parsedResponse)
    state := string(parsedResponse.Server.State)

    return state == "running"
}

func isAlive(w http.ResponseWriter, r *http.Request) {
	var res isAliveResponse
    res.IsAlive = clusterIsRunning()
    respondWithJSON(w, 200, res)
}

func main() {
	router := mux.NewRouter().StrictSlash(true)
	router.HandleFunc("/is-alive", isAlive)
	log.Fatal(http.ListenAndServe(":8080", router))
}
