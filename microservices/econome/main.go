package main

import (
    "log"
    "net/http"
    "encoding/json"
    "econome/ws"
    "econome/recaptcha"
    "econome/utils"
    "econome/clusterManager"
    "github.com/gorilla/mux"
    "github.com/rs/cors"
)

func startCluster(w http.ResponseWriter, r *http.Request) {
    var body recaptcha.RecaptchaRequest
    if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
        utils.RespondWithJSON(w, http.StatusBadRequest, cluster.StartClusterResponse{Status: "Bad request"})
        return
    }

    if err := recaptcha.CheckRecaptcha(body.RecaptchaResponse); err != nil {
        utils.RespondWithJSON(w, http.StatusUnauthorized, cluster.StartClusterResponse{Status: "Unauthorized"})
        return
    }

    var res = cluster.StartClusterResponse{Status: cluster.StartCluster()}

    if res.Status == "" {
        utils.RespondWithJSON(w, 500, cluster.StartClusterResponse{Status: "Error"})
    }

    utils.RespondWithJSON(w, 200, res)
}

func getClusterStatus(w http.ResponseWriter, r *http.Request) {
    var res = cluster.ClusterStatusResponse{State: cluster.GetStateCluster()}

    if res.State == "" {
        utils.RespondWithJSON(w, 500, res)
    }

    utils.RespondWithJSON(w, 200, res)
}

func main() {
    var goEnv = utils.GoDotEnvVariable("GO_ENV")
    var addr = ":8080"
    if goEnv == "production" {
        addr = "0.0.0.0:80"
    }
	router := mux.NewRouter().StrictSlash(true)
	router.HandleFunc("/econome/get-status", getClusterStatus)
	router.HandleFunc("/econome/start-cluster", startCluster).Methods("POST")
    router.HandleFunc("/econome/ws", websocketWorker.WsSubscriber)

    corsConfig := cors.New(cors.Options{
        AllowedOrigins: []string{"http://localhost:3000", "https://leaf-valley.com"},
        AllowCredentials: true,
    })
    handler := corsConfig.Handler(router)

    log.Println("listen on", addr)
	log.Fatal(http.ListenAndServe(addr, handler))
}
