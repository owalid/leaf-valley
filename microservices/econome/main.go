package main

import (
    "log"
    "net/http"
    "econome/ws"
    "econome/utils"
    "econome/clusterManager"
    "github.com/gorilla/mux"
    "github.com/rs/cors"
)

func getClusterStatus(w http.ResponseWriter, r *http.Request) {
    var res = cluster.ClusterStatusResponse{State: cluster.GetStateCluster()}

    if res.State == "" {
        utils.RespondWithJSON(w, 500, res)
    }

    utils.RespondWithJSON(w, 200, res)
}

func main() {
    addr := ":8080"
	router := mux.NewRouter().StrictSlash(true)
	router.HandleFunc("/econome/get-status", getClusterStatus)
    router.HandleFunc("/econome/ws", websocketWorker.WsSubscriber)

    corsConfig := cors.New(cors.Options{
        AllowedOrigins: []string{"http://localhost:3000"},
        AllowCredentials: true,
    })
    handler := corsConfig.Handler(router)

    log.Println("listen on", addr)
	log.Fatal(http.ListenAndServe(addr, handler))
}
