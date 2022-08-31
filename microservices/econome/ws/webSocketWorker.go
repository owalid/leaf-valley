package websocketWorker

import (
	"log"
    "net/http"
    "econome/clusterManager"
	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true	
	},
}

func WsSubscriber(w http.ResponseWriter, r *http.Request) {
    c, err := upgrader.Upgrade(w, r, nil)
    if err != nil {
        log.Println(w, err, http.StatusInternalServerError)
        return
    }
    defer c.Close()
    for {
        var state = cluster.GetStateCluster()
        mt, _, err := c.ReadMessage()
        if err != nil {
            log.Println(w, err, http.StatusInternalServerError)
            break
        }
        if mt != websocket.TextMessage {
            log.Println(w, "Only text message are supported", http.StatusNotImplemented)
            break
        }
        err = c.WriteMessage(websocket.TextMessage, []byte(state))
        if err != nil {
            log.Println(w, err, http.StatusInternalServerError)
            break
        }
    }
}
