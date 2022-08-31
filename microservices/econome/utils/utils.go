package utils

import (
	"net/http"
    "encoding/json"
    "fmt"
    "log"
    "os"
    "github.com/joho/godotenv"
)

func GoDotEnvVariable(key string) string {
    err := godotenv.Load()
    
    if err != nil {
      log.Fatalf("Error loading .env file")
      return ""
    }
  
    return os.Getenv(key)
}

func RespondWithJSON(w http.ResponseWriter, code int, payload interface{}) {
    response, _ := json.Marshal(payload)
    fmt.Println(string(response))
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(code)
    w.Write(response)
}
