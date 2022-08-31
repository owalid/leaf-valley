<template>
  <v-container fill-height>
    <v-col justify="center" align="center">
      <v-row justify="center">
        <v-progress-circular
        indeterminate
        color="light-green darken-1"
        :size="$vuetify.breakpoint.mdAndUp ? 350 : 150"
        width="1"
        >
        </v-progress-circular>
      </v-row>
      <v-row justify="center">
        <h2 class="mt-10">The api cluster is starting up</h2>
      </v-row>
      <v-row justify="center">
        <p class="">Please wait few minutes</p>
      </v-row>
    </v-col>
  </v-container>
</template>
<script>
  export default {
    name: "TestWs",
    layout: 'pending',
    middleware: 'cluster-status',
    data() {
      return {
        resultWs: 'lol',
        interval: null
      }
    },
    mounted() {
      this.socket = new WebSocket("ws://localhost:8080/econome/ws");
      this.socket.onopen = function () {
        console.log("Status: Connected\n");
      };
      // this.socket.send("send")
      this.getStatus()
    },
    beforeDestroy() {
      this.interval = null
    },
    methods: {
      getStatus() {
        if (!this.interval) {
          this.interval = setInterval(() => {
            this.socket.send("getStatus")
            this.socket.onmessage = ({data}) => {
              console.log("onmessage:", data)
              if (data === 'running') {
                this.$router.push('/')
              }
              this.resultWs = data
            };
          }, 5000)
        }
      }
    },
  }
</script>
