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
  name: 'TestWs',
  layout: 'pending',
  middleware: 'cluster-status',
  data() {
    return {
      resultWs: 'lol',
      interval: null,
      recaptchaResponse: null,
      shouldValidateRecaptcha: false
    }
  },
  async mounted() {
    try {
      await this.$recaptcha.init()
    } catch (error) {
      console.error(error);
    } finally {
      this.socket = new WebSocket('ws://localhost:8080/econome/ws')
      this.runListenerWs()
    }
  },
  beforeDestroy() {
    this.interval = null
    this.$recaptcha.destroy()
  },
  methods: {
    runListenerWs() {
      if (!this.interval) {
        this.interval = setInterval(() => {
          this.socket.send('getStatus')
          this.socket.onmessage = ({ data }) => {
            if (data === 'stopped') {
              this.shouldValidateRecaptcha = true
              this.$recaptcha.execute('startCluster').then(async (res) => {
                const recaptchaResponse = res
                this.shouldValidateRecaptcha = false
                await this.$recaptcha.reset()
                this.$econome.post('/start-cluster', {'g-recaptcha-response': recaptchaResponse})
              })
            }
            if (data === 'running') {
              this.$router.push('/')
            }
            this.resultWs = data
          }
        }, 5000)
      }
    },
  },
}
</script>
