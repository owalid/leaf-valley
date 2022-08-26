<template>
  <v-container>
    <v-row align="baseline">
      <v-col cols="12" sm="12" md="6" lg="3">
        <v-select
          v-model="modelSelected"
          :disabled="processingPrediction"
          label="Select a model"
          :items="models"
          hint="Please note, machine learning models (ML-*) are unstable and took long to execute"
          persistent-hint
        />
      </v-col>
      <v-col cols="12" sm="12" md="6" lg="4">
        <v-file-input
          v-model="rawFiles"
          multiple
          label="Upload a file(s) (maximum 5)"
          accept="image/*"
          placeholder="No file chosen"
          prepend-icon="fas fa-file-image"
          :disabled="processingPrediction"
          @change="onChangeFileInput"
        />
      </v-col>
      <v-col cols="12" sm="12" md="6" lg="2">
        <v-switch v-model="shouldRemoveBg" label="Remove background ?" />
      </v-col>
      <v-col cols="12" sm="12" md="6" lg="2">
        <v-btn
          color="green lighten-1"
          :disabled="processingPrediction"
          @click="getPredictions"
        >
          Predict
        </v-btn>
      </v-col>
    </v-row>
    <v-row v-for="result in results" :key="result.indexPayload">
      <render-prediction-result :result="result" />
    </v-row>
  </v-container>
</template>

<script>
import RenderPredictionResult from '~/components/RenderPredictionResult'

export default {
  name: 'IndexPage',
  components: { RenderPredictionResult },
  async asyncData({ $axios }) {
    const res = await $axios.get('/models/')
    const { result } = res.data
    return {
      models: [...result.models.DL, ...result.models.ML],
    }
  },
  data() {
    return {
      shouldRemoveBg: false,
      modelSelected: null,
      rawFiles: [],
      b64Files: null,
      models: {},
      results: [],
      processingPrediction: false,
    }
  },
  computed: {
    payloads() {
      const result = []

      this.b64Files.forEach((b64file) => {
        result.push({
          model_name: this.modelSelected,
          img: b64file,
          should_remove_bg: this.shouldRemoveBg,
        })
      })
      return result
    },
  },
  methods: {
    async onChangeFileInput(newFiles) {
      newFiles = newFiles.slice(0, 5)
      const vm = this
      this.b64Files = []
      await Promise.all(
        newFiles.map(async (file) => {
          const b64File = await vm.toBase64(file)
          this.b64Files.push(b64File.split(',')[1])
        })
      )
    },

    // Transform files to base64
    toBase64(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader()
        reader.readAsDataURL(file)
        reader.onload = () => resolve(reader.result)
        reader.onerror = (error) => reject(error)
      })
    },
    async sendPostAndAddToResults(payload, indexPayload) {
      try {
        this.results.push({ indexPayload, source_img: payload.img })
        const request = await this.$axios.post('/models/predict', payload)
        const prediction = request.data.result

        this.results.forEach((result, indexResult) => {
          if (result.indexPayload === indexPayload) {
            this.results[indexResult] = {
              ...this.results[indexResult],
              ...prediction,
            }
          }
        })
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error(error)
        this.results.forEach((result, indexResult) => {
          if (result.indexPayload === indexPayload) {
            const { result } = error.response.data
            this.$store.dispatch('ACTION_SET_ALERT', result.error)
            this.results[indexResult] = {
              ...this.results[indexResult],
              error: result,
            }
          }
        })
      } finally {
        // We sort because promises are async and its exucted in parallel
        this.results.sort((a, b) => a.indexPayload - b.indexPayload)
      }
    },
    async getPredictions() {
      this.results = []
      this.processingPrediction = true

      // Post to server each images in parallel
      try {
        await Promise.all(
          this.payloads.map((payload, indexPayload) =>
            this.sendPostAndAddToResults(payload, indexPayload)
          )
        )
      } catch (error) {
        const { result } = error.response.data
        let errorMessage = 'Unknow error'
        if ('error' in result) {
          errorMessage = result.error
        }
        this.$store.dispatch('ACTION_SET_ALERT', errorMessage)
      } finally {
        this.processingPrediction = false
      }
    },
  },
}
</script>
<style lang="scss" scoped>
.horizontal-line {
  border: 2px solid #ccc;
  border-radius: 5px;
  width: 25%;
}
</style>
