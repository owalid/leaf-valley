<template>
  <v-container>
    <v-row align="baseline">
      <v-col>
        <v-select
          v-model="modelSelected"
          label="Select a model"
          :items="models"
        />
      </v-col>
      <v-col>
        <v-file-input
          v-model="rawFiles"
          multiple
          label="Upload a file(s) (maximum 5)"
          accept="image/*"
          placeholder="No file chosen"
          @change="onChangeFileInput"
        />
      </v-col>
      <v-col>
        <v-btn
          @click="getPredictions"
        >
          Predict
        </v-btn>
      </v-col>
    </v-row>
    <v-row>
      <!-- <div v-for="result in results" :key="result.indexPayload">
        
      </div> -->
      <div v-if="rawFiles" class="d-flex align-items-center justify-space-around">
          <v-img width="20%" height="20%" :src="b64Files[0]" />

          <v-progress-linear
            indeterminate
            color="green darken-2"
          ></v-progress-linear>

          <v-img width="20%" height="20%" :src="b64Files[0]" />
      </div>
      <!-- <v-row v-if="rawFiles" align="center">
        <v-col>
        </v-col>
        <v-col cols="6">
          <v-progress-linear
            indeterminate
            color="green darken-2"
          ></v-progress-linear>
        </v-col>
        <v-col>
        </v-col>
      </v-row> -->
    </v-row>
  </v-container>
</template>

<script>
export default {
  name: 'IndexPage',
  async asyncData({$axios}) {
    const res = await $axios.get('/models/')
    const {result} = res.data
    return {
      models: result.models
    }
  },
  data() {
    return {
      modelSelected: null,
      rawFiles: null,
      b64Files: null,
      models: {},
      results: [],
      processingPrediction: false
    };
  },
  computed: {
    shouldDisableForm() {
      return !this.processingPrediction
    },
    payloads() {
      const result = [];

      this.b64Files.forEach(b64file => {
        result.push({ model: this.modelSelected, file: b64file })
      });
      return result;
    }
  },
  methods: {
    onChangeFileInput(newFiles) {
      const vm = this;
      this.b64Files = [];
      newFiles.forEach(async (file) => {
        const b64File = await vm.toBase64(file);
        // this.b64Files.push(b64File.split(',')[1]);
        this.b64Files.push(b64File);
      });
    },

  // transform files to base64
  toBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = error => reject(error);
    });
  }, 
  async sendPostAndAddToResults(payload, indexPayload) {
    try {
      const request = await this.$axios.post('/predict/', payload);
      const { prediction } = request.data;
      this.results.push({ ...prediction, indexPayload });
      this.results = this.results.sort((a, b) => a.indexPayload - b.indexPayload);
    } catch (error) {
      console.error(error);
    }
  },
  async getPredictions() {
      const promises = [];

      this.payloads.forEach((payload, indexPayload) => {
        promises.push(this.sendPostAndAddToResults(payload, indexPayload));
      });
      this.processingPrediction = true;
      await Promise.all(promises);
      this.processingPrediction = false;
    }
  }
}
</script>
<style lang="scss" scoped>
.horizontal-line {
  border: 2px solid #ccc;
  border-radius: 5px;
  width: 25%;
}
</style>