<template>
  <v-container>
    <v-alert v-if="isValid" class="text-center" type="error">
      {{ errorMessage }}
    </v-alert>
    <v-row class="mt-2" justify="center">
      <v-col class="mx-3">
        <v-combobox
          v-model="selectedClass"
          label="Class"
          :items="classes"
          :disabled="predictionInProgress"
          clearable
          clear-icon="fas fa-times-circle"
          dense
        />
      </v-col>
      <v-col class="mx-3">
        <v-file-input
          v-model="selectdFile"
          label="Upload a file"
          accept="image/*"
          placeholder="No file chosen"
          show-size
          :disabled="predictionInProgress"
          :error-messages="selectdFileErrors"
          dense
          @change="onChangeFileInput"
        />
      </v-col>
      <v-col class="mx-8">
        <v-select
          v-model="dpModelSelected"
          label="DP model"
          :items="dpModels"
          :disabled="predictionInProgress"
          :error-messages="selectedModelsErrors"
          dense
        ></v-select>
      </v-col>
      <v-col class="mx-8">
        <v-select
          v-model="mlModelSelected"
          label="ML model"
          :items="mlModels"
          :disabled="predictionInProgress"
          :error-messages="selectedModelsErrors"
          dense
        ></v-select>
      </v-col>

      <v-col class="mx-8">
        <v-btn
          color="green lighten-1"
          :disabled="predictionInProgress"
          @click="getPredictions"
        >
          Predict
        </v-btn>
      </v-col>
    </v-row>

    <v-row v-if="isLoading" justify="center">
      <v-progress-circular
        indeterminate
        color="green lighten-1"
      ></v-progress-circular>
    </v-row>
    <v-row v-else justify="center" class="mt-15">
      <div v-for="result in results" :key="result.indexPayload">
        <display-images-and-prediction :result="result" />
      </div>
    </v-row>
  </v-container>
</template>

<script>
import { validationMixin } from 'vuelidate'
import { required, requiredIf } from 'vuelidate/lib/validators'
import DisplayImagesAndPrediction from '~/components/DisplayImagesAndPrediction'

export default {
  name: 'SelectPage',
  components: { DisplayImagesAndPrediction },
  mixins: [validationMixin],
  validations() {
    return {
      selectdFile: { required },
      mlModelSelected: {
        required: requiredIf(function () {
          return !this.dpModelSelected
        }),
      },
      dpModelSelected: {
        required: requiredIf(function () {
          return !this.mlModelSelected
        }),
      },
    }
  },
  async asyncData({ $axios }) {
    const resModels = await $axios.get('/models/')
    const resClasses = await $axios.get('/models/classes')
    return {
      classes: resClasses.data.result.classes,
      dpModels: resModels.data.result.models.DP,
      mlModels: resModels.data.result.models.ML,
    }
  },
  data() {
    return {
      results: [],
      b64Files: null,
      fileName: null,
      errorMessage: '',
      selectdFile: null,
      selectedClass: null,
      dpModelSelected: null,
      mlModelSelected: null,
      isValid: false,
      isLoading: false,
      predictionInProgress: false,
    }
  },
  computed: {
    selectdFileErrors() {
      const errors = []
      if (!this.$v.selectdFile.$dirty) {
        return errors
      }
      !this.$v.selectdFile.required && errors.push('Image file is required')
      return errors
    },
    selectedModelsErrors() {
      const errors = []
      if (!this.$v.mlModelSelected.$dirty && !this.$v.dpModelSelected.$dirty) {
        return errors
      }
      !this.mlModelSelected &&
        !this.dpModelSelected &&
        errors.push('You must select at least one ML or PL model')
      return errors
    },
  },
  methods: {
    async onChangeFileInput(file) {
      this.fileName = file.name
      const b64File = await this.toBase64(file)
      this.b64Files = b64File.split(',')[1]
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
    async getPredictions() {
      this.$v.$touch()
      if (!this.$v.$invalid) {
        try {
          this.isValid = false
          this.isLoading = true
          this.predictionInProgress = true
          const payload = {
            class_name: this.selectedClass
              ? this.selectedClass + '/' + this.fileName
              : null,
            b64Files: this.b64Files,
            ml_model: this.mlModelSelected,
            dp_model: this.dpModelSelected,
          }
          const request = await this.$axios.post('/models/select-img', payload)

          this.results = []
          Object.keys(request.data.result).forEach((key) => {
            this.results.push(request.data.result[key])
          })
          this.isLoading = false
          this.predictionInProgress = false
        } catch (error) {
          // eslint-disable-next-line no-console
          console.error(error)
        }
      } else {
        this.errorMessage =
          'Data selection is not valid. Please select the valid ones'
        this.isValid = true
        setInterval(() => {
          this.isValid = false
        }, 3000)
      }
    },
  },
}
</script>
