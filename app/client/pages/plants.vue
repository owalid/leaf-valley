<template>
  <v-container>
    <v-alert v-if="isValid" class="text-center" type="error">
      {{ errorMessage }}
    </v-alert>
    <v-row class="mt-2">
      <v-col cols="12" sm="12" md="6" lg="2">
        <v-select
          v-model="selectedSpecies"
          label="Species"
          :items="species"
          :error-messages="selectedSpeciesErrors"
          :disabled="predictionInProgress"
          dense
          @change="changeSpecies"
        />
      </v-col>
      <v-col cols="12" sm="12" md="6" lg="2">
        <v-select
          v-model="selectedDesease"
          label="Deseases"
          :items="plants[selectedSpecies]"
          :disabled="predictionInProgress"
          dense
          :error-messages="selectedDeseaseErrors"
        ></v-select>
      </v-col>
      <v-col cols="12" sm="12" md="6" lg="2">
        <v-select
          v-model="dlModelSelected"
          label="DL model"
          :items="dlModels"
          :disabled="predictionInProgress"
          :error-messages="selectedModelsErrors"
          dense
        ></v-select>
      </v-col>
      <v-col cols="12" sm="12" md="6" lg="2">
        <v-select
          v-model="mlModelSelected"
          label="ML model"
          :disabled="predictionInProgress"
          :items="mlModels"
          :error-messages="selectedModelsErrors"
          hint="Please note, machine learning models (ML-*) are unstable and took long to execute"
          persistent-hint
          dense
        ></v-select>
      </v-col>
      <v-col
        class="ml-8 pa-0"
        :class="[{ 'mt-5': $vuetify.breakpoint.mdAndDown }]"
        sm="1"
      >
        <v-subheader class="pt-1">Number of images</v-subheader>
      </v-col>
      <v-col class="pl-0" :class="[{ 'mt-5': $vuetify.breakpoint.mdAndDown }]">
        <v-card-text class="pl-0">
          <v-slider
            v-model="selectedSlider"
            max="25"
            min="0"
            step="5"
            thumb-color="green lighten-1"
            color="green lighten-1"
            thumb-label="always"
            :disabled="predictionInProgress"
            :error-messages="selectedSliderErrors"
          >
          </v-slider>
        </v-card-text>
      </v-col>

      <v-col class="mx-8" :class="[{ 'mt-5': $vuetify.breakpoint.mdAndDown }]">
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
    <v-row v-else>
      <v-col
        v-for="result in results"
        :key="result.indexPayload"
        align="center"
      >
        <display-images-and-prediction :result="result" />
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import { validationMixin } from 'vuelidate'
import { required, requiredIf } from 'vuelidate/lib/validators'
import DisplayImagesAndPrediction from '~/components/DisplayImagesAndPrediction'

export default {
  name: 'PlantsPage',
  components: { DisplayImagesAndPrediction },
  mixins: [validationMixin],
  validations() {
    return {
      selectedSpecies: { required },
      selectedDesease: { required },
      selectedSlider: { required },
      mlModelSelected: {
        required: requiredIf(function () {
          return !this.dlModelSelected
        }),
      },
      dlModelSelected: {
        required: requiredIf(function () {
          return !this.mlModelSelected
        }),
      },
    }
  },
  async asyncData({ $axios }) {
    const resModels = await $axios.get('/models/')
    const resPlants = await $axios.get('/models/plants')
    const species = Object.keys(resPlants.data.result.plants)
    return {
      species,
      plants: resPlants.data.result.plants,
      dlModels: resModels.data.result.models.DL,
      mlModels: resModels.data.result.models.ML,
    }
  },
  data() {
    return {
      selectedSlider: 5,
      selectedSpecies: null,
      selectedDesease: null,
      dlModelSelected: null,
      mlModelSelected: null,
      dlModels: [],
      mlModels: [],
      results: [],
      errorMessage: '',
      isValid: false,
      isLoading: false,
      predictionInProgress: false,
    }
  },
  computed: {
    selectedSpeciesErrors() {
      const errors = []
      if (!this.$v.selectedSpecies.$dirty) {
        return errors
      }
      !this.$v.selectedSpecies.required &&
        errors.push('Spicies field is required')
      return errors
    },
    selectedDeseaseErrors() {
      const errors = []
      if (!this.$v.selectedDesease.$dirty) {
        return errors
      }
      !this.$v.selectedDesease.required &&
        errors.push('Desease field is required')
      return errors
    },
    selectedSliderErrors() {
      const errors = []
      if (!this.$v.selectedSlider.$dirty) {
        return errors
      }
      ;(!this.$v.selectedSlider.required || this.selectedSlider <= 0) &&
        errors.push('Number of image is required and should be positive')
      return errors
    },
    selectedModelsErrors() {
      const errors = []
      if (!this.$v.mlModelSelected.$dirty && !this.$v.dlModelSelected.$dirty) {
        return errors
      }
      !this.mlModelSelected &&
        !this.dlModelSelected &&
        errors.push('You must select at least one ML or PL model')
      return errors
    },
  },
  methods: {
    changeSpecies() {
      this.selectedDesease = this.plants[this.selectedSpecies][0]
    },
    async getPredictions() {
      this.$v.$touch()
      if (!this.$v.$invalid) {
        try {
          this.isValid = false
          this.isLoading = true
          this.predictionInProgress = true
          const payload = {
            number_img: this.selectedSlider,
            spacies: this.selectedSpecies,
            desease: this.selectedDesease,
            ml_model: this.mlModelSelected,
            dl_model: this.dlModelSelected,
          }
          const request = await this.$axios.post('/models/random-img', payload)

          this.results = request.data.result.result_list
          this.isLoading = false
          this.predictionInProgress = false
        } catch (error) {
          // eslint-disable-next-line no-console
          console.error(error)
          const { result } = error.response.data
          let errorMessage = 'Unknow error'
          if ('error' in result) {
            errorMessage = result.error
          }
          this.$store.dispatch('ACTION_SET_ALERT', errorMessage)
        } finally {
          this.isLoading = false
          this.predictionInProgress = false
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
