<template>
  <v-row class="mt-10 pt-10" align="center">
    <v-col>
      <v-img
        contain
        :src="`data:image/png;base64,${result.source_img}`"
        class="ml-auto img-prediction"
      />
    </v-col>
    <v-col
      xl="8"
      md="6"
      sm="4"
      xs="2"
    > 
      <v-progress-linear
        v-bind="linearProgressProps"
      ></v-progress-linear>
    </v-col>
    <v-col>
      <!-- Loading -->
      <v-img
        v-if="state === 'LOADING'"
        :lazy-src="`data:image/png;base64,${result.source_img}`"
        class="img-prediction loading"
      >
        <template #default>
          <v-row
            class="fill-height ma-0"
            align="center"
            justify="center"
          >
            <v-progress-circular
              indeterminate
              color="light-blue darken-1"
            ></v-progress-circular>
          </v-row>
        </template>
      </v-img>

      <!-- Error -->
      <v-row v-else-if="state === 'ERROR'">
        <v-icon color="red darken-2" size="80px" class="ml-5">fa-times</v-icon>
      </v-row>

      <!-- Result -->
      <v-img
        v-else
        contain
        :src="`data:image/png;base64,${result.im_withoutbg_b64}`"
        class="img-prediction"
      />
    </v-col>
  </v-row>
</template>
<script>
export default {
  name: "RenderPredictionResult",
  props: {
    result: {
      type: Object,
      required: true
    }
  },
  computed: {
    linearProgressProps() {
      if (this.state === "LOADING") {
        return {
          stream: true,
          'buffer-value': 0,
          reverse: true,
          color:"green darken-1"
        };
      } else if (this.state === "ERROR") {
        return {
          value: 100,
          color: "red darken-2"
        };
      } else {
        return {
          value: 100,
          color: "green darken-1"
        };
      }
    },
    state() {
      if (!this.result || 'error' in this.result) {
        return 'ERROR'
      } else if (this.result && 'prediction' in this.result && 'accuracy' in this.result && 'im_withoutbg_b64' in this.result) {
        return 'SUCCESS'
      } else {
        return 'LOADING'
      }
    }
  }
}
</script>
<style lang="scss" scoped>

.img-prediction {
  max-width: 200px;
  height: 200px;
}

@media (max-width: 768px) {
  .img-prediction {
    max-width: 100px;
    height: 100px;
  }
}

@media (max-width: 576px) {
  .img-prediction {
    max-width: 70px;
    height: 70px;
  }
  .progress-prediction {
    width: 10px;
  }
}

.message-error {
  max-width: 150px;
}

.icon-error {
  width: 150px;
}
</style>