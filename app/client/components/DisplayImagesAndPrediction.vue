<template>
  <div class="pa-2">
    <v-card width="300">
      <v-col>
        <div class="text-subtitle-1 mb-1">
          Species : {{ result.img_species }}
        </div>
        <div class="text-caption text-truncate">
          Desease : {{ result.img_desease }}
        </div>
        <div class="text-caption">Img Num : {{ result.img_num }}</div>
      </v-col>
      <v-row class="ma-1" align="center">
        <v-img
          contain
          :src="`data:image/png;base64,${result.rgb_img}`"
          class="ml-auto img-prediction"
          width="130"
          height="130"
        />
        <v-img
          contain
          :src="`data:image/png;base64,${result.masked_img}`"
          class="ml-auto img-prediction"
          width="130"
          height="130"
        />
      </v-row>
      <v-col v-if="isDLModel">
        <div class="text-subtitle-1 mb-1">
          ===<span class="mx-12">DL classification</span>===
        </div>
        <div class="text-caption">
          Class predicted :
          <span :class="colorDLMatching">{{ result.dl_prediction.class }}</span>
        </div>
        <div class="text-caption">
          Score :
          <span :class="colorDLScore">{{ result.dl_prediction.score }}</span>
        </div>
      </v-col>
      <v-col v-if="isMLModel">
        <div class="text-subtitle-1 mb-1">
          ===<span class="mx-12">ML classification</span>===
        </div>
        <div class="text-caption">
          Class predicted :
          <span :class="colorMLMatching">{{ result.ml_prediction.class }}</span>
        </div>
        <div class="text-caption">
          Score :
          <span :class="colorMLScore">{{ result.ml_prediction.score }}</span>
        </div>
      </v-col>

      <v-row v-if="addCommentFlag" justify="end" class="ma-2 pb-3">
        <v-btn
          style="font-size: 8px; height: 24px"
          :disabled="result.img_num === null"
          @click="addCommentClick"
        >
          Add comment
        </v-btn>
      </v-row>
      <v-row v-else>
        <v-row v-if="showCommentFlag" justify="end" class="px-6 py-3">
          <v-col>
            <v-textarea
              v-model="commentText"
              :label="labelFunc"
              filled
              :success-messages="msgSuccess"
              :error-messages="msgError"
            ></v-textarea>
            <v-row justify="end" class="ma-0">
              <v-btn
                class="mr-1"
                color="green lighten-1"
                style="font-size: 8px; height: 24px"
                @click="insertCommentClick"
                >Submit</v-btn
              >
              <v-btn
                class="ml-1"
                color="red lighten-1"
                style="font-size: 8px; height: 24px"
                @click="cancelCommentClick"
                >Cancel</v-btn
              >
            </v-row>
          </v-col>
        </v-row>
        <v-row v-else class="mx-3">
          <v-col>
            <v-expansion-panels>
              <v-expansion-panel>
                <v-expansion-panel-header
                  class="pa-0"
                  color="grey--text text--darken-2"
                >
                  Comment
                  <template #actions>
                    <v-icon
                      class="fas fa-duotone fa-square"
                      color="grey darken-2"
                      size="17px"
                    >
                      $expand
                    </v-icon>
                  </template>
                </v-expansion-panel-header>
                <v-expansion-panel-content>
                  <v-row justify="end" class="pb-2 mx-n5">
                    <v-textarea
                      v-model="commentText"
                      :label="labelFunc"
                      filled
                      :rules="commentRules"
                      :success-messages="msgSuccess"
                      :error-messages="msgError"
                    ></v-textarea>

                    <v-btn
                      class="mr-1"
                      color="green lighten-1"
                      style="font-size: 8px; height: 24px"
                      @click="updateCommentClick"
                      >Update</v-btn
                    >
                    <v-btn
                      class="ml-1"
                      color="red lighten-1"
                      style="font-size: 8px; height: 24px"
                      @click="deleteCommentClick"
                      >Delete</v-btn
                    >
                  </v-row>
                </v-expansion-panel-content>
              </v-expansion-panel>
            </v-expansion-panels>
          </v-col>
        </v-row>
      </v-row>
    </v-card>
  </div>
</template>

<script>
export default {
  name: 'DisplayImagesAndPrediction',
  props: {
    result: {
      type: Object,
      required: true,
    },
  },
  data() {
    return {
      showCommentFlag: this.result.comment === '',
      addCommentFlag: this.result.comment === '',
      commentText: this.result.comment,
      msgError: [],
      msgSuccess: [],
      commentRules: [
        (v) =>
          (v && /^[0-9a-zA-Z]+$/.test(v)) || 'Comment must be alphanumeric',
      ],
    }
  },
  computed: {
    colorDLMatching() {
      if (Object.keys(this.result).includes('dl_prediction')) {
        return !this.result.img_num
          ? 'white--text'
          : this.result.dl_prediction.matching
          ? 'green--text text--lighten-2'
          : 'red--text text--lighten-2'
      } else {
        return 'white--text'
      }
    },
    colorMLMatching() {
      if (Object.keys(this.result).includes('ml_prediction')) {
        if (!this.result.img_num) {
          return 'white--text'
        } else if (this.result.ml_prediction.matching) {
          return 'green--text text--lighten-2'
        } else {
          return 'red--text text--lighten-2'
        }
      } else {
        return 'white--text'
      }
    },
    colorDLScore() {
      if (Object.keys(this.result).includes('dl_prediction')) {
        const val = this.result.dl_prediction.score
        if (val < 50 || !this.result.img_num) {
          return 'white--text'
        } else if (!this.result.dl_prediction.matching) {
          return 'red--text text--lighten-2'
        } else if (val < 70) {
          return 'lime--text text--accent-1'
        } else if (val < 90) {
          return 'light-green--text text--accent-2'
        } else {
          return 'green--text text--accent-4'
        }
      } else {
        return 'white--text'
      }
    },
    colorMLScore() {
      if (Object.keys(this.result).includes('ml_prediction')) {
        const val = this.result.ml_prediction.score
        if (val < 50 || !this.result.img_num) {
          return 'white--text'
        } else if (!this.result.ml_prediction.matching) {
          return 'red--text text--lighten-2'
        } else if (val < 70) {
          return 'lime--text text--accent-1'
        } else if (val < 90) {
          return 'light-green--text text--accent-2'
        } else {
          return 'green--text text--accent-4'
        }
      } else {
        return 'white--text'
      }
    },
    linearProgressProps() {
      if (this.state === 'LOADING') {
        return {
          stream: true,
          'buffer-value': 0,
          reverse: true,
          color: 'green darken-1',
        }
      } else if (this.state === 'ERROR') {
        return {
          value: 100,
          color: 'red darken-2',
        }
      } else {
        return {
          value: 100,
          color: 'green darken-1',
        }
      }
    },
    state() {
      if (!this.result || 'error' in this.result) {
        return 'ERROR'
      } else if (
        this.result &&
        'img_name' in this.result &&
        'rgb_img' in this.result &&
        'masked_img' in this.result
      ) {
        return 'SUCCESS'
      } else {
        return 'LOADING'
      }
    },
    isDLModel() {
      return Object.keys(this.result).includes('dl_prediction')
    },
    isMLModel() {
      return Object.keys(this.result).includes('ml_prediction')
    },
    labelFunc() {
      if (this.commentText === '') {
        return 'Add comment'
      } else {
        return ''
      }
    },
    matchingFunc() {
      return 'color:' + this.result.dl_prediction.matching
    },
  },
  methods: {
    addCommentClick() {
      this.addCommentFlag = false
      this.showCommentFlag = true
    },
    insertCommentClick() {
      this.addCommentFlag = false
      this.showCommentFlag = false
      this.processComment('insert')
    },
    updateCommentClick() {
      this.processComment('update')
    },
    deleteCommentClick() {
      this.processComment('delete')
      this.addCommentFlag = true
      this.commentText = ''
    },
    cancelCommentClick() {
      this.addCommentFlag = true
    },

    async processComment(method) {
      try {
        const payload = {
          method,
          comment: {
            species: this.result.img_species,
            desease: this.result.img_desease,
            img_num: this.result.img_num,
            comment: this.commentText,
          },
        }
        await this.$axios.post('/comment', payload)
        this.msgSuccess = ['Done with success']
        setInterval(() => {
          this.msgSuccess = []
        }, 3000)
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error(error)
        this.msgError = ['Done with error']
        const { result } = error.response.data
        this.$store.commit('SET_ALERTS', result)
      }
    },
  },
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
