import Vue from 'vue'
import Vuetify, {
  VIcon,
  VBtn,
  VContainer,
  VProgressLinear,
  VProgressCircular,
  VImg
} from 'vuetify/lib'
import '@fortawesome/fontawesome-free/css/all.css' // Ensure you are using css-loader


Vue.use(Vuetify, {
  components: {
    VIcon,
    VBtn,
    VContainer,
    VProgressLinear,
    VProgressCircular,
    VImg
  },
  icons: {
    iconfont: 'fa'
  },
  options: {
    minifyTheme (css) {
      return css.replace(/[\s|\r\n|\r|\n]/g, '')
    }
  }
})