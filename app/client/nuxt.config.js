import colors from 'vuetify/es5/util/colors'

export default {
  // Global page headers: https://go.nuxtjs.dev/config-head
  head: {
    titleTemplate: '%s - vic2i',
    title: 'vic2i',
    meta: [
      { charset: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      { hid: 'description', name: 'description', content: '' },
      { name: 'format-detection', content: 'telephone=no' },
    ],
    link: [{ rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' }],
  },

  // Global CSS: https://go.nuxtjs.dev/config-css
  css: [],
  serverMiddleware: ['~/server-middleware/morgan'],

  // Plugins to run before rendering page: https://go.nuxtjs.dev/config-plugins
  plugins: [
    '~/plugins/vuetify',
    '~/plugins/axios',
    '~/plugins/axios-instances/api.js',
    '~/plugins/axios-instances/econome.js',
  ],

  // Auto import components: https://go.nuxtjs.dev/config-components
  components: true,

  // Modules for dev and build (recommended): https://go.nuxtjs.dev/config-modules
  buildModules: [
    // https://go.nuxtjs.dev/eslint
    '@nuxtjs/eslint-module',
    // https://go.nuxtjs.dev/stylelint
    '@nuxtjs/stylelint-module',
    // https://go.nuxtjs.dev/vuetify
    '@nuxtjs/vuetify',
  ],

  // Modules: https://go.nuxtjs.dev/config-modules
  modules: [
    // https://go.nuxtjs.dev/axios
    '@nuxtjs/axios',
    // https://go.nuxtjs.dev/pwa
    '@nuxtjs/pwa',
    '@nuxtjs/proxy',
    ...(process.env.NODE_ENV === 'production' ? ['@nuxtjs/recaptcha'] : []),
  ],

  proxy: {
    '/api': {
      target: process.env.NUXT_BASE_API_URL || 'http://127.0.0.1:5000/api',
      pathRewrite: {
        '^/api': '/api',
      },
    },
    '/econome': {
      target:
        process.env.NUXT_ECONOME_MS_URL || 'http://127.0.0.1:8080/econome',
      pathRewrite: {
        '^/econome': '/econome',
      },
    },
  },

  publicRuntimeConfig: {
    NODE_ENV: process.env.NODE_ENV,
    NUXT_BASE_API_URL: process.env.NUXT_BASE_API_URL,
    NUXT_ECONOME_MS_URL: process.env.NUXT_ECONOME_MS_URL,
    NUXT_ECONOME_MS_WS: process.env.NUXT_ECONOME_MS_WS,
    recaptcha: {
      hideBadge: false,
      mode: 'base',
      version: 3,
      siteKey: process.env.NUXT_RECAPTCHA_SITE_KEY,
    },
  },

  recaptcha: {
    hideBadge: false,
    mode: 'base',
    version: 3,
    siteKey: process.env.NUXT_RECAPTCHA_SITE_KEY,
  },

  // Axios module configuration: https://go.nuxtjs.dev/config-axios
  axios: {
    // Workaround to avoid enforcing hard-coded localhost:3000: https://github.com/nuxt-community/axios-module/issues/308
    baseURL: process.env.NUXT_BASE_API_URL || 'http://127.0.0.1:5000/api',
    // proxy: true
  },

  // PWA module configuration: https://go.nuxtjs.dev/pwa
  pwa: {
    manifest: {
      lang: 'en',
    },
  },

  // Vuetify module configuration: https://go.nuxtjs.dev/config-vuetify
  vuetify: {
    treeShake: true,
    customVariables: ['~/assets/variables.scss'],
    defaultAssets: false,
    theme: {
      dark: true,
      themes: {
        dark: {
          primary: colors.blue.darken2,
          accent: colors.grey.darken3,
          secondary: colors.amber.darken3,
          info: colors.teal.lighten1,
          warning: colors.amber.base,
          error: colors.deepOrange.accent4,
          success: colors.green.accent3,
        },
      },
    },
  },

  // Build Configuration: https://go.nuxtjs.dev/config-build
  build: {
    transpile: [/^vuetify/],
    extractCSS: true,
    standalone: true,
    ignoreOrder: false,
    filenames: {
      chunk: ({ isDev }) => (isDev ? '[name].js' : '[id].[contenthash].js'),
    },
    html: {
      minify: {
        collapseBooleanAttributes: true,
        decodeEntities: true,
        minifyCSS: true,
        minifyJS: true,
        processConditionalComments: true,
        removeEmptyAttributes: true,
        removeRedundantAttributes: true,
        trimCustomFragments: true,
        useShortDoctype: true,
        preserveLineBreaks: false,
        collapseWhitespace: true,
      },
    },
    optimization: {
      splitChunks: {
        chunks: 'all',
        automaticNameDelimiter: '.',
        name: 'vic2i',
        minSize: 20000,
        minChunks: 1,
        maxAsyncRequests: 30,
        maxInitialRequests: 30,
        enforceSizeThreshold: 50000,
        cacheGroups: {
          styles: {
            name: 'styles',
            test: /\.(css|vue)$/,
            chunks: 'all',
            enforce: true,
          },
          defaultVendors: {
            test: /[\\/]node_modules[\\/]/,
            priority: -10,
            reuseExistingChunk: true,
          },
          default: {
            minChunks: 2,
            priority: -20,
            reuseExistingChunk: true,
          },
        },
      },
    },
    extend(config, ctx) {},
  },
}
