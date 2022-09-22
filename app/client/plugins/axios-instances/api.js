/* eslint-disable no-console */
export default function ({ $axios, $config }, inject) {
  // Create a custom axios instance
  const api = $axios.create({
    headers: {
      common: {
        Accept: 'text/plain, */*',
      },
    },
  })

  api.proxy = true
  // Set baseURL
  api.setBaseURL(
    process.env.NUXT_BASE_API_URL ||
      $config.NUXT_BASE_API_URL ||
      'http://127.0.0.1:5000/api'
  )

  if (process.env.NODE_ENV !== 'production') {
    api.onResponse((response) => {
      console.log(`[${response.status}] ${response.request.path}`)
      console.log(response)
    })

    api.onError((err) => {
      console.log(
        `[${err.response && err.response.status}] ${
          err.response && err.response.request.path
        }`
      )
      console.log(err.response && err.response.data)
    })
  }
  inject('api', api)
}
