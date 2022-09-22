/* eslint-disable no-console */
export default function ({ $axios, $config }, inject) {
  let api = null

  if (process.env.NODE_ENV !== 'production') {
    api = $axios
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

    if (process.client) {
      api.setBaseURL('http://127.0.0.1:5000/api')
      inject('api', api)
      return
    }
  } else {
    // Create a custom axios instance
    api = $axios.create({
      headers: {
        common: {
          Accept: 'text/plain, */*',
        },
      },
    })
  }
  
  // Set baseURL
  api.setBaseURL(
    process.env.NUXT_BASE_API_URL ||
    $config.NUXT_BASE_API_URL ||
    'http://127.0.0.1:5000/api'
  )
  
  inject('api', api)
}