/* eslint-disable no-console */
export default function ({ $axios }, inject) {
  // Create a custom axios instance
  const econome = $axios.create({
    headers: {
      common: {
        Accept: 'text/plain, */*',
      },
    },
  })

  // Set baseURL
  econome.setBaseURL(
    process.env.NUXT_ECONOME_MS_URL || 'http://127.0.0.1:8080/econome'
  )

  if (process.env.NODE_ENV !== 'production') {
    econome.onResponse((response) => {
      console.log(`[${response.status}] ${response.request.path}`)
      console.log(response)
    })

    econome.onError((err) => {
      console.log(
        `[${err.response && err.response.status}] ${
          err.response && err.response.request.path
        }`
      )
      console.log(err.response && err.response.data)
    })
  }
  inject('econome', econome)
}
