// ~/plugin/api.js
export default function ({ $axios }, inject) {
  // Create a custom axios instance
  const api = $axios.create({
    headers: {
      common: {
        Accept: 'text/plain, */*'
      }
    }
  })

  // Set baseURL to something different
  api.setBaseURL(process.env.NUXT_BASE_API_URL || 'http://127.0.0.1:5000/api')

  api.onResponse(response => {
    console.log(`[${response.status}] ${response.request.path}`);
    console.log(response)
  });

  api.onError(err => {
    console.log(`[${err.response && err.response.status}] ${err.response && err.response.request.path}`);
    console.log(err.response && err.response.data);
  })

  // Inject to context as $api
  inject('api', api)
}