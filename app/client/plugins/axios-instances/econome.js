// ~/plugin/api.js
export default function ({ $axios }, inject) {
  // Create a custom axios instance
  const econome = $axios.create({
    headers: {
      common: {
        Accept: 'text/plain, */*'
      }
    }
  })

  // Set baseURL to something different
  econome.setBaseURL(process.env.NUXT_ECONOME_MS_URL || 'http://127.0.0.1:8080/econome')


  econome.onResponse(response => {
    console.log(`[${response.status}] ${response.request.path}`);
  });

  econome.onError(err => {
    console.log(`[${err.response && err.response.status}] ${err.response && err.response.request.path}`);
    console.log(err.response && err.response.data);
  })

  // Inject to context as $api
  inject('econome', econome)
}