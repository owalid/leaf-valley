<template>
  <div>
    <v-row justify="center" align="center">
      <p>Hello world</p>
    </v-row>
    <v-row justify="center" align="center">
      Result api: {{ result }}
    </v-row>
  </div>
</template>

<script>
export default {
  name: 'IndexPage',
  async asyncData({$axios}) {
    const res = await $axios.get('/leaf/')
    const {result} = res.data
    return {
      result
    }
  }
}
</script>
