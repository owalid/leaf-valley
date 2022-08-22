<template>
  <v-app dark>
    <h1 v-if="error.statusCode === 404">
      {{ pageNotFound }}
    </h1>
    <h1 v-else>
      {{ otherError }}
    </h1>
    <div v-if="alert">
      <v-snackbar right left dense type="error">
        {{ alert }}
      </v-snackbar>
    </div>
    <NuxtLink to="/"> Home page </NuxtLink>
  </v-app>
</template>

<script>
import { mapGetters } from 'vuex'
export default {
  name: 'EmptyLayout',
  layout: 'empty',
  props: {
    error: {
      type: Object,
      default: null,
    },
  },
  data() {
    return {
      pageNotFound: '404 Not Found',
      otherError: 'An error occurred',
    }
  },
  head() {
    const title =
      this.error.statusCode === 404 ? this.pageNotFound : this.otherError
    return {
      title,
    }
  },
  computed: {
    ...mapGetters({
      alert: 'alert',
    }),
  },
}
</script>

<style scoped>
h1 {
  font-size: 20px;
}
</style>
