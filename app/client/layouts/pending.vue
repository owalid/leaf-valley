<template>
  <v-app dark>
    <div v-if="alert">
      <v-snackbar :value="!!alert" right top dense color="error">
        {{ alert }}
      </v-snackbar>
    </div>
    <v-main>
      <nuxt />
    </v-main>
  </v-app>
</template>

<script>
import { mapGetters } from 'vuex'
export default {
  name: 'PendingLayout',
  computed: {
    ...mapGetters({
      alert: 'alert',
    }),
  },
}
</script>
