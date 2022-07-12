<template>
  <v-app dark>
    <div class="px-15 ma-6">
      <v-bottom-navigation color="green lighten-1" height="50px">
        <v-spacer></v-spacer>
        <v-toolbar-items>
          <v-btn v-for="item in menuItems" :key="item.title" :to="item.path">
            <span>{{ item.title }}</span>
          </v-btn>
        </v-toolbar-items>
      </v-bottom-navigation>
      </div>

    <v-main>
      <nuxt />
    </v-main>
  </v-app>
</template>

<script>
export default {
  name: 'DefaultLayout',
  data() {
    return {
      menuItems: [
        { title: 'Home', path: '/'},
        { title: 'Select image', path: '/select'},
        { title: 'Random images', path: '/plants'},
      ]
    }
  },
}
</script>
